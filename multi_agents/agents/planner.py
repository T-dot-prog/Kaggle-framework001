import json
import logging
import traceback

from multi_agents.config import MODEL_MAPPINGS, TEMPERATURES
from multi_agents.tool_registry import get_best_tools_for_task, register_tool, list_all_tools
from multi_agents.utils import groq_chat

from multi_agents.state import State

logger = logging.getLogger(__name__)


class PlannerAgent:
    def __init__(self, model_name=None, temperature=None):
        self.model_name = model_name or MODEL_MAPPINGS["planner_model"]
        self.temperature = temperature or TEMPERATURES["planner"]

    def run(self, state: State):
        logger.info("PlannerAgent: generating execution plan...")

        try:
            summary = state.competition_summary
            if isinstance(summary, str):
                import json as _json
                summary = _json.loads(summary) if summary else {}

            data_cleaning_tools = get_best_tools_for_task("data cleaning handle missing values outliers")
            feature_eng_tools = get_best_tools_for_task("feature engineering encode scale categorical")
            modeling_tools = get_best_tools_for_task("modeling train evaluate classification regression")

            from tool_registry import _registry
            all_tool_names = list(_registry.keys()) if _registry else []

            system_prompt = (
                "You are a senior data science strategist. "
                "Analyze the competition summary and create a detailed execution plan. "
                "Return ONLY a valid JSON object with no additional text. "
                "The JSON must have these keys:\n"
                '  "plan": a list of phase objects, each with:\n'
                '    "phase": string ("data_cleaning", "feature_engineering", or "modeling")\n'
                '    "tools_needed": list of tool names — each MUST be one of the available tool names listed below\n'
                '    "expected_outputs": string describing what this phase produces\n'
                '  "modeling_approach": string (e.g. "RandomForest", "XGBoost")\n'
                '  "validation_strategy": string (e.g. "k-fold cross-validation")\n\n'
                "CRITICAL: tools_needed MUST be chosen EXACTLY from this list. "
                "Do NOT invent names. Do NOT use generic library names. "
                "Only use these: " + json.dumps(all_tool_names) + "\n\n"
                "If no tool matches a task, use 'implement_from_scratch' as a fallback.\n\n"
                "Base your plan on the competition summary below."
            )

            user_prompt = (
                f"## Competition Summary\n{json.dumps(summary, indent=2)}\n\n"
                f"## Recommended Tools for Data Cleaning\n{json.dumps(data_cleaning_tools)}\n\n"
                f"## Recommended Tools for Feature Engineering\n{json.dumps(feature_eng_tools)}\n\n"
                f"## Recommended Tools for Modeling\n{json.dumps(modeling_tools)}\n\n"
                "Create an execution plan for this competition. "
                "For each phase, pick tools_needed from the recommended tools above or the full available tools list. "
                "Do NOT use generic library names."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            raw = groq_chat(
                messages=messages,
                model=self.model_name,
                temperature=self.temperature,
            )

            parsed = self._parse_json(raw)
            if parsed is None:
                raise ValueError("Failed to parse LLM response as JSON")

            plan = parsed.get("plan", [])
            if not plan:
                raise ValueError("Plan is empty")

            for step in plan:
                tools = step.get("tools_needed", [])
                for tool_name in tools:
                    all_tools = list_all_tools()
                    if tool_name not in all_tools:
                        logger.info("Tool '%s' not in registry; auto-registering", tool_name)
                        register_tool(tool_name, [step["phase"], "auto_detected"])

            state.execution_plan = plan

            extra = {
                "modeling_approach": parsed.get("modeling_approach", ""),
                "validation_strategy": parsed.get("validation_strategy", ""),
            }

            state.decision_memory.append({
                "agent": "PlannerAgent",
                "action": "generated execution plan",
                "detail": {"plan": plan, **extra},
            })

            state.save_checkpoint("planner_complete")

            logger.info("PlannerAgent: plan generated successfully")
            logger.info("Plan:\n%s", json.dumps(parsed, indent=2))

        except Exception as e:
            logger.error("PlannerAgent failed: %s", e)
            logger.error(traceback.format_exc())
            state.error_logs.append({
                "agent": "PlannerAgent",
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

        return state

    def revise_plan(self, state: State, failed_phase, error_logs):
        logger.info("PlannerAgent: revising plan after failure in '%s'", failed_phase)
        try:
            summary = state.competition_summary
            if isinstance(summary, str):
                import json as _json
                summary = _json.loads(summary) if summary else {}

            revision_prompt = (
                f"The phase '{failed_phase}' failed with these errors:\n"
                f"{json.dumps(error_logs, indent=2)}\n\n"
                f"Original plan:\n{json.dumps(state.execution_plan, indent=2)}\n\n"
                "Provide a revised plan focusing on the failed phase. "
                "Return ONLY valid JSON with the same structure as the original plan."
            )

            messages = [
                {"role": "system", "content": "You are a data science strategist revising a failed plan."},
                {"role": "user", "content": revision_prompt},
            ]

            raw = groq_chat(
                messages=messages,
                model=self.model_name,
                temperature=self.temperature,
            )

            parsed = self._parse_json(raw)
            if isinstance(parsed, dict):
                revised_plan = parsed.get("plan")
            elif isinstance(parsed, list):
                revised_plan = parsed
            else:
                revised_plan = None

            if revised_plan:
                state.execution_plan = revised_plan
                state.decision_memory.append({
                    "agent": "PlannerAgent",
                    "action": "revised plan after failure",
                    "detail": {"failed_phase": failed_phase, "revised_plan": revised_plan},
                })

        except Exception as e:
            logger.error("revise_plan failed: %s", e)

        return state

    @staticmethod
    def _parse_json(raw):
        if not raw:
            return None
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            if "```" in cleaned:
                cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        try:
            import re
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError:
            pass
        return None
