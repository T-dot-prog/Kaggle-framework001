import json
import logging
from typing import Any, Dict

from multi_agents.config import MODEL_MAPPINGS, TEMPERATURES
from multi_agents.state import State
from multi_agents.utils import groq_chat, parse_json_response

logger = logging.getLogger(__name__)


class ReviewerAgent:
    def __init__(self, model_name=None, temperature=None):
        self.model_name = model_name or MODEL_MAPPINGS["reviewer_model"]
        self.temperature = temperature or TEMPERATURES["reviewer"]

    def review(
        self,
        state: State,
        phase: str,
        code: str,
        execution_output: str,
        error: str,
    ) -> Dict[str, Any]:
        logger.info("ReviewerAgent: reviewing code for phase '%s'", phase)

        summary = state.competition_summary or {}
        plan_steps = [
            s for s in (state.execution_plan or []) if s.get("phase") == phase
        ]

        system_prompt = (
            "You are a senior code reviewer for data science pipelines. "
            "Evaluate the generated code and its execution output. "
            "Return ONLY a valid JSON object with these exact keys:\n"
            '  "is_acceptable": true/false (whether code is correct and achieves the goal)\n'
            '  "score": 0-100\n'
            '  "suggestions": "string of specific improvements"\n'
            '  "errors_found": ["list", "of", "issues"]\n\n'
            "Consider:\n"
            "- Code correctness (syntax, imports, logic)\n"
            "- Does it use df_train variable (the in-memory DataFrame)?\n"
            "- Does it handle errors gracefully?\n"
            "- Does it use the expected tools for this phase?\n"
            "- Are transformations appropriate for the competition data?\n"
            "- Check for target leakage or using test data during training\n"
            "- Execution output shows success (no crash, expected results)\n"
            "Return ONLY valid JSON."
        )

        MAX_CODE_LEN = 2000
        MAX_OUTPUT_LEN = 1000

        code_trunc = code[:MAX_CODE_LEN] + (
            "\n# ... truncated" if len(code) > MAX_CODE_LEN else ""
        )
        output_trunc = execution_output[:MAX_OUTPUT_LEN]
        output_trunc += (
            "\n... (truncated)" if len(execution_output) > MAX_OUTPUT_LEN else ""
        )
        error_trunc = error[:MAX_OUTPUT_LEN]
        error_trunc += "\n... (truncated)" if len(error) > MAX_OUTPUT_LEN else ""

        user_prompt = (
            f"## Phase\n{phase}\n\n"
            f"## Competition Summary\n{json.dumps(summary, indent=2)}\n\n"
            f"## Plan Steps for this Phase\n{json.dumps(plan_steps, indent=2)}\n\n"
            f"## Generated Code\n```python\n{code_trunc}\n```\n\n"
            f"## Execution Output\n{output_trunc}\n\n"
            f"## Error (if any)\n{error_trunc}\n\n"
            "Evaluate this code in context of the competition summary and plan steps above."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(2):
            raw = ""
            try:
                raw = groq_chat(
                    messages=messages,
                    model=self.model_name,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                parsed = parse_json_response(raw)
                if parsed is None:
                    raise json.JSONDecodeError("parse failed", raw, 0)

                required = ["is_acceptable", "score", "suggestions", "errors_found"]
                missing = [k for k in required if k not in parsed]
                if missing:
                    raise ValueError(f"Missing fields: {missing}")

                if not isinstance(parsed["is_acceptable"], bool):
                    parsed["is_acceptable"] = parsed["is_acceptable"] in (
                        True,
                        "true",
                        "True",
                        1,
                    )
                parsed["score"] = max(0, min(100, int(parsed.get("score", 0))))

                logger.info(
                    "ReviewerAgent: score=%d, acceptable=%s",
                    parsed["score"],
                    parsed["is_acceptable"],
                )

                state.add_code_attempt(
                    code_snippet=code,
                    error_if_any=error,
                    success=parsed["is_acceptable"],
                    feedback=parsed.get("suggestions", ""),
                    phase=phase,
                )

                return parsed

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    "Reviewer JSON parse failed attempt %d/2: %s", attempt + 1, e
                )
                if attempt == 0:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": raw,
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": "Return ONLY a valid JSON object with is_acceptable, score, suggestions, errors_found.",
                        }
                    )

        fallback: Dict[str, Any] = {
            "is_acceptable": False,
            "score": 0,
            "suggestions": "Reviewer failed to parse response",
            "errors_found": ["Reviewer API error"],
        }
        return fallback
