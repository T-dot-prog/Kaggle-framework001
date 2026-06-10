import json
import logging
import os

import pandas as pd
import numpy as np

from multi_agents.agents.developer import DeveloperAgent
from multi_agents.agents.reviewer import ReviewerAgent
from multi_agents.agents.planner import PlannerAgent

from multi_agents.exceptions import PhaseFailedError

from multi_agents.state import State

from multi_agents.tools.ml_tools import (
    handle_missing_values, detect_outliers, scale_features,
    encode_categorical, create_polynomial_features,
    select_features_by_variance, split_data, train_model, evaluate_model,
)

logger = logging.getLogger(__name__)


def _exec_code(code, df_train, target_col=""):
    exec_globals = {
        "pd": pd,
        "np": np,
        "df_train": df_train,
        "handle_missing_values": handle_missing_values,
        "detect_outliers": detect_outliers,
        "scale_features": scale_features,
        "encode_categorical": encode_categorical,
        "create_polynomial_features": create_polynomial_features,
        "select_features_by_variance": select_features_by_variance,
        "split_data": split_data,
        "train_model": train_model,
        "evaluate_model": evaluate_model,
        "__builtins__": __builtins__,
    }
    exec_locals = {}
    try:
        exec(code, exec_globals, exec_locals)
        return exec_locals.get("df_train", df_train), None
    except Exception as e:
        return df_train, str(e)


def run_phase(state: State, phase: str, max_attempts=3) -> State:
    logger.info("=" * 60)
    logger.info("Starting phase: %s (max %d attempts)", phase, max_attempts)
    logger.info("=" * 60)

    output_path = os.path.join(os.getcwd(), "artifacts", f"{phase}_data.csv")

    dev = DeveloperAgent()
    reviewer = ReviewerAgent()

    for attempt in range(1, max_attempts + 1):
        logger.info("─" * 50)
        logger.info("Attempt %d/%d for phase '%s'", attempt, max_attempts, phase)
        logger.info("─" * 50)

        code_result = dev.run(state, phase)

        if not code_result["success"] and code_result["error"]:
            logger.warning("Developer code generation failed: %s", code_result["error"][:200])
            exec_result = {"success": False, "stdout": "", "stderr": code_result["error"]}
        else:
            df_train, error = _exec_code(code_result["code"], state.load_train_df(), state.competition_summary.get("target_column", "") if isinstance(state.competition_summary, dict) else "")
            exec_result = {
                "success": error is None,
                "stdout": "",
                "stderr": error or "",
            }
            state.df_train = df_train

        review = reviewer.review(
            state=state,
            phase=phase,
            code=code_result.get("code", ""),
            execution_output=exec_result.get("stdout", ""),
            error=exec_result.get("stderr", "") or code_result.get("error", ""),
        )

        state.decision_memory.append({
            "agent": "Developer+Reviewer",
            "action": f"attempt {attempt} for {phase}",
            "detail": {
                "attempt": attempt,
                "code_success": code_result["success"],
                "exec_success": exec_result["success"],
                "review_score": review["score"],
                "review_acceptable": review["is_acceptable"],
                "suggestions": review.get("suggestions", ""),
            },
        })

        if review["is_acceptable"] and exec_result["success"]:
            final_code = code_result["code"]
            code_path = os.path.join(os.getcwd(), "generated_code", f"{phase}_final.py")
            with open(code_path, "w") as f:
                f.write(final_code)
            logger.info("Saved final code to %s", code_path)

            rationale = code_result.get("rationale", "")
            rationale_path = os.path.join(os.getcwd(), "generated_code", f"{phase}_rationale.md")
            with open(rationale_path, "w") as f:
                f.write(_build_rationale_md(state, phase, rationale, review))
            logger.info("Saved rationale to %s", rationale_path)

            if os.path.exists(output_path):
                logger.info("Output saved to %s", output_path)
            else:
                df_train, error = _exec_code(final_code, state.load_train_df(), state.competition_summary.get("target_column", "") if isinstance(state.competition_summary, dict) else "")
                state.df_train = df_train
                df_train.to_csv(output_path, index=False)
                logger.info("Re-executed and saved output to %s", output_path)

            state.generated_code[phase] = code_path
            logger.info("Phase '%s' completed successfully on attempt %d", phase, attempt)
            return state

        else:
            suggestions = review.get("suggestions", "")
            dev.receive_feedback(suggestions)
            logger.info(
                "Attempt %d not acceptable (score=%d). Feedback given.",
                attempt, review["score"],
            )

    # After max_attempts failure: call Planner.revise_plan() + one more retry
    planner = PlannerAgent()
    state = planner.revise_plan(state, phase, state.error_logs)

    for attempt in range(max_attempts + 1, max_attempts + 2):
        logger.info("─" * 50)
        logger.info("Final attempt %d for phase '%s'", attempt, phase)
        logger.info("─" * 50)

        code_result = dev.run(state, phase)

        if not code_result["success"] and code_result["error"]:
            logger.warning("Developer code generation failed in final attempt: %s", code_result["error"][:200])
            exec_result = {"success": False, "stdout": "", "stderr": code_result["error"]}
        else:
            df_train, error = _exec_code(code_result["code"], state.load_train_df(), state.competition_summary.get("target_column", "") if isinstance(state.competition_summary, dict) else "")
            exec_result = {
                "success": error is None,
                "stdout": "",
                "stderr": error or "",
            }
            state.df_train = df_train

        review = reviewer.review(
            state=state,
            phase=phase,
            code=code_result.get("code", ""),
            execution_output=exec_result.get("stdout", ""),
            error=exec_result.get("stderr", "") or code_result.get("error", ""),
        )

        state.decision_memory.append({
            "agent": "Developer+Reviewer",
            "action": f"final attempt {attempt} for {phase}",
            "detail": {
                "attempt": attempt,
                "code_success": code_result["success"],
                "exec_success": exec_result["success"],
                "review_score": review["score"],
                "review_acceptable": review["is_acceptable"],
                "suggestions": review.get("suggestions", ""),
            },
        })

        if review["is_acceptable"] and exec_result["success"]:
            final_code = code_result["code"]
            code_path = os.path.join(os.getcwd(), "generated_code", f"{phase}_final.py")
            with open(code_path, "w") as f:
                f.write(final_code)
            logger.info("Saved final code to %s", code_path)

            rationale = code_result.get("rationale", "")
            rationale_path = os.path.join(os.getcwd(), "generated_code", f"{phase}_rationale.md")
            with open(rationale_path, "w") as f:
                f.write(_build_rationale_md(state, phase, rationale, review))
            logger.info("Saved rationale to %s", rationale_path)

            if os.path.exists(output_path):
                logger.info("Output saved to %s", output_path)
            else:
                df_train, error = _exec_code(final_code, state.load_train_df(), state.competition_summary.get("target_column", "") if isinstance(state.competition_summary, dict) else "")
                state.df_train = df_train
                df_train.to_csv(output_path, index=False)
                logger.info("Re-executed and saved output to %s", output_path)

            state.generated_code[phase] = code_path
            logger.info("Phase '%s' completed successfully on final attempt %d", phase, attempt)
            return state

    # After second failure: mark skipped, return state (don't raise)
    logger.warning("Phase '%s' failed after %d attempts including final attempt. Marking as skipped.", phase, max_attempts + 1)
    state.skipped_phases.append(phase)
    state.save_checkpoint(f"{phase}_skipped")
    return state


def _build_rationale_md(state, phase, rationale, review):
    plan_steps = [
        s for s in (state.execution_plan or [])
        if s.get("phase") == phase
    ]
    lines = [
        f"# {phase.replace('_', ' ').title()} — Decision Rationale",
        "",
        f"**Phase:** {phase}",
        "",
        "## Plan Steps",
        "```json",
        json.dumps(plan_steps, indent=2),
        "```",
        "",
        "## Tools Selected & Why",
        rationale or "*(No rationale provided)*",
        "",
        "## Review Feedback",
        f"- **Score:** {review.get('score', 'N/A')}/100",
        f"- **Acceptable:** {review.get('is_acceptable', 'N/A')}",
        f"- **Suggestions:** {review.get('suggestions', 'None')}",
        f"- **Errors Found:** {', '.join(review.get('errors_found', []))}",
        "",
    ]
    return "\n".join(lines)
