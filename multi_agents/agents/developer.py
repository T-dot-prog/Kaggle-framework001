import json
import logging
import os

import pandas as pd
import numpy as np

from multi_agents.config import MODEL_MAPPINGS, TEMPERATURES
from multi_agents.tool_registry import list_all_tools
from multi_agents.utils import groq_chat

from multi_agents.state import State

logger = logging.getLogger(__name__)


class DeveloperAgent:
    def __init__(self, model_name=None, temperature=None):
        self.model_name = model_name or MODEL_MAPPINGS["developer_model"]
        self.temperature = temperature or TEMPERATURES["developer"]
        self._feedback = ""

    def run(self, state: State, phase: str) -> dict:
        plan = state.execution_plan or []
        plan_steps = [s for s in plan if s.get("phase") == phase or phase in s.get("phase", "")]

        prev_attempts = state.get_code_attempts(phase)
        all_tools = list_all_tools()
        tool_names = list(all_tools.keys())

        df_preview = state.get_train_df(num_rows=5)
        preview_str = df_preview.to_string() if df_preview is not None else "No data"
        shape_str = str(df_preview.shape) if df_preview is not None else "N/A"
        dtypes_str = df_preview.dtypes.to_string() if df_preview is not None else "N/A"

        target_col = ""
        summary = state.competition_summary or {}
        if isinstance(summary, dict):
            target_col = summary.get("target_column", "")

        messages = self._build_prompt(
            phase, plan_steps, tool_names, preview_str, shape_str, dtypes_str, prev_attempts, target_col
        )

        try:
            raw = groq_chat(
                messages=messages,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=4096,
            )

            rationale, code = self._split_response(raw)
            if not code:
                return {"code": "", "rationale": rationale, "success": False, "error": "Empty code returned"}

            code = self._sanitize_code(code)
            success = True
            error_msg = None

            if success:
                logger.info("DeveloperAgent: phase '%s' code generated successfully", phase)
            else:
                logger.warning("DeveloperAgent: phase '%s' code generation failed: %s", phase, error_msg or "Unknown error")

            return {
                "code": code,
                "rationale": rationale,
                "success": success,
                "output": "",
                "error": error_msg or "",
            }

        except Exception as e:
            logger.error("DeveloperAgent.run failed: %s", e)
            return {"code": "", "rationale": "", "success": False, "output": "", "error": str(e)}

    def receive_feedback(self, feedback: str):
        self._feedback = feedback

    def _build_prompt(self, phase, plan_steps, tool_names, preview_str, shape_str, dtypes_str, prev_attempts, target_col=""):
        PHASE_INSTRUCTIONS = {
            "data_cleaning": "Handle missing values and outliers. Use: handle_missing_values, detect_outliers. Save the cleaned df.",
            "feature_engineering": "Encode categoricals, scale features, create polynomial features. Use: encode_categorical, scale_features, create_polynomial_features, select_features_by_variance. Save the transformed df.",
            "modeling": "Train and evaluate a model. Use: split_data (with target_col='Survived' or the competition's target), train_model, evaluate_model. Do NOT call handle_missing_values or detect_outliers. Save only the evaluation metrics dict to output_path as CSV.",
        }
        phase_inst = PHASE_INSTRUCTIONS.get(phase, "")

        system_prompt = (
            "You are an expert data scientist writing Python code for a Kaggle competition. "
            "You MUST use the provided tool functions from tools.ml_tools.\n"
            f"Your task for the '{phase}' phase: {phase_inst}\n\n"
            "Available tool functions (both `tools.func()` and `func()` work):\n"
            "- handle_missing_values(df, strategy, ..., columns) -> (df, aux)\n"
            "    strategy: 'mean' | 'median' | 'mode' | 'constant' | 'drop' | 'ffill' | 'bfill'\n"
            "- detect_outliers(df, method, ..., columns) -> (df, outlier_mask)\n"
            "    method: 'iqr' | 'zscore' (NOT 'z-score', NOT 'isolation forest')\n"
            "- scale_features(df, method, columns) -> (df, scaler)\n"
            "    method: 'standard' | 'minmax' | 'robust'\n"
            "- encode_categorical(df, method, columns) -> (df, mapping)\n"
            "    method: 'onehot' (NOT 'one-hot') | 'label' | 'ordinal' | 'target'\n"
            "- create_polynomial_features(df, degree, columns) -> (df, aux)\n"
            "- select_features_by_variance(df, threshold, columns) -> (df, info)\n"
            "- split_data(df, target_col, test_size) -> ((X_train, X_test, y_train, y_test), aux)\n"
            "    IMPORTANT: unpack as: (X_train, X_test, y_train, y_test), _ = split_data(...)\n"
            "- train_model(X_train, y_train, model_type) -> (model, aux)\n"
            "    model_type: 'random_forest' | 'xgboost' | 'logistic_regression' | 'linear_regression'\n"
            "- evaluate_model(model, X_test, y_test) -> (metrics_dict, preds_dict)\n\n"
            "All tools return (result, aux) tuple. ALWAYS unpack with destructuring: result, _ = func(...) or (a, b, ...), _ = func(...).\n\n"
            "Output format: rationale (2-3 sentences) then a line with '---CODE---' then ONLY Python code. "
            "Do NOT use !pip, import, or raw sklearn/scipy code.\n\n"
            "The environment already has df_train loaded as a pandas DataFrame.\n"
            "Your code will be executed in-process. Do NOT import anything.\n"
            "Just write the transformation code. The final df_train will be captured.\n\n"
            "Example:\n"
            "df_train, _ = handle_missing_values(df_train, strategy='median')\n"
            "df_train, _ = encode_categorical(df_train, method='label')\n"
            "Save results using df_train.to_csv(output_path, index=False) (or for modeling, save metrics dict as CSV)."
        )

        prev_attempts_str = ""
        if prev_attempts:
            prev_attempts_str = "## Previous Attempts (avoid these mistakes)\n"
            for i, att in enumerate(prev_attempts[-3:], 1):
                prev_attempts_str += (
                    f"Attempt {i}:\n"
                    f"  Error: {att.get('error', 'N/A')[:300]}\n"
                    f"  Feedback: {att.get('feedback', 'N/A')[:300]}\n\n"
                )

        feedback_str = ""
        if self._feedback:
            feedback_str = f"## Feedback from Previous Attempt\n{self._feedback}\n\n"

        repeated_not_defined = False
        if prev_attempts:
            not_defined_count = sum(1 for a in prev_attempts if "is not defined" in a.get("error", ""))
            repeated_not_defined = not_defined_count >= 2 and not_defined_count == len(prev_attempts)

        diagnostic_warning = ""
        if repeated_not_defined:
            diagnostic_warning = (
                "## IMPORTANT: Repeated 'not defined' Errors Detected\n"
                "Your code is failing because function names are not recognized.\n"
                "The tool functions (handle_missing_values, encode_categorical, etc.)\n"
                "are ALREADY available in the execution environment. Do NOT define them.\n"
                "Do NOT use tools.func() syntax. Just call the function directly.\n"
                "Example: df_train, _ = handle_missing_values(df_train, strategy='median')\n\n"
            )

        user_prompt = (
            f"## Phase\n{phase}\n\n"
            f"## Plan Steps\n{json.dumps(plan_steps, indent=2)}\n\n"
            f"## Target Column\n{target_col}\n\n"
            f"## Available Tools\n{tool_names}\n\n"
            f"## Data Preview (first 5 rows)\n{preview_str}\n\n"
            f"## DataFrame Shape\n{shape_str}\n\n"
            f"## Column Dtypes\n{dtypes_str}\n\n"
            f"{prev_attempts_str}"
            f"{feedback_str}"
            f"{diagnostic_warning}"
            "Generate the rationale and code now."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _split_response(raw):
        if not raw:
            return "", ""
        if "---CODE---" in raw:
            parts = raw.split("---CODE---", 1)
            rationale = parts[0].strip()
            code = parts[1].strip()
            code = DeveloperAgent._clean_code(code)
            return rationale, code
        code = DeveloperAgent._clean_code(raw)
        return "", code

    @staticmethod
    def _clean_code(code):
        if code.startswith("```"):
            code = code.split("\n", 1)[-1]
        if code.endswith("```"):
            code = code.rsplit("```", 1)[0]
        code = code.strip()
        lines = [l for l in code.split("\n") if not l.strip().startswith("```")]
        return "\n".join(lines).strip()

    @staticmethod
    def _sanitize_code(code):
        lines = code.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("!") or stripped.startswith("pip install"):
                continue
            if stripped.startswith("import ") and any(
                lib in stripped for lib in ["os", "pandas", "numpy", "logging", "sklearn", "xgboost", "imblearn"]
            ):
                continue
            if stripped.replace(" ", "").startswith("fromtools") or stripped.replace(" ", "").startswith("importtools"):
                continue
            cleaned.append(line)
        return "\n".join(cleaned)
