import json
import logging
import os
import traceback

import pandas as pd

from config import MODEL_MAPPINGS, TEMPERATURES
from utils import groq_chat

from multi_agents.state import State

logger = logging.getLogger(__name__)


class ReaderAgent:
    def __init__(self, model_name=None, temperature=None):
        self.model_name = model_name or MODEL_MAPPINGS["reader_model"]
        self.temperature = temperature or TEMPERATURES["reader"]

    def run(self, state: State):
        logger.info("ReaderAgent: reading competition data...")

        try:
            train_preview = state.get_train_df(num_rows=10)
            test_path = state.test_csv
            test_df = pd.read_csv(test_path) if pd.io.common.file_exists(test_path) else None
            test_preview = test_df.head(10) if test_df is not None else "No test.csv found"

            overview = ""
            if state.overview_txt and os.path.exists(state.overview_txt):
                with open(state.overview_txt) as f:
                    overview = f.read()

            train_str = train_preview.to_string() if train_preview is not None else "No train.csv found"
            test_str = test_preview.to_string() if not isinstance(test_preview, str) else test_preview

            system_prompt = (
                "You are a data analyst analyzing a Kaggle competition. "
                "Return ONLY a valid JSON object with no additional text. "
                "The JSON must have these exact keys:\n"
                '  "problem_type": string (e.g. binary_classification, multiclass_classification, regression)\n'
                '  "target_column": string (name of the target variable)\n'
                '  "evaluation_metric": string (e.g. accuracy, log_loss, rmse)\n'
                '  "key_features": list of strings (most important feature columns)\n'
                '  "data_characteristics": object with keys:\n'
                '    "missing_values": "present" or "absent"\n'
                '    "high_cardinality_categoricals": list of strings\n'
                '    "class_imbalance": "yes" or "no"\n'
                "Base your analysis on the competition overview and data preview provided."
            )

            user_prompt = (
                f"## Competition Overview\n{overview}\n\n"
                f"## Train Data Preview (first 10 rows)\n{train_str}\n\n"
                f"## Test Data Preview (first 10 rows)\n{test_str}\n\n"
                "Analyze this competition and return the JSON summary."
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

            state.competition_summary = parsed

            state.decision_memory.append({
                "agent": "ReaderAgent",
                "action": "generated competition summary",
                "detail": parsed,
            })

            state.save_checkpoint("reader_complete")

            logger.info("ReaderAgent: summary generated successfully")
            logger.info("Summary:\n%s", json.dumps(parsed, indent=2))

        except Exception as e:
            logger.error("ReaderAgent failed: %s", e)
            logger.error(traceback.format_exc())
            state.error_logs.append({
                "agent": "ReaderAgent",
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

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
