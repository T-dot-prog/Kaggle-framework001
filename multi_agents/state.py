from __future__ import annotations

import os
import json
import time

import pandas as pd


class State:
    def __init__(self, phase=1, competition_path=None, output_dir=None):
        self.phase = phase
        self.competition_path = competition_path or os.path.join(
            os.getcwd(), "multi_agents", "competition"
        )
        self.output_dir = output_dir or os.path.join(
            os.getcwd(), "multi_agents", "experiments_history"
        )

        self.decision_memory = []
        self.code_attempts = []
        self.competition_summary = ""
        self.execution_plan = []
        self.generated_code = {}
        self.error_logs = []
        self.skipped_phases = []
        self.df_train: pd.DataFrame | None = None
        self._current_train_csv: str | None = None

    @property
    def train_csv(self):
        return os.path.join(self.competition_path, "train.csv")

    @property
    def test_csv(self):
        return os.path.join(self.competition_path, "test.csv")

    @property
    def sample_sub_csv(self):
        return os.path.join(self.competition_path, "sample_submission.csv")

    @property
    def overview_txt(self):
        return os.path.join(self.competition_path, "overview.txt")

    @property
    def comp_dir(self):
        return self.competition_path

    def add_code_attempt(self, code_snippet, error_if_any, success, feedback, phase=""):
        self.code_attempts.append({
            "code": code_snippet,
            "error": error_if_any,
            "success": success,
            "feedback": feedback,
            "phase": phase,
            "timestamp": time.time(),
        })

    def get_code_attempts(self, phase):
        return [a for a in self.code_attempts if a.get("phase") == phase]

    def append_error_log(self, error):
        if isinstance(error, str):
            self.error_logs.append({"error": error, "timestamp": time.time()})
        elif isinstance(error, dict):
            self.error_logs.append(error)

    def load_train_df(self):
        """Load train.csv into self.df_train if not loaded."""
        if self.df_train is None:
            self.df_train = pd.read_csv(self.train_csv)
        return self.df_train

    def get_train_df(self, num_rows=None):
        try:
            df = pd.read_csv(self.train_csv)
            if num_rows is not None:
                df = df.head(num_rows)
            return df
        except FileNotFoundError:
            print(f"Warning: train.csv not found at {self.train_csv}")
            return None

    @property
    def current_train_csv(self):
        if self._current_train_csv and os.path.exists(self._current_train_csv):
            return self._current_train_csv
        return self.train_csv

    def set_current_train_csv(self, path):
        self._current_train_csv = path

    def ensure_checkpoint_dir(self):
        path = os.path.join(os.getcwd(), "checkpoints")
        os.makedirs(path, exist_ok=True)
        return path

    def save_checkpoint(self, phase_name):
        self.ensure_checkpoint_dir()
        path = os.path.join(os.getcwd(), "checkpoints", f"{phase_name}.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)
        print(f"Checkpoint saved: {path}")

    @classmethod
    def load_checkpoint(cls, phase_name):
        path = os.path.join(os.getcwd(), "checkpoints", f"{phase_name}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            return cls.from_dict(data)
        return None

    def to_dict(self):
        d = {
            "phase": self.phase,
            "competition_path": self.competition_path,
            "output_dir": self.output_dir,
            "decision_memory": self.decision_memory,
            "code_attempts": self.code_attempts,
            "competition_summary": self.competition_summary,
            "execution_plan": self.execution_plan,
            "generated_code": self.generated_code,
            "error_logs": self.error_logs,
            "skipped_phases": self.skipped_phases,
        }
        d.pop("df_train", None)
        return d

    @classmethod
    def from_dict(cls, data):
        state = cls(
            phase=data["phase"],
            competition_path=data["competition_path"],
            output_dir=data["output_dir"],
        )
        state.decision_memory = data.get("decision_memory", [])
        state.code_attempts = data.get("code_attempts", [])
        state.competition_summary = data.get("competition_summary", "")
        state.execution_plan = data.get("execution_plan", [])
        state.generated_code = data.get("generated_code", {})
        state.error_logs = data.get("error_logs", [])
        state.skipped_phases = data.get("skipped_phases", [])
        state.df_train = None
        state._current_train_csv = None
        return state
