from __future__ import annotations

import json
import logging
import os
import time

from utils import ensure_kb_directory

logger = logging.getLogger(__name__)

KB_DIR = os.path.join(os.getcwd(), "kb")
REGISTRY_PATH = os.path.join(KB_DIR, "tool_success_rates.json")

_registry = None


def _load_registry():
    global _registry
    ensure_kb_directory()
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH) as f:
                _registry = json.load(f)
        except (json.JSONDecodeError, OSError):
            _registry = {}
    else:
        _registry = {}
    _save_registry()


def _save_registry():
    ensure_kb_directory()
    with open(REGISTRY_PATH, "w") as f:
        json.dump(_registry, f, indent=2)


def register_tool(name, tags):
    if _registry is None:
        _load_registry()
    if name not in _registry:
        _registry[name] = {
            "times_used": 0,
            "times_succeeded": 0,
            "last_used": None,
            "tags": tags,
        }
        _save_registry()
        logger.debug("Registered tool '%s' with tags %s", name, tags)
    else:
        existing_tags = _registry[name].get("tags", [])
        for t in tags:
            if t not in existing_tags:
                existing_tags.append(t)
        _registry[name]["tags"] = existing_tags
        _save_registry()


def record_outcome(name, success):
    if _registry is None:
        _load_registry()
    if name not in _registry:
        logger.warning("Tool '%s' not registered; creating record", name)
        _registry[name] = {
            "times_used": 0,
            "times_succeeded": 0,
            "last_used": None,
            "tags": [],
        }
    _registry[name]["times_used"] += 1
    if success:
        _registry[name]["times_succeeded"] += 1
    _registry[name]["last_used"] = time.time()
    _save_registry()


def get_best_tools_for_task(task_description: str, top_n:int =3):
    if _registry is None:
        _load_registry()
    if not _registry:
        return []

    task_lower = task_description.lower()
    task_words = set(task_lower.split())

    scored = []
    for name, record in _registry.items():
        tags = record.get("tags", [])
        tag_lower = [t.lower() for t in tags]
        keyword_matches = sum(1 for w in task_words if any(w in t for t in tag_lower))
        tag_overlap = len(set(task_lower.split()) & set(" ".join(tag_lower).split()))

        used = record["times_used"]
        succeeded = record["times_succeeded"]
        if used > 0:
            success_rate = succeeded / used
        else:
            success_rate = 0.5

        score = (keyword_matches + tag_overlap) * success_rate
        scored.append((score, name))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [name for _, name in scored[:top_n]]


def list_all_tools():
    if _registry is None:
        _load_registry()
    return dict(_registry)


_load_registry()
