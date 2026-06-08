import os
import time
import logging
from groq import Groq

from config import GROQ_API_KEY, GROQ_BASE_URL, MODEL_MAPPINGS


def load_api_key():
    api_key_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api_key.txt")
    key = os.environ.get("GROQ_API_KEY")
    base_url = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    if key:
        return key, base_url
    if os.path.exists(api_key_path):
        with open(api_key_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("api_key") and "=" in line:
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                elif line.startswith("base_url") and "=" in line:
                    base_url = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key, base_url


def ensure_directories():
    root = os.getcwd()
    dirs = [
        os.path.join(root, "multi_agents", "competition"),
        os.path.join(root, "multi_agents", "experiments_history"),
        os.path.join(root, "checkpoints"),
        os.path.join(root, "logs"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)

    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def groq_chat(messages, model=None, temperature=0.2, tools=None):
    if model is None:
        model = MODEL_MAPPINGS["planner_model"]

    client = Groq(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools

    last_exception = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(**kwargs)
            msg = response.choices[0].message
            if tools and msg.tool_calls:
                return msg
            return msg.content if msg else ""
        except Exception as e:
            last_exception = e
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise last_exception
