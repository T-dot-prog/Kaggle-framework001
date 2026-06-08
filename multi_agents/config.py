import os

API_KEY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api_key.txt")

def _load_config():
    key = os.environ.get("GROQ_API_KEY")
    base_url = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    if key:
        return key, base_url
    if os.path.exists(API_KEY_PATH):
        with open(API_KEY_PATH) as f:
            for line in f:
                line = line.strip()
                if line.startswith("api_key") and "=" in line:
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                elif line.startswith("base_url") and "=" in line:
                    base_url = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key, base_url

GROQ_API_KEY, GROQ_BASE_URL = _load_config()

MODEL_MAPPINGS = {
    "planner_model": "llama-3.1-8b-instant",
    "developer_model": "llama-3.1-8b-instant",
    "reader_model": "llama-3.1-8b-instant",
    "reviewer_model": "llama-3.1-8b-instant",
    "summarizer_model": "llama-3.1-8b-instant",
}

TEMPERATURES = {
    "reader": 0.3,
    "planner": 0.4,
    "developer": 0.2,
    "reviewer": 0.2,
    "summarizer": 0.3,
}


def get_groq_client():
    from openai import OpenAI
    return OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
