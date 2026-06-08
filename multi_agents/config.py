import os

txt_api_path = os.path.join(os.path.dirname(__file__), "api_key.txt")

def _load_config() -> tuple[any, any]:
    if not os.path.exists(txt_api_path):
        print("Configure api key and base url")
        return None, None
    else:
        with open(txt_api_path, "r") as f:
            content = f.readlines()
            key = content[0].strip()
            base_url = content[1].strip()

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
    from groq import Groq
    return Groq(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
