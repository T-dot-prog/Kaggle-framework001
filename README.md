# AutoKaggle

Multi-agent system for automated Kaggle competition pipeline.

## Setup

### 1. Create environment

```bash
conda create -n AutoKaggle python=3.11 -y
conda activate AutoKaggle
pip install -r requirements.txt
```

### 2. API key

Create `api_key.txt` in the project root with your Groq API key:

```
api_key = "gsk_your_groq_api_key_here"
base_url = "https://api.groq.com/openai/v1"
```

You can get a free API key at https://console.groq.com.

Alternatively, set environment variables:

```bash
export GROQ_API_KEY="gsk_your_groq_api_key_here"
export GROQ_BASE_URL="https://api.groq.com/openai/v1"
```

### 3. Competition data

Place competition files in the required directory structure:

```
multi_agents/competition/<competition_name>/
├── train.csv
├── test.csv
├── sample_submission.csv
└── overview.txt
```

`overview.txt` should contain the competition description copied from Kaggle.

### 4. Run

```bash
bash run_multi_agent.sh
```

Override parameters via environment variables:

```bash
export competitions=("titanic" "spaceship_titanic")
export start_run=1 end_run=3 model="llama-3.1-8b-instant"
bash run_multi_agent.sh
```

## Outputs

Results are saved to:

```
multi_agents/experiments_history/<competition>/<model>/<dest_dir>/<run>/
```
