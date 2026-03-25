from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
PROMPTS_DIR = BASE_DIR / "prompts"
DATA_DIR = BASE_DIR / "data"

OUTCOME_MODEL_PATH = MODELS_DIR / "outcome_model.joblib"
INJURY_MODEL_PATH = MODELS_DIR / "injury_model.joblib"
COACHING_PROMPT_PATH = PROMPTS_DIR / "coaching_prompt.txt"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)