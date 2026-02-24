"""Configuration for the intent preservation experiments."""
# Note from Praggnya: I went through all the files in main and reviewed their code. 
# As I reviewed their code, I commented the general code structure to take note of what was happening
# in order to later help with understanding what kind of improvements I can add.
# For this file though, as it is just the config, I did not have much to comment.
import os
import random
import numpy as np
import torch
from datetime import datetime

SEED = 42
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Model configs
MODELS = {
    "gpt-4.1": {
        "api": "openai",
        "model_id": "gpt-4.1",
        "temperature": 0,
    },
    "claude-sonnet-4.5": {
        "api": "openrouter",
        "model_id": "anthropic/claude-sonnet-4.5",
        "temperature": 0,
    },
}

# Experiment parameters
N_SAMPLES_PER_DATASET = 200  # queries per dataset
N_METRIC_VALIDATION = 100    # for metric correlation study
BATCH_SIZE = 20              # API calls per batch

# Correction prompt strategies
CORRECTION_PROMPTS = {
    "fix_errors": (
        "Fix any errors in the following user query. "
        "Return ONLY the corrected query, nothing else. "
        "If there are no errors, return the query unchanged.\n\n"
        "Query: {query}"
    ),
    "rewrite_clearly": (
        "Rewrite the following user query to be clearer and more precise. "
        "Return ONLY the rewritten query, nothing else.\n\n"
        "Query: {query}"
    ),
    "improve": (
        "Improve the following user query to better express the user's intent. "
        "Return ONLY the improved query, nothing else.\n\n"
        "Query: {query}"
    ),
}

# Confidence thresholds for Experiment 3
HIGH_CONFIDENCE_THRESHOLD = 0.80
LOW_CONFIDENCE_THRESHOLD = 0.40

# Paths (absolute, using project root)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATA_DIR = os.path.join(PROJECT_ROOT, "results", "data")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "results", "plots")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

# Directories
DATASET_PATHS = {
    "banking77": os.path.join(PROJECT_ROOT, "datasets", "banking77"),
    "clinc150": os.path.join(PROJECT_ROOT, "datasets", "clinc150"),
}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_experiment_config():
    return {
        "seed": SEED,
        "n_samples_per_dataset": N_SAMPLES_PER_DATASET,
        "models": list(MODELS.keys()),
        "correction_prompts": list(CORRECTION_PROMPTS.keys()),
        "timestamp": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
