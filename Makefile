# ==========================================
# Makefile- Suggested as Improvement
# ==========================================

VENV=.venv
PYTHON=python

# ------------------------------------------
# Initial setup related commands
# ------------------------------------------

setup:
	uv venv
	source $(VENV)/bin/activate && \
	uv add datasets numpy scipy scikit-learn matplotlib seaborn tqdm sentence-transformers torch bert-score

# ------------------------------------------
# For downloading datasets
# ------------------------------------------

datasets-list:
	cd datasets && $(PYTHON) download_datasets.py --list

datasets-high:
	cd datasets && $(PYTHON) download_datasets.py --priority HIGH

datasets-all:
	cd datasets && $(PYTHON) download_datasets.py --all

datasets-ids:
	cd datasets && $(PYTHON) download_datasets.py --ids $(IDS)

data-setup: datasets-high

# ------------------------------------------
# This Makefile only reflects what I was able to run!
# ------------------------------------------

classifier:
	cd src && $(PYTHON) intent_classifier.py

analyze:
	cd src && $(PYTHON) analyze_results.py

quick:
	cd src && $(PYTHON) intent_classifier.py

clean:
	rm -rf results/data/*
	rm -rf results/plots/*
