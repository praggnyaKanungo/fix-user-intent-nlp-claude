"""Main experiment runner for intent preservation evaluation.

Experiment 1: Measures intent violation rates across LLM models and prompting strategies
Experiment 2: Validates automated metrics against LLM-as-judge
Experiment 3: Tests confidence-aware correction strategy
"""
# Note from Praggnya: I went through all the files in main and reviewed their code. 
# As I reviewed their code, I commented the general code structure to take note of what was happening
# in order to later help with understanding what kind of improvements I can add.

# all imports
import os
import sys
import json
import time
import numpy as np
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
from config import (
    set_seed, SEED, MODELS, CORRECTION_PROMPTS, N_SAMPLES_PER_DATASET,
    N_METRIC_VALIDATION, RESULTS_DIR, DATA_DIR, PLOTS_DIR,
    HIGH_CONFIDENCE_THRESHOLD, LOW_CONFIDENCE_THRESHOLD,
    get_experiment_config,
)
from data_loader import sample_queries
from llm_client import call_llm, call_llm_for_judge
from metrics import compute_all_metrics
from intent_classifier import EmbeddingIntentClassifier


# this function runs experiment 1 which is what looks at the classifier disagrements to decide if 
# something is in intent shift or not
def run_experiment_1(samples, model_name, prompt_key, classifier):
    """Run corrections and measure intent preservation.

    Returns list of result dicts for each sample.
    """
    prompt_template = CORRECTION_PROMPTS[prompt_key]
    results = []

    originals = [s["text"] for s in samples]
    rewrites = []

    # Call LLM for each query
    print(f"\n  Calling {model_name} with '{prompt_key}' on {len(samples)} queries...")
    for i, sample in enumerate(tqdm(samples, desc=f"  {model_name}/{prompt_key}")):
        prompt = prompt_template.format(query=sample["text"])
        response = call_llm(model_name, prompt)
        if response is None:
            response = sample["text"]  # fallback: no change

        # here is it getting the rewrites and giving track of them
        rewrites.append(response)

        # Small delay to avoid rate limits
        if (i + 1) % 20 == 0:
            time.sleep(0.5)

    # Compute automated metrics in batch
    print(f"  Computing metrics...")
    metrics = compute_all_metrics(originals, rewrites)

    # Detect intent shifts
    print(f"  Checking intent shifts...")

    # here it calls this function (that must be in another file) but this is the function that checks
    # if the original and rewrites are the same (then no shift) or different (then there is an intention shift)
    shifts = classifier.check_intent_shift(originals, rewrites)

    # Also check against ground truth labels

    # I am assuming they check against ground truth becasue if the claissfied couldnt even classify that, its unreliable
    # question though: this could be used to filter these out, but i don't think that's being done. Why?
    gt_labels = [s["label_id"] for s in samples]
    gt_names = [s["label_name"] for s in samples]
    orig_preds = classifier.classify(originals)

    # this is storing the results
    for i, sample in enumerate(samples):
        # Check if classifier correctly identifies original intent
        classifier_correct_on_original = orig_preds[i]["predicted_label"] == gt_labels[i]

        results.append({
            "dataset": sample["dataset"],
            "index": sample["index"],
            "original": originals[i],
            "rewrite": rewrites[i],
            "gt_intent": gt_names[i],
            "gt_intent_id": gt_labels[i],
            "model": model_name,
            "prompt_strategy": prompt_key,
            "classifier_correct_on_original": classifier_correct_on_original,
            **metrics[i],
            **shifts[i],
        })

    return results


# function for experiment 2 which takes the LLM as the judge
def run_experiment_2(exp1_results, model_name="gpt-4.1"):
    """Validate metrics by using LLM-as-judge.

    Takes a subset of Experiment 1 results and asks an LLM whether
    the rewrite preserves the original intent.
    """
    # Sample from results that have varied outcomes
    shifted = [r for r in exp1_results if r["intent_shifted"]]
    not_shifted = [r for r in exp1_results if not r["intent_shifted"]]

    rng = np.random.RandomState(SEED)

    # here we are balancing the amount of shifted and non shifted (based on experiment 1)
    n_per_group = min(N_METRIC_VALIDATION // 2, len(shifted), len(not_shifted))

    if n_per_group == 0:
        # If no shifts found, just sample randomly
        subset = list(rng.choice(exp1_results, size=min(N_METRIC_VALIDATION, len(exp1_results)), replace=False))
    else:
        subset = (
            list(rng.choice(shifted, size=n_per_group, replace=False)) +
            list(rng.choice(not_shifted, size=n_per_group, replace=False))
        )
    rng.shuffle(subset)

    print(f"\n  Running LLM-as-judge on {len(subset)} examples...")

    system_prompt = (
        "You are an expert evaluator. You will be given an original user query and a rewritten version. "
        "Determine whether the rewrite preserves the user's original intent.\n\n"
        "Respond with EXACTLY one of:\n"
        "- PRESERVED: The rewrite keeps the same intent/meaning\n"
        "- CHANGED: The rewrite alters the user's intent or goal\n"
        "- AMBIGUOUS: It's unclear whether intent was preserved\n\n"
        "Then briefly explain why (1 sentence)."
    )

    judge_results = []

    #
    for r in tqdm(subset, desc="  LLM-as-judge"):
        user_prompt = (
            f"Original query: \"{r['original']}\"\n"
            f"Rewritten query: \"{r['rewrite']}\"\n\n"
            f"Was the user's intent preserved?"
        )
        judge_response = call_llm_for_judge(model_name, system_prompt, user_prompt)

        if judge_response:
            if "PRESERVED" in judge_response.upper():
                judge_label = "preserved"
            elif "CHANGED" in judge_response.upper():
                judge_label = "changed"
            else:
                judge_label = "ambiguous"
        else:
            judge_label = "error"

        # storing the result of GPT said
        # question: is there a reason they decided to use GPT as the judge here?
        judge_results.append({
            **r,
            "judge_response": judge_response,
            "judge_label": judge_label,
        })

        time.sleep(0.3)

    return judge_results


# function for the experiment 3 which is what tests the confident aware strategy
# note: this method was already largely commented so I did not add any comments for myself
def run_experiment_3(samples, classifier, model_name="gpt-4.1"):
    """Test confidence-aware correction strategy.

    Strategy:
    - High confidence (>0.8): Auto-correct using 'fix_errors' prompt
    - Medium confidence (0.4-0.8): Generate clarifying question
    - Low confidence (<0.4): Leave query unchanged

    Compare against:
    - Always-correct: Fix every query
    - Always-clarify: Ask for every query
    - No-action: Leave everything as-is
    """
    originals = [s["text"] for s in samples]

    # Classify to get confidence scores
    print(f"\n  Classifying {len(samples)} queries for confidence...")
    classifications = classifier.classify(originals, k=5)

    # Assign categories
    high_conf = []
    med_conf = []
    low_conf = []

    for i, (sample, clf_result) in enumerate(zip(samples, classifications)):
        conf = clf_result["confidence"]
        if conf >= HIGH_CONFIDENCE_THRESHOLD:
            high_conf.append(i)
        elif conf >= LOW_CONFIDENCE_THRESHOLD:
            med_conf.append(i)
        else:
            low_conf.append(i)

    print(f"  Confidence distribution: high={len(high_conf)}, med={len(med_conf)}, low={len(low_conf)}")

    # Strategy 1: Confidence-aware
    strategy_results = []
    clarification_prompt = (
        "The user asked: \"{query}\"\n"
        "This query could mean several things. Generate a brief, helpful "
        "clarifying question to understand their intent better. "
        "Do NOT rewrite the query. Just ask ONE short clarifying question."
    )

    fix_prompt = CORRECTION_PROMPTS["fix_errors"]

    print(f"  Running confidence-aware strategy...")
    for i, sample in enumerate(tqdm(samples, desc="  confidence-aware")):
        conf = classifications[i]["confidence"]
        action = None
        response = None

        if i in high_conf:
            # Auto-correct
            action = "auto_correct"
            prompt = fix_prompt.format(query=sample["text"])
            response = call_llm(model_name, prompt)
            if response is None:
                response = sample["text"]
        elif i in med_conf:
            # Clarify
            action = "clarify"
            prompt = clarification_prompt.format(query=sample["text"])
            response = call_llm(model_name, prompt)
            if response is None:
                response = "Could you please clarify what you mean?"
        else:
            # Abstain
            action = "abstain"
            response = sample["text"]  # unchanged

        if (i + 1) % 20 == 0:
            time.sleep(0.5)

        strategy_results.append({
            "dataset": sample["dataset"],
            "index": sample["index"],
            "original": sample["text"],
            "gt_intent": sample["label_name"],
            "gt_intent_id": sample["label_id"],
            "confidence": conf,
            "action": action,
            "response": response,
            "model": model_name,
        })

    # For auto-correct actions, measure intent preservation
    auto_indices = [i for i, r in enumerate(strategy_results) if r["action"] == "auto_correct"]
    if auto_indices:
        auto_originals = [strategy_results[i]["original"] for i in auto_indices]
        auto_rewrites = [strategy_results[i]["response"] for i in auto_indices]

        auto_metrics = compute_all_metrics(auto_originals, auto_rewrites)
        auto_shifts = classifier.check_intent_shift(auto_originals, auto_rewrites)

        for j, idx in enumerate(auto_indices):
            strategy_results[idx]["metrics"] = auto_metrics[j]
            strategy_results[idx]["intent_shifted"] = auto_shifts[j]["intent_shifted"]
            strategy_results[idx]["rewrite_intent"] = auto_shifts[j]["rewrite_intent"]

    # For abstain/clarify, intent is trivially preserved
    for r in strategy_results:
        if r["action"] in ("abstain", "clarify"):
            r["intent_shifted"] = False
            r["metrics"] = {"edit_ratio": 0.0} if r["action"] == "abstain" else {}

    # Strategy 2: Always-correct baseline (reuse exp1 results if available)
    print(f"  Running always-correct baseline...")
    always_correct_results = []
    for i, sample in enumerate(tqdm(samples, desc="  always-correct")):
        prompt = fix_prompt.format(query=sample["text"])
        response = call_llm(model_name, prompt)
        if response is None:
            response = sample["text"]
        always_correct_results.append({
            "original": sample["text"],
            "response": response,
            "gt_intent": sample["label_name"],
            "gt_intent_id": sample["label_id"],
            "action": "auto_correct",
        })
        if (i + 1) % 20 == 0:
            time.sleep(0.5)

    # Measure intent shifts for always-correct
    ac_originals = [r["original"] for r in always_correct_results]
    ac_rewrites = [r["response"] for r in always_correct_results]
    ac_shifts = classifier.check_intent_shift(ac_originals, ac_rewrites)
    for i, s in enumerate(ac_shifts):
        always_correct_results[i]["intent_shifted"] = s["intent_shifted"]

    return {
        "confidence_aware": strategy_results,
        "always_correct": always_correct_results,
        "confidence_distribution": {
            "high": len(high_conf),
            "medium": len(med_conf),
            "low": len(low_conf),
        },
    }


# this is the main function and it runs all the experiments, does the set up, and also handles the after math
# I also did not comment this portion since it was already largely commented
def main():
    set_seed(SEED)

    # Ensure directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Save experiment config
    config = get_experiment_config()
    with open(os.path.join(DATA_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Experiment config: {json.dumps(config, indent=2)}")

    # ─── Sample data ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SAMPLING DATA")
    print("=" * 60)

    b77_samples = sample_queries("banking77", N_SAMPLES_PER_DATASET)
    c150_samples = sample_queries("clinc150", N_SAMPLES_PER_DATASET)
    all_samples = b77_samples + c150_samples

    print(f"Banking77: {len(b77_samples)} samples across {len(set(s['label_id'] for s in b77_samples))} intents")
    print(f"CLINC150: {len(c150_samples)} samples across {len(set(s['label_id'] for s in c150_samples))} intents")

    # Save samples
    with open(os.path.join(DATA_DIR, "samples.json"), "w") as f:
        json.dump(all_samples, f, indent=2)

    # ─── Build intent classifiers ──────────────────────────
    print("\n" + "=" * 60)
    print("BUILDING INTENT CLASSIFIERS")
    print("=" * 60)

    clf_b77 = EmbeddingIntentClassifier("banking77", device="cuda:0")
    clf_c150 = EmbeddingIntentClassifier("clinc150", device="cuda:0")

    # Validate classifier accuracy on test set
    from datasets import load_from_disk
    from config import DATASET_PATHS

    for name, clf, ds_path, label_key in [
        ("banking77", clf_b77, DATASET_PATHS["banking77"], "label"),
        ("clinc150", clf_c150, DATASET_PATHS["clinc150"], "intent"),
    ]:
        ds = load_from_disk(ds_path)
        test_texts = ds["test"]["text"][:500]
        test_labels = ds["test"][label_key][:500]
        preds = clf.classify(test_texts)
        correct = sum(1 for p, l in zip(preds, test_labels) if p["predicted_label"] == l)
        acc = correct / len(test_labels)
        print(f"  {name} classifier accuracy (500 test): {acc:.1%}")

    # ─── Experiment 1: Intent Violation Rate ─────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: INTENT VIOLATION RATE")
    print("=" * 60)

    all_exp1_results = []
    for model_name in MODELS.keys():
        for prompt_key in CORRECTION_PROMPTS.keys():
            # Run on banking77
            b77_results = run_experiment_1(
                b77_samples, model_name, prompt_key, clf_b77
            )
            all_exp1_results.extend(b77_results)

            # Run on clinc150
            c150_results = run_experiment_1(
                c150_samples, model_name, prompt_key, clf_c150
            )
            all_exp1_results.extend(c150_results)

    # Save Experiment 1 results
    with open(os.path.join(DATA_DIR, "experiment1_results.json"), "w") as f:
        json.dump(all_exp1_results, f, indent=2, default=str)

    # Print summary
    print("\n--- Experiment 1 Summary ---")
    for model_name in MODELS.keys():
        for prompt_key in CORRECTION_PROMPTS.keys():
            subset = [r for r in all_exp1_results
                      if r["model"] == model_name and r["prompt_strategy"] == prompt_key]
            n_shifted = sum(1 for r in subset if r["intent_shifted"])
            n_total = len(subset)
            avg_edit = np.mean([r["edit_ratio"] for r in subset])
            avg_sim = np.mean([r["semantic_similarity"] for r in subset])
            print(f"  {model_name} / {prompt_key}: "
                  f"intent_shift={n_shifted}/{n_total} ({n_shifted/n_total:.1%}), "
                  f"edit_ratio={avg_edit:.3f}, sem_sim={avg_sim:.3f}")

    # ─── Experiment 2: Metric Validation ──────────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: METRIC VALIDATION (LLM-as-Judge)")
    print("=" * 60)

    exp2_results = run_experiment_2(all_exp1_results, model_name="gpt-4.1")
    with open(os.path.join(DATA_DIR, "experiment2_results.json"), "w") as f:
        json.dump(exp2_results, f, indent=2, default=str)

    # Print summary
    labels = [r["judge_label"] for r in exp2_results]
    print(f"  Judge labels: preserved={labels.count('preserved')}, "
          f"changed={labels.count('changed')}, ambiguous={labels.count('ambiguous')}, "
          f"error={labels.count('error')}")

    # ─── Experiment 3: Confidence-Aware Strategy ──────────
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: CONFIDENCE-AWARE STRATEGY")
    print("=" * 60)

    # Use banking77 for Experiment 3 (single dataset for cleaner comparison)
    # Take a fresh sample to avoid overlap with Exp1 API calls
    exp3_samples = sample_queries("banking77", 150, seed=SEED + 1)
    exp3_results = run_experiment_3(exp3_samples, clf_b77, model_name="gpt-4.1")

    with open(os.path.join(DATA_DIR, "experiment3_results.json"), "w") as f:
        json.dump(exp3_results, f, indent=2, default=str)

    # Print Experiment 3 summary
    ca = exp3_results["confidence_aware"]
    ac = exp3_results["always_correct"]

    ca_shifts = sum(1 for r in ca if r.get("intent_shifted", False))
    ac_shifts = sum(1 for r in ac if r.get("intent_shifted", False))
    ca_clarify = sum(1 for r in ca if r["action"] == "clarify")
    ca_abstain = sum(1 for r in ca if r["action"] == "abstain")

    print(f"\n  Confidence-Aware: intent_shifts={ca_shifts}/{len(ca)} ({ca_shifts/len(ca):.1%}), "
          f"clarifications={ca_clarify}/{len(ca)} ({ca_clarify/len(ca):.1%}), "
          f"abstentions={ca_abstain}/{len(ca)}")
    print(f"  Always-Correct:   intent_shifts={ac_shifts}/{len(ac)} ({ac_shifts/len(ac):.1%})")
    print(f"  Confidence dist: {exp3_results['confidence_distribution']}")

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"Results saved to {DATA_DIR}/")


if __name__ == "__main__":
    main()
