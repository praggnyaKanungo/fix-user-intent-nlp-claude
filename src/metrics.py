"""Evaluation metrics for intent preservation."""
# Note from Praggnya: I went through all the files in main and reviewed their code. 
# As I reviewed their code, I commented the general code structure to take note of what was happening
# in order to later help with understanding what kind of improvements I can add.
# for this file though, I did not comment frequently and instead focused on understanding how the
# metrics were calculated. On the bottom of each calculating function, I took notes of my understanding
# of the metrics.

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Lazy-loaded models
_sbert_model = None
_nli_model = None


def get_sbert_model():
    global _sbert_model
    if _sbert_model is None:
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda:0")
    return _sbert_model


def get_nli_model():
    global _nli_model
    if _nli_model is None:
        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base", device="cuda:0")
    return _nli_model


def compute_edit_ratio(original: str, rewritten: str) -> float:
    """Word-level edit distance normalized by original length."""
    orig_words = original.lower().split()
    rew_words = rewritten.lower().split()

    if len(orig_words) == 0:
        return 1.0 if len(rew_words) > 0 else 0.0

    m, n = len(orig_words), len(rew_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if orig_words[i - 1] == rew_words[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[m][n] / max(m, 1)
# this ratio essentially tries to give us a sense of how much of the original message was changed
# objectively in terms of characters in the rewrite 


def compute_semantic_similarity_batch(originals: list[str], rewrites: list[str]) -> list[float]:
    """Cosine similarity using sentence embeddings (proxy for BERTScore)."""
    if not originals:
        return []
    model = get_sbert_model()
    emb_o = model.encode(originals, batch_size=64, show_progress_bar=False, convert_to_tensor=True)
    emb_r = model.encode(rewrites, batch_size=64, show_progress_bar=False, convert_to_tensor=True)
    sims = torch.nn.functional.cosine_similarity(emb_o, emb_r, dim=1)
    return sims.cpu().tolist()
# this just checks the embeddings of sentences in comparison to each other


def compute_nli_scores(originals: list[str], rewrites: list[str]) -> list[dict]:
    """Compute bidirectional NLI entailment scores using CrossEncoder.

    The cross-encoder model outputs logits for [contradiction, neutral, entailment].
    We use softmax to get probabilities.

    Returns list of dicts with:
      - forward_entailment: P(original entails rewrite)
      - backward_entailment: P(rewrite entails original)
      - bidirectional: min of both (proxy for semantic equivalence)
    """
    model = get_nli_model()

    # Forward: (original, rewrite) - does original entail rewrite?
    forward_pairs = list(zip(originals, rewrites))
    # Backward: (rewrite, original) - does rewrite entail original?
    backward_pairs = list(zip(rewrites, originals))

    # CrossEncoder.predict returns logits for [contradiction, neutral, entailment]
    fwd_logits = model.predict(forward_pairs, batch_size=32, show_progress_bar=False)
    bwd_logits = model.predict(backward_pairs, batch_size=32, show_progress_bar=False)

    # Softmax to get probabilities; entailment is index 2 (for nli-deberta-v3-base:
    # label mapping is contradiction=0, neutral=1, entailment=2)
    fwd_probs = torch.softmax(torch.tensor(fwd_logits), dim=1)
    bwd_probs = torch.softmax(torch.tensor(bwd_logits), dim=1)

    results = []
    for i in range(len(originals)):
        f_ent = fwd_probs[i][2].item()  # entailment is index 2
        b_ent = bwd_probs[i][2].item()
        results.append({
            "forward_entailment": f_ent,
            "backward_entailment": b_ent,
            "bidirectional": min(f_ent, b_ent),
        })

    return results
# this checks in both directions if one sentence entails the other
# honestly, to me it makes sense why this might be a better way to capture intent preservation


def compute_all_metrics(originals: list[str], rewrites: list[str]) -> list[dict]:
    """Compute all metrics for a batch of original-rewrite pairs."""
    n = len(originals)
    assert len(rewrites) == n

    edit_ratios = [compute_edit_ratio(o, r) for o, r in zip(originals, rewrites)]
    sem_sims = compute_semantic_similarity_batch(originals, rewrites)
    nli_scores = compute_nli_scores(originals, rewrites)

    results = []
    for i in range(n):
        results.append({
            "edit_ratio": edit_ratios[i],
            "semantic_similarity": sem_sims[i],
            "nli_forward": nli_scores[i]["forward_entailment"],
            "nli_backward": nli_scores[i]["backward_entailment"],
            "nli_bidirectional": nli_scores[i]["bidirectional"],
        })

    return results


if __name__ == "__main__":
    originals = [
        "How do I check my balance?",
        "I want to cancel my card",
        "What is the exchange rate for euros?",
    ]
    rewrites = [
        "How can I check my account balance?",
        "I want to order a new card",  # intent changed!
        "What is the current EUR exchange rate?",
    ]
    results = compute_all_metrics(originals, rewrites)
    for o, r, m in zip(originals, rewrites, results):
        print(f"Original: {o}")
        print(f"Rewrite:  {r}")
        for k, v in m.items():
            print(f"  {k}: {v:.4f}")
        print()
