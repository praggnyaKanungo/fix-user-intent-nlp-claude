"""Intent classification using sentence embeddings + nearest neighbor.

We use the training set of each dataset to build an embedding-based classifier,
then classify both original and rewritten queries to detect intent shifts.
"""


# Note from Praggnya: I went through all the files in main and reviewed their code. 
# As I reviewed their code, I commented the general code structure to take note of what was happening
# in order to later help with understanding what kind of improvements I can add.


# all the imports
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from datasets import load_from_disk
from collections import Counter
from config import DATASET_PATHS, SEED


class EmbeddingIntentClassifier:
    """Classifies intent by nearest-neighbor lookup in embedding space."""

    def __init__(self, dataset_name: str, device="cuda:0"):
        self.dataset_name = dataset_name
        self.device = device
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        self._build_index()

    def _build_index(self):
        ds = load_from_disk(DATASET_PATHS[self.dataset_name])
        train = ds["train"]

        if self.dataset_name == "banking77":
            self.texts = train["text"]
            self.labels = train["label"]
            self.label_names = train.features["label"].names
        elif self.dataset_name == "clinc150":
            self.texts = train["text"]
            self.labels = train["intent"]
            self.label_names = train.features["intent"].names
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        # Encode training set
        self.embeddings = self.model.encode(
            self.texts, batch_size=128, show_progress_bar=False,
            convert_to_tensor=True, device=self.device,
        )
        self.embeddings = torch.nn.functional.normalize(self.embeddings, dim=1)


    # this is the most core function here for actually classifying the labels
    # as we can see here, they are defaulting to use k=5
    def classify(self, queries: list[str], k: int = 5) -> list[dict]:
        """Classify queries using k-NN in embedding space.

        Returns list of dicts with:
          - predicted_label: int
          - predicted_name: str
          - confidence: float (fraction of k neighbors with majority label)
          - top_k_labels: list of (label_id, score) tuples
        """
        q_emb = self.model.encode(
            queries, batch_size=64, show_progress_bar=False,
            convert_to_tensor=True, device=self.device,
        )
        q_emb = torch.nn.functional.normalize(q_emb, dim=1)

        # Cosine similarity with all training examples
        sims = torch.mm(q_emb, self.embeddings.T)  # (n_queries, n_train)

        results = []
        for i in range(len(queries)):
            topk_vals, topk_idx = torch.topk(sims[i], k)
            topk_labels = [self.labels[idx.item()] for idx in topk_idx]

            # Majority vote with confidence
            counter = Counter(topk_labels)
            predicted_label = counter.most_common(1)[0][0]
            confidence = counter[predicted_label] / k

            # Top-k label distribution
            label_scores = {}
            for label, sim_val in zip(topk_labels, topk_vals.tolist()):
                if label not in label_scores:
                    label_scores[label] = []
                label_scores[label].append(sim_val)

            top_k_labels = [
                (label, np.mean(scores))
                for label, scores in sorted(label_scores.items(), key=lambda x: -np.mean(x[1]))
            ]

            results.append({
                "predicted_label": predicted_label,
                "predicted_name": self.label_names[predicted_label],
                "confidence": confidence,
                "top_k_labels": top_k_labels,
                "max_similarity": topk_vals[0].item(),
            })

        return results


    # this is the check intent shift function which checks if the prediction for original is different from the rewrite predictions
    def check_intent_shift(self, originals: list[str], rewrites: list[str], k: int = 5):
        """Check if intent shifts between original and rewrite.

        Returns list of dicts with classification results for both and shift flag.
        """
        orig_results = self.classify(originals, k)
        rew_results = self.classify(rewrites, k)

        results = []
        for orig, rew in zip(orig_results, rew_results):
            intent_shifted = orig["predicted_label"] != rew["predicted_label"]
            results.append({
                "original_intent": orig["predicted_name"],
                "original_confidence": orig["confidence"],
                "rewrite_intent": rew["predicted_name"],
                "rewrite_confidence": rew["confidence"],
                "intent_shifted": intent_shifted,
            })

        return results


if __name__ == "__main__":
    print("Testing Banking77 classifier...")
    clf = EmbeddingIntentClassifier("banking77")

    # Test accuracy on a few test examples
    ds = load_from_disk(DATASET_PATHS["banking77"])
    test_texts = ds["test"]["text"][:20]
    test_labels = ds["test"]["label"][:20]
    label_names = ds["test"].features["label"].names

    preds = clf.classify(test_texts)
    correct = sum(1 for p, l in zip(preds, test_labels) if p["predicted_label"] == l)
    print(f"Test accuracy (20 samples): {correct}/{len(test_labels)} = {correct/len(test_labels):.1%}")

    # Test intent shift detection
    originals = ["How do I check my balance?", "I want to cancel my card"]
    rewrites = ["How can I check my account balance?", "I want to order a new card"]
    shifts = clf.check_intent_shift(originals, rewrites)
    for o, r, s in zip(originals, rewrites, shifts):
        print(f"\nOriginal: {o}")
        print(f"Rewrite: {r}")
        print(f"Shift: {s}")
