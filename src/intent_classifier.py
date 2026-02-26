"""
Intent classification using sentence embeddings + nearest neighbor.

Standalone runnable version that:
- Automatically downloads CLINC150 (plus config)
- Builds embeddings
- Tests classification + intent shift detection
"""

# Note from Praggnya: I went through all the files in main and reviewed their code. 
# As I reviewed their code, I commented the general code structure to take note of what was happening
# in order to later help with understanding what kind of improvements I can add.


import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from collections import Counter
from config import DATASET_PATHS, SEED

# Praggnya: added these imports for my improvements
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import LogisticRegression
from metrics import compute_nli_scores


class EmbeddingIntentClassifier:
    """Classifies intent by nearest-neighbor lookup in embedding space."""

    # Praggnya: changed this because I only had access to one dataset, and also since I have a mac, I needed CPU
    # IN TERMS OF RESOURCES: Initially running this file did give me many errors (such as CUDA related errors) 
    # so I did ask ChatGPT to debug. It said I should change it to CPU, and doing so worked
    def __init__(self, dataset_name: str = "clinc150", device="cpu"):
        self.dataset_name = dataset_name
        self.device = device
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        self._build_index()

    def _build_index(self):
        # Praggnya: Changed this method to only use clinc150
        if self.dataset_name == "clinc150":
            ds = load_dataset("clinc_oos", "plus")
            train = ds["train"]
            self.texts = train["text"]
            self.labels = train["intent"]
            self.label_names = train.features["intent"].names
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        # Encode training set
        self.embeddings = self.model.encode(
            self.texts,
            batch_size=128,
            show_progress_bar=True,
            convert_to_tensor=True,
        )
        self.embeddings = torch.nn.functional.normalize(self.embeddings, dim=1)

        # Praggnya: for my Improvement with Centroids: 
        # I looked at this link: https://www.geeksforgeeks.org/machine-learning/ml-nearest-centroid-classifier/
        # I'll be using scikit learn so I will convert the embeddings to numpy
        self.embeddings_numpy = self.embeddings.cpu().numpy()
        # then I will train the centroid, as in the link!
        self.centroids = NearestCentroid()
        self.centroids.fit(self.embeddings_numpy, self.labels)
        # normalizing these too because that's what we did earlier in these methods!
        self.centroids = torch.tensor(self.centroids.centroids_, dtype=torch.float32)
        self.centroids = torch.nn.functional.normalize(self.centroids, dim=1)

        # Praggnya: for my Improvement with Logisitic Regression:
        # I looked here to understand how to use the function:
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html 
        self.logreg = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            n_jobs=-1
        )
        self.logreg.fit(self.embeddings_numpy, self.labels)

    # Praggnya: this is the most core function here for actually classifying the labels
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
        sims = torch.mm(q_emb, self.embeddings.T) # (n_queries, n_train)

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
                # Changed type to float here because some of my results were printing oddly 
                (label, float(np.mean(scores)))
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
    
    # for Improvement: Centroids!
    # in terms of reference, I looked at this link: https://www.geeksforgeeks.org/machine-learning/ml-nearest-centroid-classifier/
    def classify_centroid(self, queries: list[str]) -> list[dict]:
        # here I largely keep the exact same format from the previous classify method
        # though I do remove the top k part
        q_emb = self.model.encode(
            queries,
            batch_size=64,
            show_progress_bar=False,
            convert_to_tensor=True,
            device=self.device,
        )
        q_emb = torch.nn.functional.normalize(q_emb, dim=1)
        # Cosine similarity with centroids
        sims = torch.mm(q_emb, self.centroids.T)  # (n_queries, n_classes)
        results = []
        for i in range(len(queries)):
            top_vals, top_idx = torch.topk(sims[i], k=2)
            predicted_label = top_idx[0].item()
            max_similarity = top_vals[0].item()
            margin = top_vals[0] - top_vals[1]
            confidence = margin.item()
            results.append({
                "predicted_label": predicted_label,
                "predicted_name": self.label_names[predicted_label],
                "confidence": confidence,
                "max_similarity": max_similarity,
            })
        return results
    
    # for Improvement: Logistic Regression!
    # Here I looked at a couple links for references for implementing this:
    # https://realpython.com/logistic-regression-python/
    # https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/
    def classify_logreg(self, queries: list[str]) -> list[dict]:
        # here I also largely keep the exact same format from the previous classify method
        q_emb = self.model.encode(
            queries,
            batch_size=64,
            show_progress_bar=False,
            convert_to_tensor=True,
            device=self.device,
        )
        q_emb = torch.nn.functional.normalize(q_emb, dim=1)
        q_np = q_emb.cpu().numpy()
        # this is a little different and specific to this method of logreg
        # getting the winning class and then the probbaility for every class
        preds = self.logreg.predict(q_np)
        probs = self.logreg.predict_proba(q_np)
        results = []
        for i in range(len(queries)):
            # getting the label
            predicted_label = int(preds[i])
            confidence = float(probs[i][predicted_label])
            results.append({
                "predicted_label": predicted_label,
                "predicted_name": self.label_names[predicted_label],
                "confidence": confidence,
            })

        return results

    # Praggnya: this is the check intent shift function which checks if the prediction for original is different from the rewrite predictions
    # changed method header slightly to include nli threshold for my improvement
    def check_intent_shift(self, originals, rewrites, k=5, nli_threshold=0.8):
        """Check if intent shifts between original and rewrite.

        Returns list of dicts with classification results for both and shift flag.
        """
        orig_results = self.classify(originals, k)
        rew_results = self.classify(rewrites, k)

        results = []
        for orig, rew in zip(orig_results, rew_results):
            # Changing this variable name and now I am checking if they agree!
            classifier_same = orig["predicted_intent"] == rew["predicted_intent"]
            # My Improvement: 
            # using NLI along with classifier agreement to determine intent shift
            nli_result = compute_nli_scores(
                [orig["query"]],
                [rew["query"]]
            )[0]
            nli_score = nli_result["bidirectional"]
            # Now I need to turn this into a boolean so I can make a decision about intent shift later
            # the threshold I am using here is 0.8 because that would be a high score showing preservation 
            nli_boolean = nli_score >= nli_threshold

            # now we are making the decisions togther!
            is_intent_preserved = classifier_same and nli_boolean

            results.append({
                "original": orig["query"],
                "original_intent": orig["predicted_intent"],
                "rewrite": rew["query"],
                "rewrite_intent": rew["predicted_intent"],
                "classifier_same": classifier_same,
                # added these here!
                "nli_score": nli_score,
                "nli_boolean": nli_boolean,
                "is_intent_preserved": is_intent_preserved,
            })

        return results


# for the sake of testing this file, I essentially had to change the entire main function!
if __name__ == "__main__":
    # replacing with CLINC150
    print("Testing CLINC150 classifier...\n")
    clf = EmbeddingIntentClassifier(device="cpu")
    # then I am going to test the queries from CLINC150 they list in REPORT.md
    print("\n--- CLINC150 REPORT.md Queries ---")

    test_texts = [
        "how would you say fly in italian",
        "can you let me know if my vacation was approved",
        "20 yen equals how many dollars"
    ]
    
    # doing the k classification as before!
    preds = clf.classify(test_texts)
    # printing them out
    for p in preds:
        print(p)

    print("\n--- Looking at the Labels Translate vs Change_Language ---")

    translation_interesting = [
        "how do you translate fly to italian",
        "translate fly into italian",
        "say fly in italian"
    ]

    preds = clf.classify(translation_interesting)
    for p in preds:
        print(p)

    print("\n--- Looking at Centroid ---")
    centroid_preds = clf.classify_centroid(test_texts)
    for p in centroid_preds:
        print(p)

    print("\n--- Lookng at Logistic Regression ---")
    logreg_preds = clf.classify_logreg(test_texts)
    for p in logreg_preds:
        print(p)

    
    print("\n--- Looking at different values of k ---")

    example_query = "how would you say fly in italian"
    # testing for these queries
    for k in [1, 3, 5, 10, 20]:
        result = clf.classify([example_query], k=k)[0]
        print(f"\nk={k}")
        print(result)