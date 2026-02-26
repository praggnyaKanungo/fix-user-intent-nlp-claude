# Improvements Made
### Here are some improvements I implemented after reviewing the code for this repository. 

## Filtering Out Gold Label and Original Prediction Mismatches
I saw how an intent shift is classified as the original prompt's predicted intent being different from the rewritten prompt's predicted intent. I specifically noticed how the gold standard is not utilized here, and instead the code uses predicted labels for the original text too. Since the focus of the code is ensuring that the model retains the same intent with its rewritten prompt, which should ultimately match the gold standard as well, I believe that an improvement to this procedure would be to filter out all mismatches with the gold standard and the predicted intent of the original prompt. This ensures that the code focuses on the actual intent, which is the label in the data set, rather than making a prediction from the start that may already differ from the actual user intent.

#### In run_experiment.py:
```
# Praggnya: this is storing the results
    for i, sample in enumerate(samples):
        # Check if classifier correctly identifies original intent
        classifier_correct_on_original = orig_preds[i]["predicted_label"] == gt_labels[i]

        #IMPROVEMENT! I am filtering out the ones where the classifier was not correct in predicting the original
        if not classifier_correct_on_original:
            continue
        
        results.append({...
```
## NLI Bi-directional as Additional Indicator of Intent Shift/Preservation
The report states that for Experiment 2's results, NLIs bi-directional is a good indicator of true change in the meaning of the text and rewritten text. I believed it would be more accurate to incorporate NLI bidirectional for checking if intent stayed the same in Experiment 1 since accuracy in indentifying intent preservation is the goal. I suggest deciding intent preservation through classifier agreement along with NLI.
#### In intent_classifier.py:
```
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
```

## Makefile
This repository contains several moving pieces from previous coursework and projects. I have found that a Makefile is highly efficient in automating and executing the relevant tasks and python files in the repository. Having a Makefile will allow for cleaner commands and generally an easier process for running the experiment. Additionally, these commands would provide a more positive experience to those replicating the experiment on their own.

#### Makefile:
```
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
```
**Full file can be found in root directory.**

Another thing to note is that this Makefile only reflects the commands I was able to run. Ideally, all other commands should be added too.

## Exploring Classification Algorithms + Different k-values
While the usage of k-Nearest Neighbors is understandable, I believe it might not be the best model to use to evaluate confidence. kNN is highly sensitive to neighbors, even if they are outliers, where the smallest rewrite can cause a change in the embedding that leads to a completely different set of nearest neighbors. There is also not necessarily any decision boundary or class separation, which may be more helpful since we are dealing with different labels. Instead, there is more of a reliance on the density of training examples. As the classifier is a key part of this research, I believe it is worth it to look into different algorithms for output that better suit our needs. I tried using different algorithms: **Nearest Centroid and Logistic Regression**.

#### In intent_classifier.py:
```
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
```
**Check intent_classifier.py for full code.**

#### I received the following output:
```
--- CLINC150 REPORT.md Queries ---
{'predicted_label': 29, 'predicted_name': 'change_language', 'confidence': 0.6, 'top_k_labels': [(61, 0.6362623870372772), (29, 0.6290721694628397)], 'max_similarity': 0.6436989307403564}
{'predicted_label': 101, 'predicted_name': 'pto_request_status', 'confidence': 1.0, 'top_k_labels': [(101, 0.9504282593727111)], 'max_similarity': 0.9609124064445496}
{'predicted_label': 38, 'predicted_name': 'exchange_rate', 'confidence': 1.0, 'top_k_labels': [(38, 0.7413468480110168)], 'max_similarity': 0.8519306182861328}

--- Looking at the Labels Translate vs Change_Language ---
{'predicted_label': 29, 'predicted_name': 'change_language', 'confidence': 0.8, 'top_k_labels': [(61, 0.6184285283088684), (29, 0.6130816340446472)], 'max_similarity': 0.6355449557304382}
{'predicted_label': 29, 'predicted_name': 'change_language', 'confidence': 0.8, 'top_k_labels': [(29, 0.5928472727537155), (61, 0.5910605788230896)], 'max_similarity': 0.5986693501472473}
{'predicted_label': 29, 'predicted_name': 'change_language', 'confidence': 0.8, 'top_k_labels': [(61, 0.6466656923294067), (29, 0.6241864413022995)], 'max_similarity': 0.6466656923294067}

--- Looking at Centroid ---
{'predicted_label': 61, 'predicted_name': 'translate', 'confidence': 0.1077485978603363, 'max_similarity': 0.5704396367073059}
{'predicted_label': 101, 'predicted_name': 'pto_request_status', 'confidence': 0.28966397047042847, 'max_similarity': 0.8541290760040283}
{'predicted_label': 38, 'predicted_name': 'exchange_rate', 'confidence': 0.33924874663352966, 'max_similarity': 0.7061100006103516}

--- Lookng at Logistic Regression ---
{'predicted_label': 61, 'predicted_name': 'translate', 'confidence': 0.5681398707966957}
{'predicted_label': 101, 'predicted_name': 'pto_request_status', 'confidence': 0.8550547448630913}
{'predicted_label': 38, 'predicted_name': 'exchange_rate', 'confidence': 0.8226424711853981}

--- Looking at different values of k ---

k=1
{'predicted_label': 61, 'predicted_name': 'translate', 'confidence': 1.0, 'top_k_labels': [(61, 0.6436986923217773)], 'max_similarity': 0.6436986923217773}

k=3
{'predicted_label': 29, 'predicted_name': 'change_language', 'confidence': 0.6666666666666666, 'top_k_labels': [(61, 0.6436986923217773), (29, 0.6360936760902405)], 'max_similarity': 0.6436986923217773}

k=5
{'predicted_label': 29, 'predicted_name': 'change_language', 'confidence': 0.6, 'top_k_labels': [(61, 0.6362621188163757), (29, 0.6290722290674845)], 'max_similarity': 0.6436986923217773}

k=10
{'predicted_label': 29, 'predicted_name': 'change_language', 'confidence': 0.7, 'top_k_labels': [(61, 0.618025521437327), (29, 0.6132179072925023)], 'max_similarity': 0.6436986923217773}

k=20
{'predicted_label': 29, 'predicted_name': 'change_language', 'confidence': 0.4, 'top_k_labels': [(29, 0.604630708694458), (61, 0.5982686132192612), (46, 0.5562091022729874)], 'max_similarity': 0.6436986923217773}
```
The outputs above come from the main code block in intent_classifier.py, and this was just for the sake of minimally testing these algorithms alongside kNN with different k-values. For some queries, all three algorithms seem to agree, but these queries were simply easier cases. Therefore, they don't provide absolute information about the competency of these algorithms. However, it is interesting to examine the confidence levels we achieve from these different algorithms. 

Centroid confidence tends to be much lower than kNN and Logistic Regression. I would love to test Centroids with a large number of queries and see if this trend continues. Ideally, since the goal is to use a confidence aware strategy so the bot does not guess when unsure of user intent, we want confidence levels to truly reflect prediction ability successfully. Because of this, it's important to explore these algorithms further to see which one is more efficient in terms of prediction. 

Medium-high confidence values would be ideal, around 0.7-0.9. A confidence of 1.0, however, appears overconfident for the goal we are trying to achieve. Based on the small amount of results obtained in the output, Logistic Regression looks more promising in terms of its confidence levels. While kNN can have a confident of 1 at times (seems very high), centroid confidence levels seem too low. Once again, these three queries cannot be deemed representative or generalizable to a larger quantity of queries or even general usage, but it's interesting to note with the results obtained for now.

Due to the lack of access to APIs, I was unable to run the entire experiment. Therefore, I unfortunately cannot tell if Nearest Centroid would lead to any improvements in the overall procedure since the little testing here does not show improvement, but I theoretically believe there possibly could be an improvement. I also implemented logistic regression due to its stability and lightweight nature. There should also theoretically be an improvement here, and the output for logistic regression certainly does look better than centroids here. However, once again, it is not fair to make any big claims about the success of these algorithms in classification right now without further statistical analysis of these successes through more testing.

If we were to stick with kNN, then it may be appropriate to consider evaluating different k-values. In the output I sent above, I did actually look at different k-values and I saw more interesting results at k = 10.  Using k = 1 obtained a confidence of 1, which is not ideal. This is due to the effect of a single neighbor, which is not optimal whatsoever. As the value of k increased, the confidence appeared to rise but not get too close to 1, which is ideal. However, it appeared to dip at k = 20, making k = 10 seem rather interesting or worth noting.
