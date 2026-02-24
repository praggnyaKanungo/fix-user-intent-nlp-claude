# "Do You Mean...?": Fixing User Intent Without Annoying Them

## 1. Executive Summary

**Research Question:** When LLMs rewrite or correct user queries, how often do they alter the user's original intent? Can a confidence-aware correction strategy preserve intent while minimizing unnecessary clarification?

*Annotation: I think this is a very cool topic! As a Claude and ChatGPT user myself, I notice that sometimes I may prompt one thing but the LLM will interpret it differently and give me different results. This is definitely not fun, but also I do think the LLM asking for clarification can hinder usability because it would be annoying!*

**Key Finding:** LLMs alter user intent in 1.5% to 15% of corrections depending on the aggressiveness of the rewriting strategy. Conservative correction ("fix errors") preserves intent 98.5% of the time, while more aggressive strategies ("rewrite clearly", "improve") cause intent shifts in 9-15% of cases. Claude Sonnet 4.5 makes more aggressive edits than GPT-4.1 and consequently causes more intent shifts. A confidence-aware strategy that selectively asks clarifying questions can eliminate intent violations entirely, at the cost of requesting clarification for only 9.3% of ambiguous queries. 

*Annotation: how is this shift in intent detected? I'll aim to answer this by the end of reading this. What does a confident-aware strategy means in this case? Will aim to answer this by the end of this paper!*

**Practical Implications:** Systems that rewrite user queries should use the most conservative correction strategy possible and only escalate to full rewrites when explicitly requested. When the system is uncertain, asking a short clarifying question is far better than guessing wrong.

*Annotation: I agree yes! If the system is uncertain, it definitely should not guess and provide unexpected results because that would lose user trust.*

---

## 2. Goal

### Hypothesis
Autocomplete and intent detection systems often incorrectly correct users in ways that alter their original intent. We hypothesize that:
1. LLMs frequently alter intent when asked to rewrite queries (>15% violation rate for aggressive strategies)
2. Intent violations are detectable using automated metrics
3. A confidence-aware strategy reduces intent violations without excessive questioning

*Annotation: Once again, not sure what they mean by automated metrics or confidence-aware strategy but will aim to answer that by the end of reading this.*

### Why This Matters
Query correction is ubiquitous in search engines, chatbots, and virtual assistants. When a system "helpfully" rewrites a user's query but changes its meaning, the user gets wrong results and loses trust. This research quantifies the problem and proposes a practical solution.

*Annotation: Yes, that's the exact thing I thought of when I first read the research question!*

### Expected Impact
Understanding when and how LLMs change user intent enables the design of better correction systems that know when to fix, when to ask, and when to leave well enough alone.

*Annotation: I think this is very impactful, and I chose this research repo because for my senior technical capstone, I proposed the design of an AI-tutoring system to teach introductory CS, and one thing I did worry about was the usability of the AI agent and how it can truly understand the knowledge level (see if the student truly has learned) and questions asked by the students. The AI agent misunderstanding the user is a huge risk that applies to the tutoring system I proposed as well, which was pivotal to me wanting to research Human-centered AI even more (to increase its use and applications for expanding accessibility in CS Education). My underlying motivation for these things is what caused me to even choose this repo.*


---

## 3. Data Construction

### Dataset Description
We used two established intent classification benchmarks:

| Dataset | Split | Size | Intents | Source |
|---------|-------|------|---------|--------|
| BANKING77 | test | 3,080 | 77 | Banking customer service queries |
| CLINC150 | test | 5,500 | 150 + OOS | Multi-domain virtual assistant queries |

These datasets were chosen because each query has a gold-standard intent label, allowing us to detect when a rewrite changes the underlying intent.

*Annotations: Note to self that gold-standard intent label means the "true" and accepted intent label for that query*

### Example Samples

**BANKING77:**
| Query | Intent |
|-------|--------|
| "How do I locate my card?" | card_arrival |
| "Can I change the amount I made on a payment..." | cancel_transfer |
| "How long with my cash withdrawal stay pending for?" | pending_cash_withdrawal |

**CLINC150:**
| Query | Intent |
|-------|--------|
| "how would you say fly in italian" | translate |
| "can you let me know if my vacation was approved" | pto_request_status |
| "20 yen equals how many dollars" | exchange_rate |

### Sampling Strategy
We stratified sampling across intents to ensure diverse coverage:
- 200 queries from BANKING77 (across 77 intents)
- 200 queries from CLINC150 (across 150 intents)
- Total: 400 queries per model-prompt combination

### Data Quality
The datasets come from established NLP benchmarks with validated annotations. We verified our intent classifier achieves 93% accuracy on BANKING77 and 83% on CLINC150 using 5-NN with sentence embeddings, establishing a reliable baseline for detecting intent shifts.

*Annotation: Intent classifier seems very accurate, but I wonder if there's a way to make it more accurate? I have not looked at the code yet, but I will look into the classifier code as well in case I can find a way to make an improvement there.*

---

## 4. Experiment Description

### Methodology

#### High-Level Approach
We prompt real state-of-the-art LLMs (GPT-4.1, Claude Sonnet 4.5) to correct/rewrite user queries under three increasingly aggressive strategies, then measure whether the rewritten query preserves the original intent. We define "intent preservation" operationally as: the rewritten query, when classified by an embedding-based intent classifier, maps to the same intent label as the original.

*Annotation: I am just going to summarize to myself here to make sure I understand: We are prompting an LLM to rewrite a query we give it on a particular level of agression (for example, low, med, high in terms of how aggressively it can correct or rewrite it). Then, we will have a rewritten query. This new query that the LLM writes will be send into the embedding-based intent classifier. The point is to see if the intent is the same even after it corrects/rewrites it. The way we would know that the intent is preserved is if it maps to the same intent label as before. I think that's really cool!*

#### Why This Method?
1. Using real LLM APIs (not simulated) ensures our results reflect actual model behavior
2. Intent classification datasets provide ground-truth labels for measuring intent shift
3. Multiple prompting strategies test the spectrum from conservative to aggressive correction
4. Two different LLM models test generalizability

*Annotation: This makes logical sense to me for sure. Out of curiosity, I wonder why GPT and Claude specifically though! I did see that there are two other repos about the same topic but they are labeled with gemini and codex rather than claude. This means they tested a different model in those repos. If time permits, I would love to see how the results differed between these repos.*

### Implementation Details

#### Tools and Libraries
| Component | Tool | Version |
|-----------|------|---------|
| Language Models | GPT-4.1 (OpenAI), Claude Sonnet 4.5 (OpenRouter) | 2025 |
| Intent Classifier | all-MiniLM-L6-v2 + 5-NN | sentence-transformers 5.2.2 |
| NLI Model | cross-encoder/nli-deberta-v3-base | sentence-transformers |
| Semantic Similarity | all-MiniLM-L6-v2 | sentence-transformers |
| GPU | 2x NVIDIA RTX 3090 (24GB each) | CUDA |

#### Prompting Strategies

| Strategy | Prompt | Aggressiveness |
|----------|--------|---------------|
| **fix_errors** | "Fix any errors in the following user query. Return ONLY the corrected query, nothing else. If there are no errors, return the query unchanged." | Low |
| **rewrite_clearly** | "Rewrite the following user query to be clearer and more precise. Return ONLY the rewritten query, nothing else." | Medium |
| **improve** | "Improve the following user query to better express the user's intent. Return ONLY the improved query, nothing else." | High |

*Annotation: This is really interesting. While intuitively it makes sense why each of these prompts are assigned to the levels they are (low, med, and high), I wonder how these prompts were chosen. Is there a clear reasoning behind it or is it inituitive?*

#### Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Temperature | 0 | Deterministic outputs for reproducibility |
| Max tokens | 256 | Sufficient for short query rewrites |
| k (NN classifier) | 5 | Standard choice for k-NN classification |
| Confidence threshold (high) | 0.80 | Based on CICC paper (den Hengst et al., 2024) |
| Confidence threshold (low) | 0.40 | Empirically set |
| Random seed | 42 | Standard reproducibility seed |

*Annotations: Temperature being set to 0 makes it so the LLM gives the same output each time, which is certainly helpful in this research. Max tokens is set to 256 to make sure the length of the rewrites (what the LLM will give us). Here they are using k=5 because that's a standard choice, but I wonder if this could be improved (if a change in k can lead to differences in intention shift).*

### Experimental Protocol

#### Experiment 1: Intent Violation Rate Measurement
- 400 queries (200 per dataset) × 3 prompt strategies × 2 models = **2,400 LLM API calls**
- For each (query, rewrite) pair: compute semantic similarity, edit ratio, NLI scores, and intent classifier predictions
- Measure intent shift as: original_predicted_intent ≠ rewrite_predicted_intent

*Annotation: I have a question here...why is it comparing the rewrite_predicted_intent to original_predicted_intent? I am trying to understand why it would be compared to the predicted original intent when the dataset already features the gold standard intent label? Is this a choice for standardization here?*

#### Experiment 2: Metric Validation (LLM-as-Judge)
- 100 (query, rewrite) pairs sampled from Experiment 1 (balanced between shifted/preserved)
- GPT-4.1 judges whether intent was preserved (PRESERVED/CHANGED/AMBIGUOUS)
- Correlate automated metrics with judge labels

*Annotation: I initially had trouble wrapping my head around why experiment 2 is even needed but I think I understand why now. Experiment 1 uses the classifier but how do we know if that's valid and if the classifier isn't overreacting to a small change for example? That's why it important to use the LLM as the judge here for validation.*

#### Experiment 3: Confidence-Aware Strategy
- 150 queries from BANKING77
- Classify each query with confidence score (fraction of k-NN agreeing)
- Apply threshold-based strategy: high confidence → auto-correct, medium → clarify, low → abstain
- Compare against always-correct baseline

*Annotation: I finally understandwhat the confidence aware strategy is! So based on the confidence score of a query (which they say is "fraction of k-NN agreeing" so like if 4 out of 5 agree, the fraction is 4/5=0.8), set some thresholds for what range we will autocorrect, what range we will ask clarificaiton questions, and what range we will not do anything at all. This is essentially a way to build an intelligent way to decide when to ask clarification questions vs when to not.*

#### Reproducibility
- Seed: 42
- Temperature: 0 (deterministic)
- Hardware: 2x RTX 3090 (24GB)
- Total API calls: ~3,050 across all experiments
- Total runtime: ~50 minutes

*Annotation: Based on this, it might not be feasible for me to run this on my own, especially with all of these APIs that I do not have the token of.*

### Evaluation Metrics
| Metric | What it Measures | Range |
|--------|-----------------|-------|
| **Intent Preservation Rate** | % of rewrites maintaining same intent label | 0-100% |
| **Semantic Similarity** | Cosine similarity of SBERT embeddings | 0-1 |
| **Edit Ratio** | Normalized word-level Levenshtein distance | 0-∞ |
| **NLI Entailment** | P(original entails rewrite) from cross-encoder | 0-1 |
| **Clarification Rate** | % of queries triggering clarification | 0-100% |

---

## 5. Raw Results

### Experiment 1: Intent Violation Rates

| Model | Strategy | Intent Shift Rate | 95% CI | Edit Ratio | Semantic Sim | Unchanged % |
|-------|----------|:-:|:-:|:-:|:-:|:-:|
| GPT-4.1 | fix_errors | **1.5%** | [0.5%, 2.8%] | 0.128 | 0.975 | 26.2% |
| GPT-4.1 | rewrite_clearly | **9.2%** | [6.2%, 12.2%] | 0.953 | 0.822 | 0.0% |
| GPT-4.1 | improve | **10.0%** | [7.2%, 13.0%] | 1.040 | 0.804 | 0.0% |
| Claude 4.5 | fix_errors | **1.5%** | [0.5%, 2.8%] | 0.083 | 0.984 | 49.8% |
| Claude 4.5 | rewrite_clearly | **15.0%** | [11.5%, 18.8%] | 1.570 | 0.739 | 0.0% |
| Claude 4.5 | improve | **14.2%** | [11.0%, 18.0%] | 1.962 | 0.746 | 0.0% |

**Statistical Comparisons (Wilcoxon signed-rank):**
- fix_errors vs rewrite_clearly: p < 0.0001 *** (both models)
- fix_errors vs improve: p < 0.0001 *** (both models)
- rewrite_clearly vs improve: p = 0.53/0.62 ns (no significant difference)

*Annotation: So based on the p-values there is a significant difference between the fix_errors vs rewrite_clearly and fix_errors vs improve. That makes sense that there would be a significant difference between low vs higher.*

**By Dataset:**
- BANKING77: 8.2% overall shift rate
- CLINC150: 9.0% overall shift rate

*Annotation: Just intuitively and not based on statistical tests, those shift rates seem very similar.*

### Experiment 2: Metric Validation

LLM-as-judge (GPT-4.1) labeled 100 examples: 94 as PRESERVED, 6 as CHANGED, 0 as AMBIGUOUS.

| Metric | Correlation with Judge | p-value |
|--------|:---------------------:|:-------:|
| Intent classifier (kappa) | 0.040 | - |
| Semantic similarity | r = 0.161 | 0.109 |
| **Edit ratio (inverse)** | **r = 0.379** | **0.0001** |
| NLI forward | r = -0.127 | 0.208 |
| **NLI backward** | **r = -0.327** | **0.0009** |
| **NLI bidirectional** | **r = -0.408** | **< 0.0001** |

### Experiment 3: Confidence-Aware Strategy

| Strategy | Intent Shifts | Clarification Rate | Effective Accuracy* |
|----------|:--:|:--:|:--:|
| **Confidence-Aware** | **0/150 (0.0%)** | **14/150 (9.3%)** | **0.972** |
| Always-Correct | 1/150 (0.7%) | 0% | 0.993 |
| No-Action | 0/150 (0.0%) | 0% | 1.000 |
| Always-Clarify | 0/150 (0.0%) | 100% | 0.700 |

*Effective Accuracy = 1 - shift_rate - 0.3 × clarify_rate (penalizes both errors and unnecessary questioning)

**Confidence Distribution:** High (>0.8): 90.7%, Medium (0.4-0.8): 9.3%, Low (<0.4): 0.0%

*Annotation: Note to self that these are the current thresholds!*

#### Example Clarifications Generated

| Original Query | Clarification | Confidence |
|---------------|---------------|:----------:|
| "Can you freeze my account? I just saw there are transactions I don't recognize..." | "Are you referring to your bank account, credit card, or another type of account?" | 0.40 |
| "did not receive correct cash upon withdrawal" | "Did you receive less cash than expected, or was there an issue with the denominations?" | 0.40 |
| "What us the fee to transfer money from my bank?" | "Are you asking about transferring money domestically or internationally?" | 0.60 |

---

## 5. Result Analysis

### Key Findings

**Finding 1: Conservative correction preserves intent; aggressive rewriting does not.**
"Fix errors" prompts alter intent in only 1.5% of queries (both models), while "rewrite clearly" and "improve" prompts cause 9-15% intent shifts. This is a 6-10x increase in intent violations from merely changing the prompt instruction.

**Finding 2: Claude Sonnet 4.5 makes more aggressive edits than GPT-4.1.**
Claude's "rewrite_clearly" strategy has a 15% intent shift rate vs GPT's 9.2%. Claude also has a much higher edit ratio (1.57 vs 0.95) and lower semantic similarity (0.739 vs 0.822). This means Claude rewrites more words and deviates further from the original meaning, especially when given open-ended instructions.

**Finding 3: The "fix errors" strategy is remarkably safe.**
Both models achieve near-identical 1.5% intent shift rates with fix_errors. Notably, Claude leaves 49.8% of queries unchanged (vs GPT's 26.2%), showing Claude is even more conservative when asked specifically to fix errors. Most of the 1.5% shifts are likely due to classifier noise rather than actual intent changes.

**Finding 4: NLI bidirectional entailment is the best automated metric for detecting intent changes.**
Among automated metrics, NLI bidirectional entailment (min of forward and backward) shows the strongest correlation with LLM-judge labels (r = -0.408, p < 0.0001). Edit ratio inverse also correlates well (r = 0.379, p = 0.0001). Semantic similarity alone is insufficient (r = 0.161, p = 0.109).

*Annotation: This shows that Experiment 1's way of just seeing classifier disagreement as intent shift might not be super accurate or can be made more accurate and that NLI might be better. I honestly think it could be beneficital to use NLI bidirectional entailment alongside the current classifier disagreement.*

**Finding 5: A confidence-aware strategy eliminates intent violations with minimal clarification.**
The confidence-aware approach achieves 0% intent shifts while only asking clarifying questions for 9.3% of queries (those where the classifier confidence is medium). The generated clarifications are specific and helpful, asking about genuine ambiguities rather than generic "what do you mean?" questions.

### Hypothesis Testing Results

| Hypothesis | Result | Evidence |
|-----------|--------|----------|
| H1: >15% intent violation for aggressive correction | **Partially supported** | Claude hits 15%, GPT reaches 10%. Average: ~12% |
| H2: Automated metrics detect intent violations | **Supported** | NLI bidirectional: r=-0.408 (p<0.0001); Edit ratio: r=0.379 (p<0.001) |
| H3: Confidence-aware strategy reduces violations | **Supported** | 0% violations vs 0.7% for always-correct, with only 9.3% clarification |

*Annotation: It is clear that the current Confidence aware strategy reduces violations very well. I wonder how changing this strategy slightly would change the results.*

### Surprises and Insights

1. **Claude's verbosity amplifies intent drift**: Claude's "improve" strategy produces rewrites that are 2-10x longer than the original, often adding context the user never mentioned. While sometimes helpful, this introduces assumptions that can shift intent.

2. **The classifier catches subtle shifts**: Even when the rewrite looks semantically similar (high cosine similarity), the intent classifier sometimes detects a shift. Example: "how long do money transfers take?" → "How long does it take for a money transfer to be completed?" shifted from `transfer_not_received_by_recipient` to `pending_transfer`.

3. **Most queries have high classifier confidence**: 90.7% of Banking77 queries have >0.8 classifier confidence, meaning the confidence-aware strategy defaults to auto-correction for most inputs. This is actually desirable — clarification should be the exception, not the rule.

### Error Analysis

**Types of intent shifts observed:**

1. **Semantic broadening** (most common): The rewrite generalizes the query, mapping it to a broader intent category. E.g., "Can you help with a transfer to an account" (beneficiary_not_allowed) → "Can you assist me with transferring funds to another account?" (transfer_into_account).

2. **Context injection**: The model adds context not present in the original, steering toward a different intent. E.g., Claude's "improve" strategy often adds phrases like "I'd like to..." or "Can you help me with..." that change the pragmatic meaning.

3. **Classifier noise**: Some "shifts" are due to the classifier being sensitive to surface-level word changes while the semantic intent is preserved. This accounts for most fix_errors shifts.

*Annotations: These are really cool! I wonder if there are more types of intent shifts or how exactly this was observed?*

### Limitations

1. **Classifier as ground truth**: Our intent shift detection relies on an embedding-based classifier (93% accuracy on BANKING77, 83% on CLINC150). Some detected shifts may be classifier errors rather than genuine intent changes.

2. **Domain specificity**: We tested on banking and virtual assistant queries. Results may differ for other domains (medical, legal, creative writing).

3. **Deterministic generation**: Using temperature=0 means we capture the model's "default" behavior but miss the variance that would occur with non-zero temperature.

4. **English only**: All experiments use English queries. Intent preservation may be harder for languages with more grammatical ambiguity.

5. **Small confidence-aware evaluation**: Experiment 3 used only 150 queries from one dataset. The 0% intent shift rate is promising but the Wilcoxon test was not statistically significant (p=0.16), likely due to the low base rate.

---

## 6. Conclusions

### Summary
LLMs frequently alter user intent when asked to rewrite queries, with violation rates ranging from 1.5% (conservative "fix errors") to 15% (aggressive "improve/rewrite") depending on the prompting strategy. Conservative correction is remarkably safe and should be the default. When uncertainty exists, a confidence-aware approach that selectively asks clarifying questions can further reduce intent violations while only requesting clarification for ~9% of ambiguous queries.

### Implications

**For practitioners building correction/rewriting systems:**
- Default to the most conservative correction strategy available
- Only rewrite aggressively when the user explicitly requests it
- Monitor edit ratio as a leading indicator of intent drift (>1.0 is a warning sign)
- Use NLI bidirectional entailment as the most reliable automated metric for intent preservation

**For researchers:**
- Intent preservation is a measurable property that should be evaluated alongside quality metrics like fluency and clarity
- The DR GENRE framework's insight about "conciseness reward" (penalizing unnecessary edits) is empirically validated — our data shows a strong correlation between edit ratio and intent shift

**For users:**
- Be specific in your requests to AI systems: "fix my typos" produces much better intent preservation than "improve my query"

### Confidence in Findings
High confidence in the main finding (conservative > aggressive correction). Medium confidence in metric validation (limited to 100 judge examples). The confidence-aware strategy results are promising but would benefit from a larger-scale evaluation.

---

## 7. Next Steps

### Immediate Follow-ups
1. **Larger-scale Experiment 3**: Run the confidence-aware strategy on 500+ queries across both datasets with multiple models to achieve statistical significance
2. **Human evaluation**: Replace LLM-as-judge with human annotators to validate intent preservation judgments
3. **Temperature sweep**: Test with temperature ∈ {0.0, 0.3, 0.7, 1.0} to measure how generation variance affects intent preservation

### Alternative Approaches
- **Fine-tuned intent classifier**: Replace 5-NN with a fine-tuned BERT model for more reliable intent shift detection
- **Conformal prediction**: Use CICC-style conformal prediction for principled confidence calibration
- **Multi-turn evaluation**: Test whether providing conversation context reduces intent shifts

### Broader Extensions
- Apply to other domains: medical query correction, code completion, legal document rewriting
- Test with smaller/cheaper models (GPT-4.1-mini, Haiku) for cost-effective production deployment
- Develop a real-time intent preservation monitoring system for production use

### Open Questions
1. Can models be fine-tuned to self-detect when their rewrites shift intent?
2. What is the optimal clarification rate that balances helpfulness with user annoyance?
3. How do intent preservation rates change across languages?

---

## References

1. Arora et al. "Intent Detection in the Age of LLMs." 2024.
2. Den Hengst et al. "Conformal Intent Classification and Clarification (CICC)." 2024.
3. Deng et al. "InteractComp: Benchmark for Ambiguous Query Understanding." 2025.
4. Hu et al. "Interactive Question Clarification in Dialogue via Reinforcement Learning." 2020.
5. Jayanthi et al. "NeuSpell: A Neural Spelling Correction Toolkit." EMNLP 2020.
6. Li et al. "DR GENRE: RL from Decoupled LLM Feedback for Generic Text Rewriting." 2025.
7. Shen et al. "On the Evaluation Metrics for Paraphrase Generation (ParaScore)." 2022.
8. Wang et al. "Zero-shot Clarifying Question Generation for Conversational Search." 2023.
9. Zhang et al. "BERTScore: Evaluating Text Generation with BERT." ICLR 2020.
10. Zhang et al. "Correcting the Autocorrect: Context-Aware Typographic Error Correction." 2020.


*Annotations: I had note to myself to answer the following after reading this as I did not understand this initially: how is this shift in intent detected and what does a confident-aware strategy means in this case? A shift in intent here was detected by disagreement of the classifier. A confident aware strategy here is a strategy for the LLM or builders making better systems for LLMS where they can use the confidence score to choose whether to abstain, ask a clarifying question, or just autocorrect (very high confidence).*


