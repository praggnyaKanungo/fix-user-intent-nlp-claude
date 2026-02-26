# Reviewing and Suggesting Improvements to "Do You Mean...?": Fixing User Intent Without Annoying Them
## How I went about doing this:
- Initially, I annotated REPORT.md in order to show my thinking/learning process. This file is now named Annotated-REPORT.md.
- Then, I reviewed the code in src as this is where the core of the research lies! I commented files in this directory as well. My comments are labeled as “Praggnya: …” Doing this helped me understand how the descriptions in REPORT.md translates to code.
- Throughout both annotating REPORT.md and reviewing the code in src, my annotations and comments often reflect potential improvements and questions I have. Using these ideas, I created a list of suggested improvements in LISTED-IMPROVEMENTS.md.
- While creating this list, I also tried to implement as much of these improvements as possible into the code base. LISTED-IMPROVEMENTS.md will show code snippets I added and will describe what I changed. These changes can also be seen in the files themselves!
- Then at the end, I updated the README.md on the branch to reflect this process, explain limitations I faced, express what else I would have looked into with more time and resources, and provide links to references and how I used them.

## Limitations faced:
- Running python download_datasets.py did not result in a successful download of BANKING77 for me. This meant that for me to test the classifier, I did have to change some things in the original code (even before I added any improvements) so I can use CLINC150 only.
- Run_experiments.py was the most crucial file in src, but unfortunately I did not get to run that. That is because, from my understanding, I did not have the API tokens needed for running this file. Therefore, while I did make some tweaks in this file, I was not able to test it unfortunately. Therefore, I tried to make minimal tweaks in this file.

## Things I would have done with more time/Next Steps:
- I would try to obtain all resources needed, such as appropriate API keys, to actually be able to run run_experiments.py and see how the results differ about my improvements throughout the repository. I would iterate in making changes to achieve better results as I would now have the opportunity to actually run and obtain results as many times as needed.
- I would like to do a statistical analysis to obtain quantifiable evidence of what k value (for kNN) would be optimal in this case.
- I would also like to so a statistical analysis of different classifications methods against kNN, including the ones I limitedly implemented such as Centroids and Logistic Regression.
- I would love to see how this research and confidence aware clarification strategies can be applied to an AI chatbot (perhaps we can continue using Claude as the model) in the context of education. How would this affect student engagement and understanding and would the AI chatbot respond in ways that are true to the question the students are asking? What could be ways to measure this? These are all questions I have that I would love to explore.

## Resources Used:
- https://www.geeksforgeeks.org/machine-learning/ml-nearest-centroid-classifier/: This link has an example implementation of centroid classifier and I used this to help implement it in the intent_classifier.py file.
- https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/: This link has an example implementation of logistic regression and I used this to help implement it in the intent_classifier.py file.
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html: This link explained the functions in scikit-learn for Logistic Regression.
-  https://realpython.com/logistic-regression-python/: This link showed how Logistic Regression can be implemented in Python.
- ChatGPT 5.2 for debugging purposes.


# Information about the Original Project

Evaluating how often LLMs alter user intent when correcting/rewriting queries, and whether a confidence-aware clarification strategy can reduce intent violations without excessive questioning.

## Key Findings

- **Conservative correction is safe**: "Fix errors" prompts alter intent in only 1.5% of cases (both GPT-4.1 and Claude Sonnet 4.5)
- **Aggressive rewriting causes 9-15% intent shifts**: "Rewrite clearly" and "Improve" prompts cause significantly more intent violations
- **Claude is more aggressive than GPT**: Claude's rewrites have higher edit ratios (1.57 vs 0.95) and more intent shifts (15% vs 9.2%)
- **NLI bidirectional entailment is the best metric**: Strongest correlation with human-like judgments (r = -0.408, p < 0.0001)
- **Confidence-aware strategy works**: Eliminates intent violations while only asking clarifying questions for 9.3% of ambiguous queries

## How to Reproduce

```bash
# 1. Create and activate environment
uv venv && source .venv/bin/activate

# 2. Install dependencies
uv add openai httpx datasets numpy scipy scikit-learn matplotlib seaborn tqdm sentence-transformers torch bert-score

# 3. Set API keys
export OPENAI_API_KEY="your-key"
export OPENROUTER_API_KEY="your-key"

# 4. Run experiments (~50 min)
cd src && python run_experiments.py

# 5. Run analysis and generate figures
python analyze_results.py
```

## File Structure

```
.
├── REPORT.md                    # Full research report with results
├── README.md                    # This file
├── planning.md                  # Experimental design and methodology
├── literature_review.md         # Synthesized literature review (25 papers)
├── resources.md                 # Resource catalog
├── src/
│   ├── config.py                # Configuration and hyperparameters
│   ├── data_loader.py           # Dataset loading and sampling
│   ├── llm_client.py            # LLM API client (OpenAI, OpenRouter)
│   ├── metrics.py               # Evaluation metrics (semantic sim, NLI, edit ratio)
│   ├── intent_classifier.py     # Embedding-based intent classification
│   ├── run_experiments.py       # Main experiment runner (Exp 1-3)
│   └── analyze_results.py       # Statistical analysis and visualization
├── results/
│   ├── data/                    # Raw experiment results (JSON)
│   └── plots/                   # Generated plots
├── figures/                     # Publication-quality figures
│   ├── fig1_intent_violation_rates.png
│   ├── fig2_metric_distributions.png
│   ├── fig3_edit_vs_similarity.png
│   ├── fig4_strategy_comparison.png
│   ├── fig5_metric_validation.png
│   └── fig6_by_dataset.png
├── datasets/                    # BANKING77, CLINC150, PAWS, STS-B, ClariQ, Qulac
├── papers/                      # 25 downloaded research papers
└── code/                        # Cloned baseline repositories
```

## Datasets Used

| Dataset | Size | Purpose |
|---------|------|---------|
| BANKING77 | 13,083 examples, 77 intents | Banking customer service intent classification |
| CLINC150 | 23,700 examples, 150 intents | Multi-domain virtual assistant intent classification |

## Models Tested

- **GPT-4.1** (OpenAI) - temperature=0
- **Claude Sonnet 4.5** (Anthropic via OpenRouter) - temperature=0
