# Model Card — TF-IDF + Logistic Regression (AI vs Human Text Detector)

## Overview
A binary text classifier that labels a passage as **human-written (0)** or
**AI-generated (1)**. Features are word/bigram TF-IDF vectors; the classifier is
L2-regularized Logistic Regression. This is the model deployed behind the REST
API in `app/main.py`.

## Why this model (and not a transformer)
Five model families were benchmarked on the same splits (see
`reports/benchmark_results.csv`). All scored 0.98–0.998 F1 *in-distribution*, so
in-distribution accuracy did not distinguish them. The deciding factor was
**cross-dataset generalization** on an unseen external set:

| Model | Test F1 (in-dist.) | External F1 | External accuracy |
|---|---|---|---|
| **TF-IDF + LogReg (deployed)** | 0.9945 | **0.9366** | **0.8917** |
| BERT (fine-tuned) | 0.9822 | 0.9257 | 0.8697 |
| BiLSTM (scratch embed) | 0.9977 | 0.9034 | 0.8438 |
| RoBERTa (fine-tuned) | 0.9846 | 0.9026 | 0.8241 |

TF-IDF generalized best while being CPU-only, ~1 MB, and millisecond-latency.
The fine-tuned transformers overfit the training distribution's stylistic cues
and degraded more under domain shift (RoBERTa's human-class recall fell to ~0.05
on external data — it labelled almost everything "AI").

## Intended use
- First-pass, high-throughput screening flag for likely AI-generated text.
- A signal for human review — **not** a sole basis for consequential decisions.

## Out-of-scope / cautions
- **Not** an authoritative judgment of authorship. False positives have real
  costs (e.g. wrongly flagging a student), so a human must adjudicate.
- Trained on English essay-style text; behaviour on code, other languages, or
  very short texts is untested.
- Adversarial paraphrasing/humanizing tools are likely to evade it.

## Limitations revealed by evaluation
The near-perfect in-distribution scores (>0.99) are a sign the source dataset is
**easy** — the AI samples carry strong stylistic tells. The ~10-point drop on the
external set is the honest estimate of real-world performance. Treat 0.89
accuracy / 0.94 F1 (external) as the operative number, not 0.99.

## Training data
Kaggle "AI vs Human Text" CSV (`text`, `generated`). Duplicates dropped before a
stratified 70/15/15 split; the vectorizer is fit on the training split only.

## Metrics & reproduction
`python -m src.benchmark` regenerates the comparison;
`python -m src.pipeline` trains and saves the deployed artifact.

## Ethical considerations
AI-detection errors can unfairly harm individuals. This model is a triage aid;
deployments should keep a human in the loop, disclose its use, and provide an
appeals path.
