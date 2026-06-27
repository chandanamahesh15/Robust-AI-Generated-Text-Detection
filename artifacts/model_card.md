# Model Card: AI-Generated Text Detection

## Model Purpose
This project detects whether a given text sample is likely human-written or AI-generated.

## Models Evaluated
- Majority Class Baseline
- Stratified Baseline
- TF-IDF + Logistic Regression
- BiLSTM with scratch embeddings
- BERT fine-tuning
- RoBERTa fine-tuning
- Qwen2.5 frozen embeddings + Logistic Regression

## Best Internal Model
BiLSTM achieved the highest internal test F1 score.

## Best External Generalization Model
TF-IDF + Logistic Regression achieved the strongest external F1 score on unseen LLM-generated text.

## Intended Use
Educational and research benchmark for studying AI-generated text detection.

## Not Intended For
This model should not be used as the only evidence for academic misconduct, hiring decisions, legal decisions, or disciplinary actions.

## Limitations
- May fail on short text.
- May fail on heavily edited AI text.
- May misclassify polished human writing.
- Performance may drop on new LLMs or unseen writing domains.
- AI text detection is not perfectly reliable.

## Ethical Considerations
False positives can unfairly label human writing as AI-generated.