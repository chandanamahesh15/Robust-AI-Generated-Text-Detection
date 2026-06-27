# Robust AI-Generated Text Detection Benchmark

## Overview
This project is an end-to-end NLP benchmark for detecting AI-generated vs human-written text. It compares classical machine learning, deep learning, transformer fine-tuning, and LLM embedding-based approaches.

## Problem Statement
With the rise of large language models, distinguishing human-written and AI-generated text has become an important NLP problem. This project focuses not only on high internal accuracy, but also on external generalization across unseen LLM-generated text.

## Dataset
Main dataset:
- Kaggle AI vs Human Text dataset
- 487K+ labeled examples
- Labels:
  - 0 = Human-written
  - 1 = AI-generated

External validation:
- Hugging Face dataset with human text and generated text from Gemma, Mistral, Qwen, and LLaMA.

## Models
| Model | Type |
|---|---|
| Majority Class Baseline | Baseline |
| Stratified Baseline | Baseline |
| TF-IDF + Logistic Regression | Classical ML |
| BiLSTM | Deep Learning |
| BERT | Transformer |
| RoBERTa | Transformer |
| Qwen2.5 Embeddings + Logistic Regression | LLM Feature Extraction |

## Key Results

| Model | Internal Test F1 | External F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 0.9945 | 0.9366 |
| BiLSTM | 0.9977 | 0.9034 |
| BERT | 0.9822 | 0.9257 |
| RoBERTa | 0.9846 | 0.9026 |
| Qwen2.5 + Logistic Regression | 0.9852 | Not evaluated externally yet |

## Key Finding
Although BiLSTM achieved the highest internal test F1, TF-IDF + Logistic Regression generalized best on external unseen LLM-generated text.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, TensorFlow, Keras, PyTorch, Hugging Face Transformers, Qwen, FastAPI, Docker, MLflow.

## How to Run

```bash
pip install -r requirements.txt
