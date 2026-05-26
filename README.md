# Injury Intelligence

An NLP pipeline over OSHA Severe Injury Reports (2015–2017), combining text classification and a RAG-powered Q&A interface.

---

## What it does

- **EDA**: explores OSHA injury narratives, cleans and lemmatizes text, analyzes class distributions
- **Classification**: trains a TF-IDF + Logistic Regression baseline and a fine-tuned DistilBERT model to classify injury event types. Experiments tracked with MLflow.
- **RAG Q&A**: embeds narratives using `all-MiniLM-L6-v2` and stores them in a FAISS vector index. User queries are embedded and matched against the index, with retrieved records passed to Claude as context for grounded answers.
- **Streamlit app**: interactive interface for querying the dataset in plain English

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-2.x-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange.svg)](https://scikit-learn.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![MLflow](https://img.shields.io/badge/MLflow-3.x-0194E2.svg)](https://mlflow.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Meta_AI-4267B2?style=flat)](https://faiss.ai/)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude-191919?style=flat)](https://www.anthropic.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://osha-injury-intelligence.streamlit.app)

---

## Live Demo

View the live demo at [https://osha-injury-intelligence.streamlit.app/](https://osha-injury-intelligence.streamlit.app/)

### Screenshots

![Query Interface](app/assets/screenshot_question.png)
![Answer and Retrieved Records](app/assets/screenshot_answer.png)

---

## Project Structure

```
injury-intelligence/
├── app/
│   ├── assets/
│   │   └── safety_first.jpg
│   └── app.py                  # Streamlit RAG interface
├── notebooks/
│   ├── 1_eda.ipynb             # Data exploration and preprocessing
│   ├── 2_classification.ipynb  # TF-IDF + DistilBERT classification
│   └── 3_rag.ipynb             # Embeddings, FAISS index, RAG pipeline
├── data/                       # Raw and processed data (raw gitignored)
├── mlruns/                     # MLflow experiment tracking (gitignored)
└── README.md
```

--- 

## Data

[OSHA Accident and Injury Data](https://www.kaggle.com/datasets/ruqaiyaship/osha-accident-and-injury-data-1517) via Kaggle. Contains 4,427 severe workplace injury records with free-text narrative descriptions.