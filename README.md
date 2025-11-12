---
title: AI Interview Prep Toolkit
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
python_version: "3.13.5"
---
# 🎙️ AI Interview Prep Toolkit

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sql-coach-app-opksosjzvdcezmfnhaubq9.streamlit.app/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/Aboudougafar/nlp-interview-app)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3100/)

This project is an end-to-end NLP application built to help job seekers and students prepare for technical interviews. It provides a "smart" search tool that goes beyond simple keyword matching.

This app was built as a portfolio project to demonstrate skills in fine-tuning, deploying, and building applications with HuggingFace Transformers and Streamlit.

## 🚀 The Problem

Technical interview prep platforms have thousands of questions. A user who fails a question about "SQL JOINs" has no easy way to find *other*, similar "SQL JOIN" questions. A simple keyword search is inefficient.

This tool solves the problem by providing two core features.

## ✨ Features

1.  **Smart Question Classifier:**
    * **What it does:** Instantly classifies a user's question into a technical category (e.g., "SQL", "Python", "Arrays", "System Design").
    * **How it works:** Uses a **DistilBERT** model fine-tuned for sequence classification on a custom dataset of interview questions.

2.  **Semantic Similarity Search:**
    * **What it does:** Finds the top 5 most similar questions from a vector database of 15,000+ questions.
    * **How it works:** Uses a **Sentence-Transformer** (`all-MiniLM-L6-v2`) model to generate embeddings for the entire question corpus. It then compares the user's query embedding to the corpus to find the closest matches using cosine similarity.

## 🛠️ Tech Stack

* **Model:** HuggingFace `transformers` (DistilBERT, Sentence-Transformers), PyTorch
* **App:** Streamlit
* **Data:** Pandas, NumPy
* **Deployment:** GitHub, Streamlit Community Cloud, HuggingFace Hub (for model hosting)

## 💻 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/abduldatawork/nlp-interview-toolkit.git](https://github.com/abduldatawork/nlp-interview-toolkit.git)
    cd nlp-interview-toolkit
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
