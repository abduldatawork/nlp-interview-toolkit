import streamlit as st
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from huggingface_hub import hf_hub_download

# ======= Loading our Classification Model =======
@st.cache_resource          # decorator for caching ressources
def load_classifier():
    print("Loading classifier...")
    # Load model from local
    # model_path = "./my_question_classifier"
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model from the Hub
    model_id = "Aboudougafar/my_question_classifier"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    
    # Loading the label map we saved
    # with open(f"{model_id}/label_map.json", "r") as f:
    #     label_map = json.load(f)
    #     # JSON keys are strings, let's convert them to integers
    #     label_map = {int(k): v for k, v in label_map.items()}
    
    # Download the label_map.json file from the Hub
    label_map_path = hf_hub_download(
        repo_id=model_id,
        filename="label_map.json"
    )

    # Now we open the file using the safe path
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
        # JSON keys are strings, let's convert them to integers
        label_map = {int(k): v for k, v in label_map.items()}
    return tokenizer, model, label_map

tokenizer, model, label_map = load_classifier()
print("Classifier loaded.")

# ====== Loading our Similarity Engine ======
@st.cache_resource
def load_similarity_engine():
    print("Loading similarity engine...")
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    df = pd.read_csv("interview_questions.csv")
    questions = df["instruction"].tolist()
    
    # Create and cache embeddings
    embeddings = sbert_model.encode(questions, convert_to_tensor=True)
    
    return sbert_model, questions, embeddings

sbert_model, questions, question_embeddings = load_similarity_engine()
print("Similarity engine loaded.")

# ====== Application Logic ======
def run_analysis(query):
    # Classification
    st.subheader("Predicted Category")
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    predicted_class_id = logits.argmax().item()
    category = label_map[predicted_class_id]
    st.success(f"**{category}**")
    
    # Similarity
    st.subheader("Similar Questions")
    query_embedding = sbert_model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, question_embeddings)[0]
    top_results = torch.topk(cos_scores, k=3)
    
    for i, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
        st.markdown(f"**{i+1}.** {questions[idx]} *(Score: {score:.4f})*")

# ====== Building the Streamlit UI ======
st.title("AI-Powered Interview Prep Toolkit")
st.write("Enter a technical question to see its category and find similar questions.")

user_query = st.text_area("Enter your question:", "e.g., How do you reverse a string in Python?")

if st.button("Analyze Question"):
    if user_query:
        with st.spinner("Running models..."):
            run_analysis(user_query)
    else:
        st.error("Please enter a question.")
else:
    st.warning("Please click the 'Analyze Question' button.")
