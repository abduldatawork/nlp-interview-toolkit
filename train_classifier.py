from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sentence_transformers import SentenceTransformer, util
import torch
import json
import pandas as pd

# Load our previously saved csv file
df = pd.read_csv("interview_questions.csv")

# Create a dictionary to map labels to the IDs
label_map = dict(enumerate(df["category"].astype("category").cat.categories))
num_labels = len(label_map)

# Split the data into train and test dataframes
train_df, test_df = train_test_split(df, train_size=0.8, random_state=42, stratify=df["label"])

# Convert the new train and test dataframes into Hugging Face Dataset objects
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# ======= Tokenize (converting into tokens) the datasets ======
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    # 'padding="max_length"' ensures all inputs are the same size
    return tokenizer(examples["instruction"], padding="max_length", truncation=True)

print("Tokenizing datasets...")
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# ====== Define the model =======
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

# ====== Set up / Create the trainer
training_args = TrainingArguments(
    output_dir="./results",                    # Where to save the model
    eval_strategy="epoch",                      # Check performance every epoch
    num_train_epochs=1,                         # 1 epoch is often enough for fine-tuning
    per_device_train_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset
)

# Train the model!!!!!!!!!!!!
print("Starting training...")
trainer.train()

# ====== Save the trained model, the label map, and the tokenizer ======
# Model
print("Training complete. Saving model...")
model_save_path = "./my_question_classifier"
trainer.save_model(model_save_path)

# Tokenizer
tokenizer.save_pretrained(model_save_path)

# Label map
with open("./my_question_classifier/label_map.json", "w") as f:
    json.dump(label_map, f)

print(f"Model saved to {model_save_path}")

#####################################################
#                                                   #
# ====== Building Sementic Similarity Engine ====== #
#                                                   #
#####################################################

# No training here, we will just use a pre-trained model (Sentence-BERT)

# Load the model
print("Loading Sentence-BERT model...")             # "all-MiniLM-L6-v2" is a fantastic, fast, all-purpose model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract questions form our dataframe
questions = df["instruction"].tolist()

# Create and cache embeddings
print(f"Creating embeddings for {len(questions)} questions...")
question_embeddings = model.encode(questions, convert_to_tensor=True)

print("Embeddings created.")

# Create Finder function
def find_similar_questions(query, top_k=3):
    print(f"\nFinding results for query: '{query}'")
    
    # Convert the new query into an embedding
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Compute cosine similarity between the query and all our questions
    cos_scores = util.cos_sim(query_embedding, question_embeddings)[0]
    
    # Find the top_k highest scores
    top_results = torch.topk(cos_scores, k=top_k+1)         # +1 to exclude the query itself if it is in the list
    
    print("Top results:")
    for i, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
        if questions[idx] == query:                         # Skip self-comparison
            continue
        print(f" {i+1}, {questions[idx]} (Score: {score:.4f})")