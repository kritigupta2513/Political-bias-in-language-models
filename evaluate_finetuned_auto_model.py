import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error
import pandas as pd
import numpy as np
import os

# Configuration
MODEL_NAME = "prajjwal1/bert-tiny"
MODEL_PATH = "models/models_wo_lora/results_triplet_auto_wo_lora_5_epochs/checkpoint-1749"
TEST_DATA_PATH = "processed_data/random/test.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
MAX_LEN = 512

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Define Dataset
class EvalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx], 
            padding="max_length", 
            truncation=True, 
            max_length=MAX_LEN, 
            return_tensors="pt"
        )
        return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(0), self.labels[idx]

# Load test data
def load_test_data():
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Test data not found: {TEST_DATA_PATH}")

    df = pd.read_pickle(TEST_DATA_PATH)
    return df["content"].tolist(), df["bias"].tolist()

texts, labels = load_test_data()
dataset = EvalDataset(texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load the trained model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=3).to(DEVICE)
model.eval()

# Evaluation
preds, targets = [], []

with torch.no_grad():
    for input_ids, attention_mask, label in dataloader:
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        targets.extend(label.numpy())

# Calculate metrics
accuracy = accuracy_score(targets, preds)
macro_f1 = f1_score(targets, preds, average="macro")
mae = mean_absolute_error(targets, preds)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro-F1 Score: {macro_f1:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
