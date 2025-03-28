import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from sklearn.metrics import f1_score, mean_absolute_error
import numpy as np
import pandas as pd

MODEL_PATH = "models/models_wo_lora/triplet_wo_lora_5_epochs"
LEARNING_RATE = 2e-5
MAX_GRAD_NORM = 1
BATCH_SIZE = 16
MAX_LEN = 512
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DATA_PATH = "processed_data/random/train.pkl"
VALID_DATA_PATH = "processed_data/random/valid.pkl"
SEED = 42
NUM_LABELS = 3  # Number of bias categories

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# Load dataset
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len):
        df = pd.read_pickle(file_path)
        self.texts = df['content'].tolist()
        self.labels = df['bias'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return { 
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Create datasets
train_dataset = TextDataset(TRAIN_DATA_PATH, tokenizer, MAX_LEN)
valid_dataset = TextDataset(VALID_DATA_PATH, tokenizer, MAX_LEN)

# Load the pre-trained model with a classification head
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=NUM_LABELS).to(DEVICE)

def compute_metrics(p):
    predictions = torch.tensor(p.predictions)
    predictions = torch.argmax(predictions, dim=1)

    accuracy = (predictions == torch.tensor(p.label_ids)).float().mean().item()
    macro_f1 = f1_score(p.label_ids, predictions.numpy(), average='macro')
    mae = mean_absolute_error(p.label_ids, predictions.numpy())

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'mae': mae,
    }

training_args = TrainingArguments(
    output_dir="./results_triplet_auto_wo_lora_5_epochs",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=100,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
