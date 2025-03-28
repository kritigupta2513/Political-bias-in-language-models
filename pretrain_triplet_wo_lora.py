import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_scheduler
import pandas as pd
import os
import random
import numpy as np
from tqdm import tqdm

# Configuration
MODEL_NAME = "prajjwal1/bert-tiny"
LEARNING_RATE = 2e-5
MAX_GRAD_NORM = 1
BATCH_SIZE = 16
MAX_LEN = 512
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DATA_PATH = "processed_data/media/train.pkl"
SEED = 42

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# Define Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.norm(anchor - positive, p=2, dim=1)
        neg_dist = torch.norm(anchor - negative, p=2, dim=1)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0).mean()
        return loss

# Define Triplet Dataset
class TripletDataset(Dataset):
    def __init__(self, triplets, tokenizer):
        self.triplets = triplets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]
        inputs = self.tokenizer([anchor, positive, negative], 
                                padding="max_length", truncation=True, 
                                max_length=MAX_LEN, return_tensors="pt")
        return inputs["input_ids"], inputs["attention_mask"]

# Load dataset
def load_triplet_data():
    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Training data not found: {TRAIN_DATA_PATH}")

    df = pd.read_pickle(TRAIN_DATA_PATH)
    
    triplets = []
    df_by_topic = df.groupby("topic")

    for _, group in df_by_topic:
        same_ideology = group.groupby("bias")
        for _, pos_group in same_ideology:
            neg_group = df[df["bias"] != pos_group["bias"].iloc[0]]
            
            if len(pos_group) > 1 and len(neg_group) > 0:
                pos_samples = pos_group.sample(n=min(len(pos_group), len(neg_group)), replace=False, random_state=SEED)
                neg_samples = neg_group.sample(n=min(len(pos_group), len(neg_group)), replace=False, random_state=SEED)

                for (anchor, positive, negative) in zip(pos_samples["content"], pos_samples["content"].shift(-1), neg_samples["content"]):
                    if positive is not None:
                        triplets.append((anchor, positive, negative))

    return triplets[:35017]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
triplets = load_triplet_data()
dataset = TripletDataset(triplets, tokenizer)

# Set generator for deterministic data shuffling
generator = torch.Generator()
generator.manual_seed(SEED)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)

# Load pre-trained model without LoRA
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

# Print the number of trainable parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable Parameters: {num_params:,}")

# Training setup
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * EPOCHS)
triplet_loss_fn = TripletLoss(margin=1.0)

# Training loop with progress bar
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)

    for batch in progress_bar:
        input_ids, attention_mask = batch
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        
        outputs = model(input_ids=input_ids.view(-1, MAX_LEN), attention_mask=attention_mask.view(-1, MAX_LEN))
        embeddings = outputs.last_hidden_state[:, 0, :].contiguous()   # Using CLS token

        anchor, positive, negative = embeddings[::3], embeddings[1::3], embeddings[2::3]
        loss = triplet_loss_fn(anchor, positive, negative)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))

    print(f"Epoch {epoch+1} - Loss: {total_loss / len(dataloader):.4f}")

# Save model
model.save_pretrained("models/triplet_wo_lora_5_epochs")
tokenizer.save_pretrained("models/triplet_wo_lora_5_epochs")
