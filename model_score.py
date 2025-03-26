import os
import torch
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the political bias classifier
MODEL_NAME = "kritigupta/political-bias-roBERTa-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Load CSV file with semicolon separator
# csv_file = "mistral_7b_generated_texts.csv"
csv_file = "cleaned_mistral_7b_generated_texts.csv"
# csv_file = "llama3_8b_generated_texts.csv"
# csv_file = "qwen_7b_generated_texts.csv"
df = pd.read_csv(csv_file, sep=";")

# Ensure full column values are printed
pd.set_option("display.max_colwidth", None)

# Define batch size for efficient processing
batch_size = 50
all_predictions = []
all_confidences = []

# Process text in batches
for i in range(0, len(df), batch_size):
    print(f"Processing batch {i} to {i+batch_size}...")
    
    temp = df.iloc[i:i + batch_size]
    texts = temp["text"].to_list()  # Adjust column name if necessary

    # Tokenize inputs
    encodings = tokenizer(texts, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

    # Run model inference without computing gradients
    with torch.no_grad():
        logits = model(**encodings).logits

    # Convert logits to probabilities
    probabilities = F.softmax(logits, dim=-1)

    # Get predicted labels (0, 1, or 2)
    batch_predictions = logits.argmax(dim=-1).cpu().numpy()

    # Get confidence scores (highest probability for each prediction)
    batch_confidences = probabilities.max(dim=-1).values.cpu().numpy()

    # Store results
    all_predictions.extend(batch_predictions)
    all_confidences.extend(batch_confidences)

# Add predictions and confidence scores to DataFrame
df["bias_category"] = all_predictions
df["confidence_score"] = all_confidences

# Save the classified data
output_file = "classified_clean_responses_mistral_7b.csv"
# output_file = "classified_responses_llama3_8b.csv"
# output_file = "classified_responses_qwen_7b.csv"
df.to_csv(output_file, sep=";", index=False)

# Print sample results
print(df.head())

print(f"\nClassification complete. Results saved to {output_file}")
