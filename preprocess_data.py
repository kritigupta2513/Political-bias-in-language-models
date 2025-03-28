import pandas as pd
import json
import os
from pathlib import Path

# Define base directories
base_dir = "Article-Bias-Prediction/data/splits"
json_dir = "Article-Bias-Prediction/data/jsons"
output_dir = "processed_data"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define main folders and splits
folders = ["media", "random"]
splits = ["train", "valid", "test"]

# Function to process and save DataFrames
def process_and_save_data(folder, split):
    """
    Reads TSV split files, loads corresponding JSON articles, 
    creates a DataFrame, and saves it as a Pickle file.
    """
    split_file = Path(base_dir) / folder / f"{split}.tsv"
    output_file = Path(output_dir) / folder / f"{split}.pkl"
    os.makedirs(output_file.parent, exist_ok=True)  # Create subdirectories if needed

    # Read split file
    try:
        split_df = pd.read_csv(split_file, sep='\t')
        article_ids = split_df['ID'].tolist()
    except FileNotFoundError:
        print(f"File not found: {split_file}")
        return

    # Read corresponding JSON files
    articles = []
    for article_id in article_ids:
        json_path = Path(json_dir) / f"{article_id}.json"
        try:
            with open(json_path, 'r') as f:
                article_data = json.load(f)
                articles.append(article_data)
        except FileNotFoundError:
            print(f"Could not find JSON file for article ID: {article_id}")

    # Create and save DataFrame
    df = pd.DataFrame(articles)
    df.to_pickle(output_file)
    print(f"Saved: {output_file}")

# Process all splits for both folders
for folder in folders:
    for split in splits:
        process_and_save_data(folder, split)