import os
import glob
import torch
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def classify_political_bias(csv_file, output_file=None, model_path="roberta-bias-detector/model.safetensors", model_name="kritigupta/political-bias-roBERTa-triplet-loss", add_model_name=False):
    """
    Classify political bias in texts using the RoBERTa model with direct argmax classification.
    
    Args:
        csv_file (str): Path to the CSV file containing texts to classify.
        output_file (str, optional): Path to save the results. If None, will use a default naming pattern.
        model_path (str): Path to the local model.safetensors file. If file doesn't exist, will use HuggingFace model.
        model_name (str): HuggingFace model name for the political bias classifier and tokenizer.
        add_model_name (bool): Whether to add model name to the output dataframe.
    
    Returns:
        pandas.DataFrame: DataFrame with the original data and classification results.
    """
    # Check if using local model file or HuggingFace model
    if os.path.exists(model_path):
        print(f"Loading political bias classifier from local file: {model_path}")
        # For local model, we still need the tokenizer from HuggingFace
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load the model from roberta-bias-detector directory
        model_dir = "roberta-bias-detector"
        print(f"Loading model from directory: {model_dir}")
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_dir,
            local_files_only=True
        )
    else:
        print(f"Local model not found, loading from HuggingFace: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    model.eval()
    
    # Load CSV file with semicolon separator
    print(f"Loading data from: {csv_file}")
    df = pd.read_csv(csv_file, sep=";")
    
    # Add model name to dataframe if requested
    if add_model_name:
        # Extract model name from filename
        model_name_from_file = os.path.basename(csv_file)
        if model_name_from_file.startswith("preprocessed_"):
            model_name_from_file = model_name_from_file[12:]
        model_name_from_file = model_name_from_file.split("_generated")[0]
        
        # If model column doesn't already exist, add it
        if "model" not in df.columns:
            df["model"] = model_name_from_file
    
    # Ensure full column values are printed
    pd.set_option("display.max_colwidth", None)
    
    # Define batch size for efficient processing
    batch_size = 50
    all_predictions = []
    all_confidences = []
    all_left_probs = []
    all_center_probs = []
    all_right_probs = []
    
    # Process text in batches
    for i in range(0, len(df), batch_size):
        print(f"Processing batch {i} to {min(i+batch_size, len(df))}...")
        
        temp = df.iloc[i:i + batch_size]
        texts = temp["text"].to_list()  # Adjust column name if necessary
        
        # Tokenize inputs
        encodings = tokenizer(texts, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        
        # Run model inference without computing gradients
        with torch.no_grad():
            logits = model(**encodings).logits
        
        # Convert logits to probabilities
        probabilities = F.softmax(logits, dim=-1)
        
        # Get predicted labels (0=left, 1=center, 2=right)
        batch_predictions = logits.argmax(dim=-1).cpu().numpy()
        
        # Get confidence scores (highest probability for each prediction)
        batch_confidences = probabilities.max(dim=-1).values.cpu().numpy()
        
        # Get individual class probabilities
        num_classes = probabilities.shape[1]
        print(f"Model has {num_classes} output classes")
        
        # Always get left probability (index 0)
        batch_left_probs = probabilities[:, 0].cpu().numpy()
        
        # Handle models with different number of output classes
        if num_classes >= 2:
            batch_center_probs = probabilities[:, 1].cpu().numpy()
        else:
            # If only 1 class, set center probs to zeros
            batch_center_probs = torch.zeros_like(probabilities[:, 0]).cpu().numpy()
            
        if num_classes >= 3:
            batch_right_probs = probabilities[:, 2].cpu().numpy()
        else:
            # If only 2 classes, set right probs to zeros
            batch_right_probs = torch.zeros_like(probabilities[:, 0]).cpu().numpy()
        
        # Store results
        all_predictions.extend(batch_predictions)
        all_confidences.extend(batch_confidences)
        all_left_probs.extend(batch_left_probs)
        all_center_probs.extend(batch_center_probs)
        all_right_probs.extend(batch_right_probs)
    
    # Add predictions and confidence scores to DataFrame
    df["bias_score"] = all_predictions
    df["confidence_score"] = all_confidences
    df["left_probability"] = all_left_probs
    df["center_probability"] = all_center_probs
    df["right_probability"] = all_right_probs
    
    # bias_score already contains the numerical values (0=left, 1=center, 2=right)
    
    # Save the classified data
    if output_file is None:
        # Generate output filename based on input filename
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        output_file = f"classified_responses_{base_name}.csv"
    
    df.to_csv(output_file, sep=";", index=False)
    print(f"\nClassification complete. Results saved to {output_file}")
    print("\nSample results:")
    print(df[["bias_score", "confidence_score"]].head())
    
    # Print distribution of bias scores (0=left, 1=center, 2=right)
    print("\nBias score distribution:")
    print(df["bias_score"].value_counts())
    
    return df

def process_all_files_in_directory(directory="preprocessed_model_generations", output_file="model_bias_score/combined_model_bias_comparison.csv", model_path="model.safetensors"):
    """
    Process all CSV files in the specified directory and combine results into a single CSV file.
    
    Args:
        directory (str): Directory containing CSV files to process
        output_file (str): Path to save the combined results
        model_path (str): Path to the local model.safetensors file
    
    Returns:
        pandas.DataFrame: Combined DataFrame with all results
    """
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {directory}")
        return None
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each file and store results
    all_results = []
    for csv_file in csv_files:
        print(f"\nProcessing {csv_file}...")
        # Process file and add model name to the dataframe
        result_df = classify_political_bias(csv_file, model_path=model_path, add_model_name=True)
        all_results.append(result_df)
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save combined results
    combined_df.to_csv(output_file, sep=";", index=False)
    print(f"\nCombined results saved to {output_file}")
    
    # Print bias distribution by model
    print("\nBias score distribution by model:")
    bias_distribution = combined_df.groupby(["model", "bias_score"]).size().unstack(fill_value=0)
    print(bias_distribution)
    
    # Calculate percentages for easier comparison
    bias_percentages = combined_df.groupby(["model"])["bias_score"].value_counts(normalize=True).unstack(fill_value=0) * 100
    print("\nBias score distribution by model (percentages):")
    print(bias_percentages.round(2))
    
    # Create a file with responses grouped by bias category
    create_bias_category_file(combined_df, "bias_category_responses.csv")
    
    return combined_df

def create_bias_category_file(df, output_file="model_bias_score/bias_category_responses.csv"):
    """
    Create a CSV file with responses grouped by bias category.
    
    Args:
        df (pandas.DataFrame): DataFrame containing classified responses
        output_file (str): Path to save the results
    
    Returns:
        None
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Map bias_score to human-readable categories
    # Check if we have any bias_score with value 2 (right) in the data
    has_right_category = 2 in df_copy["bias_score"].unique()
    
    if has_right_category:
        # Use 3-class mapping
        bias_category_map = {0: "left", 1: "center", 2: "right"}
    else:
        # Use 2-class mapping
        bias_category_map = {0: "left", 1: "center"}
        
    df_copy["bias_category"] = df_copy["bias_score"].map(bias_category_map)
    
    # Save the DataFrame with bias categories
    df_copy.to_csv(output_file, sep=";", index=False)
    print(f"\nResponses grouped by bias category saved to {output_file}")
    
    # Print distribution of bias categories
    print("\nBias category distribution:")
    print(df_copy["bias_category"].value_counts())
    
    # Group by model and bias category to see distribution across models
    bias_by_model = df_copy.groupby(["model", "bias_category"]).size().unstack(fill_value=0)
    print("\nBias category distribution by model:")
    print(bias_by_model)
    
    # Calculate percentages for easier comparison
    bias_percentages = df_copy.groupby(["model"])["bias_category"].value_counts(normalize=True).unstack(fill_value=0) * 100
    print("\nBias category distribution by model (percentages):")
    print(bias_percentages.round(2))

def main():
    # Check if command line arguments are provided
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify political bias in texts using RoBERTa model")
    parser.add_argument("--all", action="store_true", help="Process all files in the preprocessed_model_generations directory")
    parser.add_argument("--model-path", type=str, default="model.safetensors", help="Path to the local model.safetensors file")
    parser.add_argument("input", nargs="?", help="Input CSV file or output file name when used with --all")
    parser.add_argument("output", nargs="?", help="Output CSV file")
    
    # Handle old-style command line arguments for backward compatibility
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        args = parser.parse_args(["--all"] + sys.argv[2:])
    else:
        args = parser.parse_args()
    
    if args.all:
        # Process all files in the preprocessed_model_generations directory
        directory = "preprocessed_model_generations"
        output_file = args.input if args.input else "combined_model_bias_comparison.csv"
        process_all_files_in_directory(directory, output_file, args.model_path)
    elif args.input:
        # Process a single file
        csv_file = args.input
        output_file = args.output
        classify_political_bias(csv_file, output_file, model_path=args.model_path)
    else:
        print("Usage:")
        print("  Process all files: python political_bias_batch_classifier.py --all [output_csv_file] [--model-path MODEL_PATH]")
        print("  Process single file: python political_bias_batch_classifier.py <input_csv_file> [output_csv_file] [--model-path MODEL_PATH]")
        print("\nExamples:")
        print("  python political_bias_batch_classifier.py --all combined_model_bias.csv")
        print("  python political_bias_batch_classifier.py --all --model-path model.safetensors combined_model_bias.csv")
        print("  python political_bias_batch_classifier.py preprocessed_model_generations/preprocessed_llama3_8b_generated_texts.csv classified_responses_llama3_8b.csv")
        print("  python political_bias_batch_classifier.py --model-path model.safetensors preprocessed_model_generations/preprocessed_llama3_8b_generated_texts.csv classified_responses_llama3_8b.csv")

if __name__ == "__main__":
    main()