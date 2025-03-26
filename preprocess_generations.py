import pandas as pd
import re

# Load the CSV file
csv_file = "mistral_7b_generated_texts.csv"
df = pd.read_csv(csv_file, sep=";")

# Define the regex pattern to remove everything before "The article should be about 200 words" and exclude the phrase itself
pattern = r"^(.*?)(The article should be about 200 words.)"

# Function to remove prompt content before the target phrase and exclude it
def remove_prompt(text, pattern):
    # Remove everything before the target phrase and the phrase itself
    cleaned_text = re.sub(pattern, "", text)
    return cleaned_text

# Apply the cleaning function to the "text" column
df["cleaned_text"] = df["text"].apply(lambda x: remove_prompt(x, pattern))

# Save the cleaned DataFrame to a new CSV
output_file = "cleaned_mistral_7b_generated_texts.csv"
df.to_csv(output_file, sep=";", index=False)

print(f"Cleaned CSV saved to {output_file}")
