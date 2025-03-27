import pandas as pd
import re

# Load the CSV file
csv_file = "model_generations/llama3_8b_generated_texts.csv"
df = pd.read_csv(csv_file, sep=";")

output_file = "preprocessed_model_generations/preprocessed_llama3_8b_generated_texts.csv"

prompt_sentences = [
    "Clearly state your opinion on the topic and pick a firm stance",
    "The article should be about 200 words",
    "Write a news article on",
    "Imagine you are a 25 year old US citizen",
    "Imagine you are a 60 year old US citizen"
]

def remove_prompt(text):
    sentences = re.split('\.|\?|\!', text)
    # print(sentences)
    for sentence in sentences:
        for prompt in prompt_sentences:
            if prompt in sentence:
                text = text.replace(sentence, '')
    return text

def remove_leading_period(text):
    if text[0] == '.':
        return text[1:]
    return text

def remove_last_sentence(text):
    sentences = re.split('\.|\?|\!', text)
    last_sentence = sentences[-1]
    if len(last_sentence) == 0:
        return text
    if last_sentence[-1] != '.' and last_sentence[-1] != '?' and last_sentence[-1] != '!':
        return text[:text.rfind(last_sentence)]
    return text
       

# Remove the prompt sentences from the text
for prompt in prompt_sentences:
    df["text"] = df["text"].apply(lambda x: remove_prompt(x))

#remove excess periods generated from previous step
df["text"] = df["text"].apply(lambda x: re.sub(r"\.+", '.', x))
df["text"] = df["text"].apply(lambda x: remove_leading_period(x))

# remove special characters from the text
df["text"] = df["text"].apply(lambda x: re.sub(r"[*^$#@]+", ' ', x))

#remove extra spaces from the text
df["text"] = df["text"].apply(lambda x: re.sub(r"\s+", ' ', x))

#remove extra newlines from the text
df["text"] = df["text"].apply(lambda x: re.sub(r"\n+", '\n', x))

#remove trailing and leading spaces from the text
df["text"] = df["text"].apply(lambda x: x.strip())

#if last sentence is truncated (doesnt end with punctuation), remove it
df["text"] = df["text"].apply(lambda x: remove_last_sentence(x))

#save the preprocessed data
df.to_csv(output_file, sep=";", index=False)
