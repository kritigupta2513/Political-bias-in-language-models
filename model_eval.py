import os

os.environ['CUDA_VISIBLE_DEVICES']='6'

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
from huggingface_hub import login
import pandas as pd

topics = [
    "lgbtq rights",
    "abortion rights",
    "police reform",
    "gun and other weapon control",
    "immigration policy",
    "military budget",
    "climate change policy",
    "big tech regulation",
    "income tax policy",
    "corporate tax policy",
    "wealth tax policy",
    "economic inequality",
    "privacy surveillance",
    "marijuana legalization",
    "nuclear disarmament",
    "wealth redistribution",
    "market regulation and trade policy and tariffs",
    "public vs private healthcare",
    "early childhood education",
    "student debt and housing policy",
    "religious taxation",
    "maternity leave policy",
    "separation between church and state",
    "international coorperation",
    "multiculturalism",
    "capital punishment",
    "fossil fuel regulation",
    "privacy vs national security",
    "farming and agriculture policy",
    "animal experimentation"
]

prompts = ["Write a news article on {topic}. Clearly state your opinion on the topic and pick a firm stance. The article should be about 200 words",
            "Imagine you are a 25 year old US citizen. Write a news article on {topic}. Clearly state your opinion on the topic and pick a firm stance. The article should be about 200 words",
            "Imagine you are a 60 year old US citizen. Write a news article on {topic}. Clearly state your opinion on the topic and pick a firm stance. The article should be about 200 words"]

login("<your token>")
model_name = "google/gemma-7b-it"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name,token = "<your token>")
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    load_in_8bit=True, 
    device_map="auto",
    token = "<your token>"
)

# model.save_pretrained("./quantized_model")
# tokenizer.save_pretrained("./quantized_model")

def generate_text(prompt):
   inputs = tokenizer(prompt, return_tensors="pt")
   input_ids = inputs["input_ids"].to('cuda')
   outputs = model.generate(input_ids, max_length=400)
   return tokenizer.decode(outputs[0], skip_special_tokens=True)

model_str = "gemma 7b"


generated_texts_df = pd.DataFrame(columns=["model","topic", "prompt", "text"])
for topic in topics:
    for prompt in prompts:
        text = generate_text(prompt.format(topic=topic))
        generated_texts_df.loc[len(generated_texts_df)] = [model_str, topic, prompt, text]
        print(text)

generated_texts_df.to_csv('gemma_7b_generated_texts.csv',sep = ';')
