
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
# login("your token here")
model_name = "mistralai/Mistral-7B-Instruct-v0.1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    load_in_8bit=True, 
    device_map="auto"
)

model.save_pretrained("./quantized_model")
tokenizer.save_pretrained("./quantized_model")

def generate_text(prompt):
   inputs = tokenizer(prompt, return_tensors="pt")
   outputs = model.generate(inputs["input_ids"], max_length=200)
   return tokenizer.decode(outputs[0], skip_special_tokens=True)

model = "mistal 7b"

generated_texts_df = pd.dataframe(columns=["model","topic", "prompt", "text"])
for topic in topics[:1]:
    for prompt in prompts[:1]:
        text = generate_text(prompt.format(topic=topic))
        generated_texts_df.loc[len(generated_texts_df)] = [model, topic, prompt, text]
        print(text)

