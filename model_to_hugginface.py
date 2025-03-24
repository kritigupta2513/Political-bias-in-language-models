from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import HfApi, HfFolder

repo_name = "kritigupta/political-bias-roBERTa-base"

hf_token = "Your token here"

api = HfApi()
api.create_repo(repo_name, token=hf_token)

api.upload_folder(
    folder_path="./results/checkpoint-7512",  # Path to the folder containing the model and tokenizer
    repo_id=repo_name,
    token=hf_token
)