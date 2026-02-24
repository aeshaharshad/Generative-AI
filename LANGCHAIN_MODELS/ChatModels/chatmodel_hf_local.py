import os
import torch

os.environ["HF_HOME"] = "D:/huggingface_cache"

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

model_id = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100
)

llm = HuggingFacePipeline(pipeline=pipe)

result = llm.invoke("What is colour of apple?")
print(result)