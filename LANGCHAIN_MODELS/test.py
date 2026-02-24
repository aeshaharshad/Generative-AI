import os
from dotenv import load_dotenv
load_dotenv()
# print("KEY:", os.getenv("OPENAI_API_KEY"))
# print("TYPE:", type(os.getenv("OPENAI_API_KEY")))

print(os.getenv("HUGGINGFACEHUB_API_TOKEN"))