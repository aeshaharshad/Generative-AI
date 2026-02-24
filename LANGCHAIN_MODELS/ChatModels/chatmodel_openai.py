from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI(model="gpt-oss-120b",
                 api_key=os.getenv("OPENAI_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                temperature=0.6, max_tokens=30
                )
result=model.invoke("write a poem on software development")
print(result.content)
