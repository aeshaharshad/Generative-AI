from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
#Paragraphs embeddings can also be generated not just sentence
embedding=OpenAIEmbeddings(model='text-embedding-3-small',dimensions=32)

documents=[
    "Islamabad is capital of Pakistan",
    "Ankara is capital of Turkey",
    "Paris is capital of France"
]

result=embedding.embed_documents(documents)
print(str(result))
