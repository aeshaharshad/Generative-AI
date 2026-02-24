from langchain_huggingface import HuggingFaceEmbeddings

embedding=HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')

# text="Islamabad is capital of Pakistan"

# vector=embedding.embed_query(text)

documents=[
    "Islamabad is capital of Pakistan",
    "Ankara is capital of Turkey",
    "Paris is capital of France"
]

result=embedding.embed_documents(documents)
print(str(result))

#print(str(vector))
