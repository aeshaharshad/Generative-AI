from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


load_dotenv()

embedding=HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Apple is a sweet fruit that comes in red, green, and yellow colors.",
    "Bananas are rich in potassium and are usually yellow when ripe.",
    "Mango is known as the king of fruits and is very popular in summer.",
    "Strawberries are small red fruits with tiny seeds on the outside.",
    "Oranges are citrus fruits that are full of vitamin C."
]

query="Tell me about Strawberries"


doc_embeddings=embedding.embed_documents(documents) #gives 5 vectors of 300 dimension
query_embedding=embedding.embed_query(query)

print(cosine_similarity([query_embedding],doc_embeddings))
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

best_index = scores.argmax()

print("Most similar document:")
print(documents[best_index])


