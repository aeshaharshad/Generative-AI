from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.header('Research Tool')

llm=HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm)

user_input=st.text_input('Enter your prompt')

if st.button:
    result=model.invoke(user_input)
    st.write(result.content)

