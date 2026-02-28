from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate   
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
import os

#open source model cant give structured output by default
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


#1st prompt - detailed report (prompt send to model extract respose)

template1=PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

#2nd prompt - summary (extracted response then again sent to same model, extracts its response then final output to user)
template2=PromptTemplate(
    template='Write a 5 line summary on following text. \n {text}',
    input_variables=['text']
)

parser=StrOutputParser()

chain=template1 | model | parser | template2 | model | parser

result=chain.invoke({'topic':'black hole'})

print(result)