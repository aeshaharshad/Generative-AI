import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field

load_dotenv()
model= ChatOpenAI(
    model="gpt-oss-120b",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# schema-> how dict will look

class Person(BaseModel):
    name:str = Field(description='Name of the person')
    age : int =Field(gt=18, description='Age of the person')
    city:str = Field(description='Name of the city the person belongs to')

parser=PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# prompt=template.invoke({'place':'Pakistani'})

# print(prompt)

# result=model.invoke(prompt)

# final_result=parser.parse(result.content)

chain=template | model | parser
final_result=chain.invoke({'place':'Turkish'})

print(final_result)