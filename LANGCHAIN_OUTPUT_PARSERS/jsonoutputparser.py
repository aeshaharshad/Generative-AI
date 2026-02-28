from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser=JsonOutputParser()

template=PromptTemplate(
    template='Give me name,age and city of fictional person \n {format_instruction}',
    input_variables={},
    partial_variables={'format_instruction':parser.get_format_instructions()}

)


# prompt=template.format()
# result = model.invoke(prompt) #response given by llm
# final_result=parser.parse(result.content)


#above three lines by chain

chain=template | model | parser
result=chain.invoke({})
print(result)
