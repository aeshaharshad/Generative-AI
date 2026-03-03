from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel ,RunnablePassthrough
load_dotenv()

prompt1=PromptTemplate(
    template='Generate a joke about {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Explain fllowig joke {text}',
    input_variables=['text']
)

model = ChatOpenAI(
    model="gpt-oss-120b",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

parser = StrOutputParser()


joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({'topic':'cricket'}))
