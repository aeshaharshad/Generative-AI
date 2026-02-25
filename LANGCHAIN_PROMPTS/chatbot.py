from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=512
)

model = ChatHuggingFace(llm=llm)

# Store messages in LangChain's format
chat_history = [SystemMessage(content='Your are AI engineer')]

while True:
    user_input = input('You: ')
    
    if user_input.lower() == 'exit':
        break
    
    # Create a HumanMessage for the user input
    user_message = HumanMessage(content=user_input)
    chat_history.append(user_message)
    
    try:
        # Invoke the model with the properly formatted chat history
        result = model.invoke(chat_history)
        
        # Create an AIMessage for the response
        ai_message = AIMessage(content=result.content)
        chat_history.append(ai_message)
        
        print("AI: ", result.content)
    except Exception as e:
        print(f"Error: {e}")
        # If there's an error, remove the last user message from history
        chat_history.pop()

print("Chat history:", chat_history)