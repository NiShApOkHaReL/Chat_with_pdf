from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pymongo import MongoClient
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import certifi
import sys
import os

# Add the root directory to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from chat_with_doc import ChatWithDocument


uri = "mongodb+srv://nisha:nisha@ragcluster.6zcesdr.mongodb.net/?retryWrites=true&w=majority&appName=RagCluster"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())

db = client.test_db
collection = db["user_db"]



# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Given 2 numbers a and b this tool returns their product"""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Given 2 numbers a and b this tool returns their sum"""
    return a + b


@tool
def save_user_contact(name: str, phone: str, email: str) -> str:
    """Saving contact information to the database"""
    try:
        user_data = {"name": name, "phone": phone, "email": email}
        result = collection.insert_one(user_data)
        print("Saving to DB:", user_data)

        if result.acknowledged:
            return f"User contact saved with id {result.inserted_id}"
        else:
            return "Failed to save user contact."
    except Exception as e:
        return f"Error saving user contact: {e}"
    
chat_doc_sessions = {}

@tool
def chat_with_document(session_id: str, question: str) -> str:
    """
    Chat with uploaded documents using the given session_id for chat history, user can say call me also.
    """
    if session_id not in chat_doc_sessions:
        chat_doc_sessions[session_id] = ChatWithDocument(session_id=session_id)
    
    chat_instance = chat_doc_sessions[session_id]
    
    # Use the ask method to get answer
    answer = chat_instance.ask(question)
    return answer

    

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    temperature=0,
    model="gemini-2.0-flash"
)

# Bind the tools to the LLM
llm_with_tools = llm.bind_tools([multiply, add,save_user_contact, chat_with_document])


def run_agent(user_input: str):
    # Step 1: Send user input to LLM with tools bound
    user_message = HumanMessage(content=user_input)
    ai_response = llm_with_tools.invoke([user_message])
    # print(f"AI Response: {ai_response}")
    # print(f"Tool Calls: {ai_response.tool_calls}")
    # Step 2: Check if LLM wants to call a tool
    if ai_response.tool_calls:
        # For simplicity, handle the first tool call only
        tool_call = ai_response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # Step 3: Execute the tool dynamically by matching tool name
        # Map tool names to tool functions
        tools_map = {
            multiply.name: multiply,
            add.name: add,
            save_user_contact.name: save_user_contact,
            chat_with_document.name: chat_with_document,
        }

        tool_func = tools_map.get(tool_name)
        if tool_func:
            try:
                tool_result = tool_func.invoke(tool_args)
            except Exception as e:
                tool_result = f"Error invoking tool '{tool_name}': {e}"
        else:
            tool_result = f"Tool '{tool_name}' not found."

        # Step 4: Create a ToolMessage with the tool result
        tool_message = ToolMessage(
            content=str(tool_result),
            tool_call_id=tool_call["id"]
        )
        # print(f"Tool message: {tool_message}")
        # print(f"User message: {user_message}")
        # print(f"AI response: {ai_response}")

        # Step 5: Send the tool result back to the LLM to get final response
        final_response = llm_with_tools.invoke([user_message, ai_response, tool_message])
        # print(f"Final response: {final_response}")
        # Step 6: Return the final response
        return final_response.content

    else:
        # No tool call, just return LLM output
        return ai_response.content

# Example usage
if __name__ == "__main__":
    queries = [
            # "What is the product of 5 and 6?"
            "session_id: User_12, question: What is the document about?"
            # "My name is xyz, phone is 1234567890, and email is abc@gmail.com"


    ]
    for q in queries:
        print(f"User: {q}")
        answer = run_agent(q)
        print(f"Agent: {answer}\n")

