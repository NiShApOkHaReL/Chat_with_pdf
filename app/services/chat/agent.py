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

# System prompt to guide the model to use tools properly
# system_message = SystemMessage(content="""
# You are an assistant that can perform calculations and save user contact information.
# When the user provides a name, phone number, and email, automatically call the save_user_contact tool with those details.
# If the user provides a math expression, use the multiply or add tools to calculate the result.
# Respond naturally and inform the user of the results.
# """)

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
            "session_id: User_12, question: What is the document about?",
            # "My name is xyz, phone is 1234567890, and email is abc@gmail.com"


    ]
    for q in queries:
        print(f"User: {q}")
        answer = run_agent(q)
        print(f"Agent: {answer}\n")


# import os
# from langchain_core.tools import tool
# from langchain_core.messages import HumanMessage, ToolMessage
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.vectorstores import FAISS
# from langchain_redis import RedisChatMessageHistory
# from pymongo.mongo_client import MongoClient
# from pymongo.server_api import ServerApi
# import certifi

# # MongoDB setup (reuse your existing setup)
# uri = "mongodb+srv://nisha:nisha@ragcluster.6zcesdr.mongodb.net/?retryWrites=true&w=majority&appName=RagCluster"
# client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())
# db = client.test_db
# collection = db["user_db"]

# # Existing tools
# @tool
# def multiply(a: int, b: int) -> int:
#     """Given 2 numbers a and b this tool returns their product"""
#     return a * b

# @tool
# def add(a: int, b: int) -> int:
#     """Given 2 numbers a and b this tool returns their sum"""
#     return a + b

# @tool
# def save_user_contact(name: str, phone: str, email: str) -> str:
#     """Saving contact information to the database"""
#     try:
#         user_data = {"name": name, "phone": phone, "email": email}
#         result = collection.insert_one(user_data)
#         print("Saving to DB:", user_data)
#         if result.acknowledged:
#             return f"User contact saved with id {result.inserted_id}"
#         else:
#             return "Failed to save user contact."
#     except Exception as e:
#         return f"Error saving user contact: {e}"

# # ChatWithDocument class adapted for tool usage
# class ChatWithDocument:
#     def __init__(self, session_id: str, redis_url: str = "redis://localhost:6379/0"):
#         self.session_id = session_id
#         self.redis_url = redis_url
#         self.chat_history = RedisChatMessageHistory(session_id=session_id, redis_url=redis_url)
#         self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

#         index_path = "faiss_index"
#         faiss_file = os.path.join(index_path, "index.faiss")

#         if os.path.exists(faiss_file):
#             self.vector_store = FAISS.load_local(
#                 index_path,
#                 embeddings=self.embedding_model,
#                 allow_dangerous_deserialization=True
#             )
#             print("Loaded FAISS index with", len(self.vector_store.docstore._dict), "documents.")
#             self._initialize_chain()
#         else:
#             print("No FAISS index found. Chatbot will be ready after uploading documents.")
#             self.vector_store = None
#             self.chain = None

#     def get_formatted_history(self):
#         messages = self.chat_history.messages
#         formatted = []
#         for msg in messages:
#             role = msg.type
#             content = msg.content
#             if role == "human":
#                 formatted.append(f"User: {content}")
#             else:
#                 formatted.append(f"Assistant: {content}")
#         return "\n".join(formatted)

#     def _initialize_chain(self):
#         self.retriever = self.vector_store.as_retriever(
#             search_type="mmr",
#             search_kwargs={"k": 5, "fetch_k": 20}
#         )
#         self.llm = ChatGoogleGenerativeAI(
#             temperature=0.5,
#             model="gemini-2.0-flash"
#         )
#         self.prompt = PromptTemplate(
#             template="""
# You are a helpful assistant.
# Use the following conversation history and context to answer the user's question.
# If the context is not sufficient, say you don't know.
# If user says "call me back", "ask my details", or "I have some problems please contact me", then say "Please enter your name, phone, and email address."
# If user provides their details (name, phone, email), extract them and call the save_contact_info tool to save to the database. Then say "Thank you for providing your details."
# If user asks questions based on the uploaded document, answer them accordingly.

# Conversation history:
# {history}

# Context:
# {context}

# Question:
# {question}
# """,
#             input_variables=['context', 'question', 'history']
#         )
#         self.chain = self.prompt | self.llm | StrOutputParser()

#     def ask(self, question: str) -> str:
#         if self.chain is None:
#             return "No documents found. Please upload a document first."

#         history_str = self.get_formatted_history()

#         try:
#             docs = self.retriever.invoke(question)
#             if not docs:
#                 print("No documents retrieved by the retriever.")
#                 context = ""
#             else:
#                 context = "\n\n".join(doc.page_content[:500] for doc in docs if hasattr(doc, 'page_content'))
#                 print(f"Retrieved {len(docs)} documents for context.")
#         except Exception as e:
#             print(f"Error during document retrieval: {e}")
#             context = ""

#         inputs = {
#             "context": context,
#             "question": question,
#             "history": history_str
#         }

#         try:
#             print(f"Inputs to chain: {inputs}")
#             answer = self.chain.invoke(inputs)
#         except Exception as e:
#             print(f"Error during chain invocation: {e}")
#             return "An error occurred while processing your request."

#         self.chat_history.add_user_message(question)
#         self.chat_history.add_ai_message(answer)

#         return answer

# # Maintain chat sessions to reuse ChatWithDocument instances
# chat_sessions = {}

# # Define the tool wrapping ChatWithDocument.ask
# @tool
# def chat_with_document(session_id: str, question: str) -> str:
#     """
#     Chat with uploaded documents using the given session_id for chat history.
#     """
#     session_id = "User_1"
#     if session_id not in chat_sessions:
#         chat_sessions[session_id] = ChatWithDocument(session_id=session_id)
#     chat_instance = chat_sessions[session_id]
#     return chat_instance.ask(question)

# # Initialize the LLM and bind all tools including the new chat_with_document tool
# llm = ChatGoogleGenerativeAI(
#     temperature=0,
#     model="gemini-2.0-flash"
# )

# llm_with_tools = llm.bind_tools([multiply, add, save_user_contact, chat_with_document])

# def run_agent(user_input: str):
#     user_message = HumanMessage(content=user_input)
#     ai_response = llm_with_tools.invoke([user_message])

#     if ai_response.tool_calls:
#         tool_call = ai_response.tool_calls[0]
#         tool_name = tool_call["name"]
#         tool_args = tool_call["args"]

#         if tool_name == multiply.name:
#             tool_result = multiply.invoke(tool_args)
#         elif tool_name == add.name:
#             tool_result = add.invoke(tool_args)
#         elif tool_name == save_user_contact.name:
#             tool_result = save_user_contact.invoke(tool_args)
#         elif tool_name == chat_with_document.name:
#             tool_result = chat_with_document.invoke(tool_args)
#         else:
#             tool_result = f"Tool '{tool_name}' not found."

#         tool_message = ToolMessage(
#             content=str(tool_result),
#             tool_call_id=tool_call["id"]
#         )

#         final_response = llm_with_tools.invoke([user_message, ai_response, tool_message])
#         return final_response.content
#     else:
#         return ai_response.content

# # Example usage
# if __name__ == "__main__":
#     queries = [
#         # "What is the product of 5 and 6?",
#         "Can you tell me about the uploaded documents?",
#         # "Please call me back.",
#         # "My name is John, phone is 1234567890, and email is john@example.com."
#     ]
#     session_id = "User_1"  # example session id for chat history
#     for q in queries:
#         print(f"User: {q}")
#         answer = run_agent(q)
#         print(f"Agent: {answer}\n")
