from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from langchain_redis import RedisChatMessageHistory
from pymongo import MongoClient
import json

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import certifi

# LangChain agent and tools imports
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

class ContactInfoSaver:
    def __init__(self):
        # self.client = MongoClient(mongo_uri)
        # self.db = self.client[db_name]
        # self.collection = self.db["contacts"]

        uri = "mongodb+srv://nisha:nisha@ragcluster.6zcesdr.mongodb.net/?retryWrites=true&w=majority&appName=RagCluster"

        # Create a new client and connect to the server
        self.client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())

        self.db = self.client.RAG_db
        self.collection = self.db["user_db"]



    def save_contact_info(self, name: str, phone: str, email: str) -> str:
        doc = {"name": name, "phone": phone, "email": email}
        self.collection.insert_one(doc)
        return "Thank you for providing your details."

class ChatWithDocument:
    def __init__(self, session_id: str, redis_url: str = "redis://localhost:6379/0", mongo_uri="mongodb+srv://nisha:nisha@ragcluster.6zcesdr.mongodb.net/?retryWrites=true&w=majority&appName=RagCluster"):
        self.session_id = session_id
        self.redis_url = redis_url
        self.mongo_uri = mongo_uri

        self.chat_history = RedisChatMessageHistory(session_id=session_id, redis_url=redis_url)
        self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

        index_path = "faiss_index"
        faiss_file = os.path.join(index_path, "index.faiss")

        if os.path.exists(faiss_file):
            try:
                self.vector_store = FAISS.load_local(
                    index_path,
                    embeddings=self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                print("Loaded FAISS index with", len(self.vector_store.docstore._dict), "documents.")
            except Exception as e:
                raise RuntimeError(f"Failed to load FAISS index: {e}")

            self._initialize_agent()
        else:
            print("No FAISS index found. Chatbot will be ready after uploading documents.")
            self.vector_store = None
            self.agent = None

    def get_formatted_history(self):
        messages = self.chat_history.messages  
        formatted = []
        for msg in messages:
            role = msg.type  
            content = msg.content
            if role == "human":
                formatted.append(f"User: {content}")
            else:
                formatted.append(f"Assistant: {content}")
        return "\n".join(formatted)

    def _initialize_agent(self):
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20}
        )

        self.llm = ChatGoogleGenerativeAI(
            temperature=0.5,
            model="gemini-2.0-flash",
            convert_system_message_to_human=True
        )

        contact_saver = ContactInfoSaver(mongo_uri=self.mongo_uri)

        def save_contact_info_tool_func(text: str) -> str:
            """
            Expects LLM to output contact info as JSON string like:
            {
                "name": "John Doe",
                "phone": "+1234567890",
                "email": "john@example.com"
            }
            """
            try:
                data = json.loads(text)
                name = data.get("name")
                phone = data.get("phone")
                email = data.get("email")
                if name and phone and email:
                    contact_saver.save_contact_info(name, phone, email)
                    return "Contact information saved successfully."
                else:
                    return "Incomplete contact details. Please provide name, phone, and email."
            except json.JSONDecodeError:
                return "Failed to parse contact details. Please provide them as a JSON object."

        save_contact_tool = Tool(
            name="save_contact_info",
            func=save_contact_info_tool_func,
            description="""
            Save user's contact information to MongoDB. 
            Input MUST be a JSON string like: 
            {"name": "John Doe", "phone": "+1234567890", "email": "john@example.com"}
            """
        )


        # Prompt instructing the LLM to output JSON when user provides details
        self.prompt = PromptTemplate(
            template="""
You are a helpful assistant.
Use the following conversation history and context to answer the user's question.
If the context is not sufficient, say you don't know.
If user says "call me back", "ask my details", or "I have some problems please contact me", then say "Please enter your name, phone, and email address."
If user provides their details (name, phone, email), respond ONLY with a JSON object like:
{
  "name": "User Name",
  "phone": "1234567890",
  "email": "user@example.com"
}
The agent will save this info automatically.
If user asks questions based on the uploaded document, answer them accordingly.

Conversation history:
{history}

Context:
{context}

Question:
{question}
""",
            input_variables=['context', 'question', 'history']
        )

        # Initialize agent with tool and LLM
        self.agent = initialize_agent(
            tools=[save_contact_tool],
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # Works better with Gemini
            verbose=True
        )

    def ask(self, question: str) -> str:
        if self.vector_store is None or self.agent is None:
            return "No documents found. Please upload a document first."

        history_str = self.get_formatted_history()

        try:
            docs = self.retriever.invoke(question)
            context = "\n\n".join(doc.page_content[:500] for doc in docs if hasattr(doc, 'page_content')) if docs else ""
        except Exception as e:
            print(f"Error during document retrieval: {e}")
            context = ""

        input_text = f"Context:\n{context}\n\nConversation history:\n{history_str}\n\nQuestion:\n{question}"


        try:
            answer = self.agent.invoke({"input": input_text})
        except Exception as e:
            print(f"Error during agent invocation: {e}")
            return "An error occurred while processing your request."

        self.chat_history.add_user_message(question)
        self.chat_history.add_ai_message(answer)

        return answer
