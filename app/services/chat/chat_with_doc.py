from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from langchain_redis import RedisChatMessageHistory
from pymongo import MongoClient

class ChatWithDocument:
    def __init__(self, session_id: str, redis_url: str = "redis://localhost:6379/0"):
        self.session_id = session_id
        self.redis_url = redis_url
        # self.chain = None

        self.chat_history = RedisChatMessageHistory(session_id=session_id, redis_url=redis_url)
        self.embedding_model = GoogleGenerativeAIEmbeddings( model="models/text-embedding-004")

        index_path = os.path.abspath("faiss_index")
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

            
            self._initialize_chain()
        else:
            print("No FAISS index found. Chatbot will be ready after uploading documents.")
            self.vector_store = None
            self.chain = None



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

    def _initialize_chain(self):
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",  # maximal marginal relevance for better search result.
            search_kwargs={"k": 5, "fetch_k": 20}
        )

        self.llm = ChatGoogleGenerativeAI(
            temperature=0.5,
            model="gemini-2.0-flash"
        )

        self.prompt = PromptTemplate(
            template="""
        You are a helpful assistant.
        Use the following conversation history and context to answer the user's question.
        If the context is not sufficient, say you don't know.
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

        self.chain = self.prompt | self.llm | StrOutputParser()

    def ask(self, question: str) -> str:
        if self.chain is None:
            return "No documents found. Please upload a document first."

        # Get formatted chat history
        history_str = self.get_formatted_history()

        # Retrieve relevant documents
        try:
            docs = self.retriever.invoke(question)
            if not docs:
                print("No documents retrieved by the retriever.")
                context = ""
            else:
                context = "\n\n".join(doc.page_content[:500] for doc in docs if hasattr(doc, 'page_content'))
                print(f"Retrieved {len(docs)} documents for context.")
        except Exception as e:
            print(f"Error during document retrieval: {e}")
            context = ""
  

        # Construct inputs for the chain
        inputs = {
            "context": context,
            "question": question,
            "history": history_str
        }

        try:
            print(f"Inputs to chain: {inputs}")  
            answer = self.chain.invoke(inputs)
        except Exception as e:
            print(f"Error during chain invocation: {e}")
            return "An error occurred while processing your request."

        # Save user question and assistant answer to Redis history
        self.chat_history.add_user_message(question)
        self.chat_history.add_ai_message(answer)

        return answer




















