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



























# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import os
# from langchain_redis import RedisChatMessageHistory
# from pymongo import MongoClient

# class ChatWithDocument:
#     def __init__(self, session_id: str, redis_url: str = "redis://localhost:6379/0"):
#         self.session_id = session_id
#         self.redis_url = redis_url
#         # self.chain = None

#         self.chat_history = RedisChatMessageHistory(session_id=session_id, redis_url=redis_url)
#         self.embedding_model = GoogleGenerativeAIEmbeddings( model="models/text-embedding-004")

#         index_path = "faiss_index"
#         faiss_file = os.path.join(index_path, "index.faiss")

#         if os.path.exists(faiss_file):
#             try:
#                 self.vector_store = FAISS.load_local(
#                     index_path,
#                     embeddings=self.embedding_model,
#                     allow_dangerous_deserialization=True
#                 )
#                 print("Loaded FAISS index with", len(self.vector_store.docstore._dict), "documents.")
#             except Exception as e:
#                 raise RuntimeError(f"Failed to load FAISS index: {e}")

            
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
#             search_type="mmr",  # maximal marginal relevance for better search result.
#             search_kwargs={"k": 5, "fetch_k": 20}
#         )

#         self.llm = ChatGoogleGenerativeAI(
#             temperature=0.5,
#             model="gemini-2.0-flash"
#         )

#         self.prompt = PromptTemplate(
#             template="""
#         You are a helpful assistant.
#         Use the following conversation history and context to answer the user's question.
#         If the context is not sufficient, say you don't know.
#         If user says "call me back", "ask my details", or "I have some problems please contact me", then say "Please enter your name, phone, and email address."
#         If user provides their details (name, phone, email), extract them and call the save_contact_info tool to save to the database. Then say "Thank you for providing your details."
#         If user asks questions based on the uploaded document, answer them accordingly.

#     Conversation history:
#     {history}

#     Context:
#     {context}

#     Question:
#     {question}
#     """,
#             input_variables=['context', 'question', 'history']
#         )

#         self.chain = self.prompt | self.llm | StrOutputParser()

#     def ask(self, question: str) -> str:
#         if self.chain is None:
#             return "No documents found. Please upload a document first."

#         # Get formatted chat history
#         history_str = self.get_formatted_history()

#         # Retrieve relevant documents
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
  

#         # Construct inputs for the chain
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

#         # Save user question and assistant answer to Redis history
#         self.chat_history.add_user_message(question)
#         self.chat_history.add_ai_message(answer)

#         return answer












#     def _initialize_chain(self):
#         self.retriever = self.vector_store.as_retriever(
#             search_type="mmr",  # maximal marginal relevance for better search result.
#             search_kwargs={"k": 5, "fetch_k": 20}
#         )

#         self.llm = ChatGoogleGenerativeAI(
#             temperature=0,
#             model="gemini-2.0-flash"
#         )

#         self.prompt = PromptTemplate(
#             template="""
# You are a helpful assistant.
# Use the following conversation history and context to answer the user's question.
# If the context is not sufficient, say you don't know.

# Conversation history:
# {history}

# Context:
# {context}

# Question:
# {question}
# """,
#             input_variables=['context', 'question', 'history']
#         )

#         def format_docs(docs):
#             if not docs:
#                 print("No documents retrieved by the retriever.")  # Debugging log

#                 return ""  # Return an empty string if no documents are retrieved
#             if isinstance(docs, list):
#                 return "\n\n".join(doc.page_content for doc in docs if hasattr(doc, 'page_content'))
#             print(f"Unexpected input to format_docs: {type(docs)} - {docs}")  # Debugging log
#             return str(docs)  # Convert unexpected input to string for debugging
  

#         self.parallel_chain = RunnableParallel({
#             'context': self.retriever | RunnablePassthrough() | format_docs,
#             'question': RunnablePassthrough(),
#             'history': RunnablePassthrough()
#         })

#         self.chain = self.parallel_chain | self.prompt | self.llm | StrOutputParser()

#     def ask(self, question: str) -> str:
#         if self.chain is None:
#             return "No documents found. Please upload a document first."

#         # Get formatted chat history
#         history_str = self.get_formatted_history()

#         # Retrieve relevant docs and ask question with history
#         inputs = {
#             "context": "",  # Ensure context is a string, even if no documents are retrieved
#             "question": question,
#             "history": history_str
#         }

#         try:
#             print(f"Inputs to chain: {inputs}")  # Debugging log
#             answer = self.chain.invoke(inputs)
#         except Exception as e:
#             print(f"Error during chain invocation: {e}")
#             return "An error occurred while processing your request."

#         # Save user question and assistant answer to Redis history
#         self.chat_history.add_user_message(question)
#         self.chat_history.add_ai_message(answer)

#         return answer

# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnableParallel, RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import os




# class ChatWithDocument:
#     def __init__(self):


#         self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")



#         index_path = "faiss_index"
#         faiss_file = os.path.join(index_path, "index.faiss")

#         if os.path.exists(faiss_file):
#             try:
#                 self.vector_store = FAISS.load_local(
#                     index_path,
#                     embeddings=self.embedding_model,
#                     allow_dangerous_deserialization=True
#                 )
#                 print("Loaded FAISS index with", len(self.vector_store.docstore._dict), "documents.")
#             except Exception as e:
#                 raise RuntimeError(f"Failed to load FAISS index: {e}")

#             # Initialize retriever and chain only if vector_store exists
#             self._initialize_chain()
#         else:
#             print("No FAISS index found. Chatbot will be ready after uploading documents.")
#             self.vector_store = None
#             self.chain = None

#     def _initialize_chain(self):
#         self.retriever = self.vector_store.as_retriever(
#             search_type="mmr", #maximal marginal relevance for better search result.
#             search_kwargs={"k": 5, "fetch_k": 20}
#         )

      
#         self.llm = ChatGoogleGenerativeAI(
#             temperature=0,
#             model="gemini-2.0-flash"
#         )


   
#         self.prompt = PromptTemplate(
#             template="""
#             You are a helpful assistant.
#             Use the following context and conversation history to answer the user's question.
#             If the context is not sufficient, say you don't know.
#             Context: {context}
            
#             Question: {question}
#             """,
#             input_variables=['context', 'question']
#         )


#         # Formating retrieved docs
#         def format_docs(docs):
#             return "\n\n".join(doc.page_content for doc in docs)

#         # Retrieval + Question parallel input
#         self.parallel_chain = RunnableParallel({
#             'context': self.retriever | RunnablePassthrough() | format_docs,
#             'question': RunnablePassthrough()
#         })


#         self.chain = self.parallel_chain | self.prompt | self.llm | StrOutputParser()
        


#     def ask(self, question: str) -> str:
#         if self.chain is None:
#             return "No documents found. Please upload a document first."
#         return self.chain.invoke(question)



