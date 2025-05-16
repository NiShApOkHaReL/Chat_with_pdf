from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv


load_dotenv()

class VectorStore:
    """Vector store for RAG-based document retrieval."""

    def __init__(self):
        self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vector_store = None
       

    def split_documents(self, documents: list[Document], chunk_size=500, chunk_overlap=100) -> list[Document]:
        """Splits documents into smaller chunks for better embedding and retrieval."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_documents(documents)

    def add_documents(self, documents: list[Document]):
            """Embeds and stores documents in FAISS vector store."""
            split_docs = self.split_documents(documents)
            self.vector_store = FAISS.from_documents(split_docs, self.embedding_model)
            self.vector_store.save_local("faiss_index")


# if __name__ == "__main__":
#     vs = VectorStore()

#     documents = [Document(page_content="""LangGraph is built for developers who want to build powerful, adaptable AI agents. 
#     Developers choose LangGraph for:

#     Reliability and controllability. Steer agent actions with moderation checks and human-in-the-loop approvals. LangGraph persists context for long-running workflows.
#     Low-level and extensible. Build custom agents with fully descriptive, low-level primitives.
#     First-class streaming support. With token-by-token streaming and streaming of intermediate steps, LangGraph gives users visibility into agent reasoning.
#     """)]

#     vs.add_documents(documents)
#     results = vs.search("What is langchain?")

#     for res in results:
#         print(res.page_content)   
    