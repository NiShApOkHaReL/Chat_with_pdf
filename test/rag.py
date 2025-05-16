from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from pydantic import BaseModel, Field
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.prompts import MessagesPlaceholder
import os
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
import asyncio

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: list[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []


class AnswerQuery:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=GOOGLE_API_KEY, model="models/text-embedding-004"
        )
        self.sessions = {}

    def get_by_session_id(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.sessions:
            self.sessions[session_id] = InMemoryHistory()
        return self.sessions[session_id]

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if (session_id) not in self.sessions:
            self.sessions[(session_id)] = InMemoryHistory()
        return self.sessions[(session_id)]

    async def in_memory_answer_query(
        self, query: str, session_id: str, prompt: str = "You are a helpful assistant."
    ):
        """ "
        Chat with bot using In Memory Implementation.
        """
        model = ChatGoogleGenerativeAI(
            temperature=0,
            model="gemini-2.0-flash"
        )
        # Vector store first!
        index_path = "./faiss_index"
        vector_store =  FAISS.load_local(
                    index_path,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )

        # Retriever
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 8, "lambda_mult": 0.5},
        )



        # Get documents
        compressed_docs = retriever.get_relevant_documents(query)

        # Build context from docs
        context_chunks = [doc.page_content for doc in compressed_docs]
        context = "\n\n".join(context_chunks)
        # Prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    prompt,
                ),
                MessagesPlaceholder(variable_name="history"),
                (
                    "human",
                    """Context:
{context}

Question: {question}""",
                ),
            ]
        )

        chain = prompt | model

        with_message_history = RunnableWithMessageHistory(
            chain,
            get_session_history=self.get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )
        # Run the memory-wrapped chain with both context & question
        response = with_message_history.invoke(
            {
                "question": query,
                "context": context,
                "history": self.get_session_history(session_id),
            },
            config={"configurable": {"session_id": session_id}},
        )
        # Stream the response
        return {"answer": response}
    
if __name__ == "__main__":
    async def main():
        obj = AnswerQuery()
        # result = await obj.in_memory_answer_query(query="Do you know my name?", session_id="123")
        # print(result)
        print(obj.get_session_history(session_id="123"))

    asyncio.run(main())