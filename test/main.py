from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_redis import RedisChatMessageHistory

# Constants
REDIS_URL = "redis://localhost:6379/0"

# FastAPI app
app = FastAPI()

# Request model
class ChatRequest(BaseModel):
    session_id: str
    input: str

# Chatbot class
class ChatBot:
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.prompt = self._create_prompt()
        self.llm = self._create_llm()
        self.chain = self._create_chain()
        self.chain_with_history = self._create_chain_with_history()

    def _create_prompt(self):
        return ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful AI assistant."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )

    def _create_llm(self):
        return ChatGoogleGenerativeAI(
            temperature=0.7,
            model="gemini-2.0-flash"
        )

    def _create_chain(self):
        return self.prompt | self.llm

    def _get_history(self, session_id: str) -> BaseChatMessageHistory:
        return RedisChatMessageHistory(session_id=session_id, redis_url=self.redis_url)

    def _create_chain_with_history(self):
        return RunnableWithMessageHistory(
            self.chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="history"
        )

    def chat(self, session_id: str, user_input: str) -> str:
        response = self.chain_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        return response.content

# Create chatbot instance
chatbot = ChatBot(redis_url=REDIS_URL)

# Endpoint to handle chat
@app.post("/chat")
async def chat(request: ChatRequest):
    result = chatbot.chat(session_id=request.session_id, user_input=request.input)
    return {"response": result}


if __name__ == "__main__":
    chatb = ChatBot(redis_url=REDIS_URL)
    print(chatb.chat(session_id="123", user_input="Hello, I am Nisha. Who are you?"))
    print(chatb.chat(session_id="123", user_input="What is my Name?"))


