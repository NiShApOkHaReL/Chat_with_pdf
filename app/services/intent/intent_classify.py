from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

class IntentDetector:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-2.0-flash")
        self.intent_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
You are an intent classification system. Classify the intent of the following user query into one of these intents:
- callback_request (user wants to be contacted)
- general_question (user asks a question)
- other

User query: {query}

Intent:
"""

        )

    def detect_intent(self, query: str) -> str:
        prompt_text = self.intent_prompt.format(query=query)
        response = self.llm.invoke([HumanMessage(content=prompt_text)])
        print(response)
        intent = response.content.strip().lower()
        if "callback_request" in intent:
            return "callback_request"
        elif "general_question" in intent:
            return "general_question"
        else:
            return "other"
        


