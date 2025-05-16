from fastapi import APIRouter
from pydantic import BaseModel
from ..services.chat.chat_with_doc import ChatWithDocument
from fastapi.responses import JSONResponse
# from ..services.chat.test_agent import ChatWithDocument


REDIS_URL = "redis://localhost:6379/0"
router = APIRouter()

class AskRequest(BaseModel):
    question: str
    session_id: str

@router.post("/ask")
async def ask_question(request: AskRequest):
    chatbot = ChatWithDocument(redis_url=REDIS_URL, session_id=request.session_id)

    if chatbot.ask is None:
        return JSONResponse(content={"message":"No document found. Plase upload a document first."}, status_code=404)
    
    answer = chatbot.ask(request.question)
    return {"answer": answer}


# from fastapi import APIRouter
# from pydantic import BaseModel
# from ..services.chat.chat_with_doc import ChatWithDocument
# from fastapi.responses import JSONResponse

# # Import your IntentDetector class
# from ..services.intent.intent_classify import IntentDetector  # adjust import path accordingly

# REDIS_URL = "redis://localhost:6379/0"
# router = APIRouter()

# class AskRequest(BaseModel):
#     question: str
#     session_id: str

# # Initialize intent detector once (reuse for all requests)
# intent_detector = IntentDetector()

# @router.post("/ask")
# async def ask_question(request: AskRequest):
#     # Detect intent first
#     intent = intent_detector.detect_intent(request.question)
#     print(f"Detected intent: {intent}")  # or use logging

#     # You can branch logic here based on intent
#     if intent == "callback_request":
#         # For now, just return intent info or handle callback flow
#         return JSONResponse(content={"intent": intent, "message": "Callback request detected. Handling not implemented yet."})

#     # Otherwise proceed with normal chat flow
#     chatbot = ChatWithDocument(redis_url=REDIS_URL, session_id=request.session_id)

#     if chatbot.chain is None:
#         return JSONResponse(content={"message":"No document found. Please upload a document first."}, status_code=404)
    
#     answer = chatbot.ask(request.question)
#     return {"answer": answer}

