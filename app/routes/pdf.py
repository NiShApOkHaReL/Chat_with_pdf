from fastapi import APIRouter, UploadFile, File
import os
import shutil
from ..services.documents.process_document import DocumentTextExtractor
from ..services.vectorstore.vector_store import VectorStore  

router = APIRouter()


UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract text from PDF
    extractor = DocumentTextExtractor()
    documents = extractor.extract_text(file_path)

    # Store in vector store
    vector_store = VectorStore()
    vector_store.add_documents(documents)

    return {"message": f"File '{file.filename}' processed and stored in vector store."}
