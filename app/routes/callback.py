from fastapi import APIRouter, HTTPException
from app.db.models import ContactInfo
from app.db.connections import collection

router = APIRouter()

@router.post("/store-contact")
async def store_contact(contact: ContactInfo):
    try:
        result = collection.insert_one(contact.model_dump())
        return {"status": "success", "inserted_id": str(result.inserted_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store contact info: {e}")
