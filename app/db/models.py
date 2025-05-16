from pydantic import BaseModel, EmailStr

class ContactInfo(BaseModel):
    name: str
    phone: str
    email: EmailStr
