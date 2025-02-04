from pydantic import BaseModel,Field 
from enum import Enum
from typing import Optional, List
from datetime import datetime 

class ModelName(str,Enum):
    LLAMA3_8B = "llama3-8b-8192"
    LLAMA3_70B = "llama3-70b-8192"
    GEMMA_7B_IT = "gemma-7b-it"
    MIXTRAL_8X7B = "mixtral-8x7b-32768"

class QueryInput(BaseModel):
    question:str
    session_id: Optional[str] = None
    model: ModelName = Field(default=ModelName. LLAMA3_70B)

class QueryResponse(BaseModel):
    answer:str
    session_id:str
    model:ModelName

class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime

class DeleteFileRequest(BaseModel):
    file_id: int
