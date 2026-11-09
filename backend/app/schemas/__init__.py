from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime

# Token Schemas
class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    user_id: int
    full_name: str
    title: str

class TokenPayload(BaseModel):
    sub: Optional[str] = None
    role: Optional[str] = None

# User Schemas
class UserBase(BaseModel):
    email: EmailStr
    full_name: str
    title: str = "Dr."
    role: str = "RADIOLOGIST"

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

# Patient & Study Schemas
class ClinicalRequisition(BaseModel):
    age: Optional[int] = 52
    gender: Optional[str] = "M"
    smoking_status: Optional[str] = "NEVER" # NEVER, FORMER, ACTIVE_SMOKER
    pack_years: Optional[int] = 0
    chief_complaint: Optional[str] = "Routine clinical screening"
    prior_malignancy: Optional[str] = None

class PatientCreate(BaseModel):
    mrn: str
    age: Optional[int] = None
    gender: Optional[str] = None

class PatientResponse(PatientCreate):
    id: int
    anon_id: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class StudyCreate(BaseModel):
    patient_id: int
    modality: str
    body_part: str
    description: Optional[str] = None
    is_demo: bool = True

class FindingResponse(BaseModel):
    id: int
    label: str
    probability: float
    location: Optional[str] = None
    measurement_text: Optional[str] = None
    heatmap_path: Optional[str] = None
    status: str
    
    class Config:
        from_attributes = True

class SegmentationResponse(BaseModel):
    id: int
    structure_name: str
    volume_cm3: Optional[float] = None
    dimensions_mm: Optional[Dict[str, Any]] = None
    mask_path: str
    color_hex: str
    
    class Config:
        from_attributes = True

class AIAnalysisResponse(BaseModel):
    id: int
    study_id: int
    model_name: str
    model_version: str
    modality: str
    confidence_score: float
    is_uncertain: bool
    status: str
    latency_ms: float
    device_used: str
    created_at: datetime
    findings: List[FindingResponse] = []
    segmentations: List[SegmentationResponse] = []
    
    class Config:
        from_attributes = True

class StudyResponse(BaseModel):
    id: int
    patient_id: int
    study_instance_uid: str
    modality: str
    body_part: str
    study_date: datetime
    description: Optional[str] = None
    status: str
    is_demo: bool
    ai_analyses: List[AIAnalysisResponse] = []
    
    class Config:
        from_attributes = True

# Report Schemas
class ReportCreate(BaseModel):
    study_id: int
    clinical_history: Optional[str] = None
    technique: Optional[str] = None
    findings: str
    ai_summary: Optional[str] = None
    impression: str
    status: str = "DRAFT"

class ReportResponse(BaseModel):
    id: int
    study_id: int
    radiologist_id: Optional[int] = None
    status: str
    clinical_history: Optional[str] = None
    technique: Optional[str] = None
    findings: str
    ai_summary: Optional[str] = None
    impression: str
    disclaimer: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# Chat Schemas
class ChatMessageCreate(BaseModel):
    content: str
    study_id: Optional[int] = None

class ChatMessageResponse(BaseModel):
    id: int
    session_id: int
    sender: str
    content: str
    citations: List[Dict[str, Any]] = []
    confidence_info: Dict[str, Any] = {}
    created_at: datetime
    
    class Config:
        from_attributes = True

# RAG Schemas
class RAGQueryRequest(BaseModel):
    query: str
    study_id: Optional[int] = None
    top_k: int = 4

class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]] = []
    confidence_indicator: str
