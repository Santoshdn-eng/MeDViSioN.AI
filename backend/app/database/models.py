from datetime import datetime, timezone
import enum
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, JSON, Enum
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class UserRole(str, enum.Enum):
    ADMIN = "ADMIN"
    RADIOLOGIST = "RADIOLOGIST"
    DOCTOR = "DOCTOR"
    RESEARCHER = "RESEARCHER"

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    title = Column(String(100), default="Dr.")
    role = Column(String(50), default=UserRole.RADIOLOGIST.value)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, index=True)
    mrn = Column(String(100), unique=True, index=True, nullable=False)  # Medical Record Number
    anon_id = Column(String(100), unique=True, index=True, nullable=False) # De-identified ID
    age = Column(Integer, nullable=True)
    gender = Column(String(20), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    studies = relationship("Study", back_populates="patient", cascade="all, delete-orphan")

class Study(Base):
    __tablename__ = "studies"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    study_instance_uid = Column(String(255), unique=True, index=True, nullable=False)
    modality = Column(String(20), nullable=False)  # X-RAY, CT, MRI
    body_part = Column(String(100), nullable=False) # Chest, Brain, Lung, Abdomen
    study_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    description = Column(String(255), nullable=True)
    status = Column(String(50), default="UPLOADED") # UPLOADED, PROCESSING, COMPLETED, REVIEWED
    is_demo = Column(Boolean, default=True)
    
    patient = relationship("Patient", back_populates="studies")
    series = relationship("Series", back_populates="study", cascade="all, delete-orphan")
    ai_analyses = relationship("AIAnalysis", back_populates="study", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="study", cascade="all, delete-orphan")

class Series(Base):
    __tablename__ = "series"
    
    id = Column(Integer, primary_key=True, index=True)
    study_id = Column(Integer, ForeignKey("studies.id"), nullable=False)
    series_instance_uid = Column(String(255), unique=True, index=True, nullable=False)
    series_number = Column(Integer, default=1)
    description = Column(String(255), nullable=True)
    slice_count = Column(Integer, default=1)
    
    study = relationship("Study", back_populates="series")
    images = relationship("Image", back_populates="series", cascade="all, delete-orphan")

class Image(Base):
    __tablename__ = "images"
    
    id = Column(Integer, primary_key=True, index=True)
    series_id = Column(Integer, ForeignKey("series.id"), nullable=False)
    sop_instance_uid = Column(String(255), unique=True, index=True, nullable=False)
    instance_number = Column(Integer, default=1)
    file_path = Column(String(500), nullable=False)
    preview_path = Column(String(500), nullable=True)
    metadata_json = Column(JSON, default=dict)
    
    series = relationship("Series", back_populates="images")

class AIAnalysis(Base):
    __tablename__ = "ai_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    study_id = Column(Integer, ForeignKey("studies.id"), nullable=False)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    modality = Column(String(50), nullable=False)
    confidence_score = Column(Float, nullable=False)
    is_uncertain = Column(Boolean, default=False)
    status = Column(String(50), default="COMPLETED")
    latency_ms = Column(Float, default=0.0)
    device_used = Column(String(50), default="CPU")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    study = relationship("Study", back_populates="ai_analyses")
    findings = relationship("Finding", back_populates="analysis", cascade="all, delete-orphan")
    segmentations = relationship("Segmentation", back_populates="analysis", cascade="all, delete-orphan")

class Finding(Base):
    __tablename__ = "findings"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("ai_analyses.id"), nullable=False)
    label = Column(String(100), nullable=False)
    probability = Column(Float, nullable=False)
    location = Column(String(255), nullable=True)
    measurement_text = Column(String(255), nullable=True)
    heatmap_path = Column(String(500), nullable=True)
    status = Column(String(50), default="FLAGGED") # FLAGGED, NORMAL, UNCERTAIN
    
    analysis = relationship("AIAnalysis", back_populates="findings")

class Segmentation(Base):
    __tablename__ = "segmentations"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("ai_analyses.id"), nullable=False)
    structure_name = Column(String(100), nullable=False) # Tumor Core, Whole Tumor, Lung Nodule
    volume_cm3 = Column(Float, nullable=True)
    dimensions_mm = Column(JSON, default=dict) # {"x": 24, "y": 18, "z": 15}
    mask_path = Column(String(500), nullable=False)
    color_hex = Column(String(20), default="#FF0000")
    
    analysis = relationship("AIAnalysis", back_populates="segmentations")

class Report(Base):
    __tablename__ = "reports"
    
    id = Column(Integer, primary_key=True, index=True)
    study_id = Column(Integer, ForeignKey("studies.id"), nullable=False)
    radiologist_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    status = Column(String(50), default="DRAFT") # DRAFT, PENDING_REVIEW, APPROVED
    clinical_history = Column(Text, nullable=True)
    technique = Column(Text, nullable=True)
    findings = Column(Text, nullable=False)
    ai_summary = Column(Text, nullable=True)
    impression = Column(Text, nullable=False)
    disclaimer = Column(Text, default="MedVision AI is a research-oriented clinical decision-support prototype. AI-generated findings and reports are not a substitute for professional medical judgment. All results require review by a qualified healthcare professional.")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    study = relationship("Study", back_populates="reports")

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    study_id = Column(Integer, ForeignKey("studies.id"), nullable=True)
    title = Column(String(255), default="MIRA Consultation")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    sender = Column(String(50), nullable=False) # user | mira | system
    content = Column(Text, nullable=False)
    citations = Column(JSON, default=list)
    confidence_info = Column(JSON, default=dict)
    audio_path = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    session = relationship("ChatSession", back_populates="messages")

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False) # STUDY_VIEW, INFERENCE_RUN, REPORT_APPROVE, CHAT_QUERY
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(String(100), nullable=True)
    details = Column(JSON, default=dict)
    ip_address = Column(String(50), nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
