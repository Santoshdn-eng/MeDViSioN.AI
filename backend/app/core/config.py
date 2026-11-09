import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "MEDVISION AI"
    VERSION: str = "1.0.0-research"
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "medvision-super-secret-key-change-in-production-2026")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./medvision.db")
    
    # LLM & Persona settings
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "mock")  # openai | anthropic | gemini | ollama | mock
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    
    # Voice / TTS
    TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "browser")  # browser | elevenlabs | azure | mock
    
    # Research / Demo Mode
    DEMO_MODE: bool = True
    CONFIDENCE_THRESHOLD: float = 0.65
    
    # Storage
    STORAGE_DIR: str = os.getenv("STORAGE_DIR", "./storage/dicom")
    
    class Config:
        case_sensitive = True

settings = Settings()
