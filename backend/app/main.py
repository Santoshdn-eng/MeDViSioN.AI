from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from backend.app.core.config import settings
from backend.app.database.session import init_db
from backend.app.api.auth import router as auth_router
from backend.app.api.router import api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB schemas on startup
    await init_db()
    os.makedirs("./backend/app/static/masks", exist_ok=True)
    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Multimodal AI Radiology Assistant & Clinical Decision-Support Platform powered by MIRA",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="./backend/app/static"), name="static")

# Include Routers
app.include_router(auth_router, prefix=settings.API_V1_STR)
app.include_router(api_router, prefix=settings.API_V1_STR)

from fastapi.responses import FileResponse

@app.get("/")
async def root():
    return FileResponse("./backend/app/static/index.html")

@app.get("/api/info")
async def system_info():
    return {
        "system": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "mode": "RESEARCH / CLINICAL DECISION-SUPPORT PROTOTYPE",
        "assistant": "MIRA (Medical Imaging & Radiology Assistant)",
        "disclaimer": "AI-generated findings and reports are not a substitute for professional medical judgment. All results require review by a qualified healthcare professional.",
        "docs_url": "/docs"
    }

@app.get("/health")
async def health_check():
    return {"status": "HEALTHY", "gpu_acceleration": "ENABLED (CUDA/CPU-Fallback)", "research_mode": True}
