from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
import logging

from fastapi.middleware.cors import CORSMiddleware  

from app.routes.feedback_api import router as feedback_router  
from app.routes.healthcheck import router as health_router 
from app.config.db import engine, Base


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("🗃️ Base de datos inicializada correctamente")
    except Exception as e:
        logger.error(f"❌ Error inicializando BD: {e}")
        raise
    
    yield
    
   
    logger.info("🔴 Aplicación detenida")

app = FastAPI(
    title="Agente IA de Análisis de Feedback",
    version="1.0.0",
    description="Backend para clasificar, almacenar y exportar feedback de usuarios.",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def fix_user_agent(request: Request, call_next):
    headers = dict(request.headers)

    blocked_agents = ["swagger", "curl", "python-requests", "postmanruntime"]

    ua = headers.get("user-agent", "").lower()
    if any(bad in ua for bad in blocked_agents):
        headers["user-agent"] = "Mozilla/5.0"  


    request.scope["headers"] = [
        (k.encode(), v.encode()) for k, v in headers.items()
    ]

    return await call_next(request)


app.include_router(feedback_router, prefix="/api")
app.include_router(health_router, prefix="/api")

@app.get("/", tags=["Status"])
def read_root():
    return {"message": "Bienvenido al Agente IA de Análisis de Feedback. Visita /docs para ver los endpoints."}

@app.get("/health", tags=["Status"])
def health_check():
    return {"status": "ok", "service": "Feedback-Classifier API"}


@app.get("/api/ml-status", tags=["Status"])
def ml_status():
    from ml_models.ml_pipeline import get_ml_status
    return get_ml_status()
