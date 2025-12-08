# Feedback-Classifier/app/main.py
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import logging
from app.routes.feedback_api import router as feedback_router  
from app.routes.healthcheck import router as health_router 
from app.config.db import engine, Base

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("🗃️ Base de datos inicializada correctamente")
    except Exception as e:
        logger.error(f"❌ Error inicializando BD: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔴 Aplicación detenida")

app = FastAPI(
    title="Agente IA de Análisis de Feedback",
    version="1.0.0",
    description="Backend para clasificar, almacenar y exportar feedback de usuarios.",
    lifespan=lifespan
)

# Registrar routers
app.include_router(feedback_router, prefix="/api")
app.include_router(health_router, prefix="/api")

@app.get("/", tags=["Status"])
def read_root():
    return {"message": "Bienvenido al Agente IA de Análisis de Feedback. Visita /docs para ver los endpoints."}

@app.get("/health", tags=["Status"])
def health_check():
    return {"status": "ok", "service": "Feedback-Classifier API"}

# ✅ NUEVO: Endpoint para verificar estado de modelos ML
@app.get("/api/ml-status", tags=["Status"])
def ml_status():
    from ml_models.ml_pipeline import get_ml_status
    return get_ml_status()
