from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import warnings
import logging
import sys
import os

# Agregar el directorio padre al path para imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config.settings import settings

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suprimir warnings de transformers
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# ✅ CONFIGURACIÓN DESDE SETTINGS
MODEL_NAME = settings.HF_SENTIMENT_MODEL
FORCE_CPU = settings.FORCE_CPU
MAX_LENGTH = settings.MAX_TEXT_LENGTH

# ✅ MAPEAMOS LAS ETIQUETAS
LABEL_MAP = {
    "negative": "NEG",
    "neutral": "NEU", 
    "positive": "POS",
    "label_0": "NEG",
    "label_1": "NEU",
    "label_2": "POS"
}

# ✅ VARIABLE GLOBAL DEL CLASIFICADOR
SENTIMENT_CLASSIFIER = None

def load_sentiment_model():
    """Carga el modelo de sentiment con configuración robusta"""
    global SENTIMENT_CLASSIFIER
    
    try:
        print(f"[ML_Sentiment] Cargando modelo: {MODEL_NAME}")
        
        # ✅ CONFIGURACIÓN DE DISPOSITIVO
        if FORCE_CPU or not torch.cuda.is_available():
            device = -1
            device_name = "CPU"
        else:
            device = 0
            device_name = "GPU"
        
        print(f"[ML_Sentiment] Usando dispositivo: {device_name}")
        
        # ✅ CONFIGURACIÓN OPTIMIZADA
        model_kwargs = {
            "low_cpu_mem_usage": True,
            "use_cache": False
        }
        
        if settings.ENABLE_QUANTIZATION and device == -1:
            model_kwargs["torch_dtype"] = torch.float16
        
        # ✅ CARGAR CON PIPELINE OPTIMIZADO
        SENTIMENT_CLASSIFIER = pipeline(
            task="sentiment-analysis",
            model=MODEL_NAME,
            device=device,
            model_kwargs=model_kwargs,
            return_all_scores=False,
            truncation=True,
            max_length=MAX_LENGTH
        )
        
        # ✅ PRUEBA DE FUNCIONAMIENTO
        test_result = SENTIMENT_CLASSIFIER("test")
        logger.info(f"[ML_Sentiment] ✅ Modelo cargado exitosamente en {device_name}")
        logger.info(f"[ML_Sentiment] Prueba: {test_result}")
        return True
        
    except Exception as e:
        logger.error(f"[ML_Sentiment] ❌ Error cargando modelo: {e}")
        
        # ✅ FALLBACK A CLASIFICADOR MOCK
        logger.warning("[ML_Sentiment] ⚠️ Usando clasificador mock")
        SENTIMENT_CLASSIFIER = None
        return False

def classify_sentiment(text: str) -> dict:
    """
    Clasifica el sentimiento de un texto (inglés o español).

    Args:
        text (str): El feedback del cliente.

    Returns:
        dict: Un diccionario con 'label', 'score' y 'source'.
    """
    # ✅ VALIDACIÓN DE ENTRADA
    if not text or not isinstance(text, str):
        return {"label": "NEU", "score": 0.5, "source": "default"}
    
    # ✅ LIMPIEZA Y TRUNCADO
    text = text.strip()
    if len(text) == 0:
        return {"label": "NEU", "score": 0.5, "source": "default"}
    
    if len(text) > MAX_LENGTH:
        text = text[:MAX_LENGTH]
    
    # ✅ USAR MODELO HUGGING FACE SI ESTÁ DISPONIBLE
    if SENTIMENT_CLASSIFIER is not None:
        try:
            result_raw = SENTIMENT_CLASSIFIER(text)
            
            # ✅ MANEJAR DIFERENTES FORMATOS DE RESPUESTA
            if isinstance(result_raw, list) and len(result_raw) > 0:
                result = result_raw[0]
            else:
                result = result_raw

            # ✅ MAPEAR ETIQUETAS
            original_label = result.get('label', 'neutral').lower()
            clean_label = LABEL_MAP.get(original_label, "NEU")

            return {
                "label": clean_label,
                "score": round(result.get('score', 0.5), 4),
                "source": "huggingface_local"
            }

        except Exception as e:
            logger.error(f"[ML_Sentiment] Error al clasificar: {e}")
            return mock_sentiment_classification(text)
    
    # ✅ FALLBACK A MOCK
    return mock_sentiment_classification(text)

def mock_sentiment_classification(text: str) -> dict:
    """Clasificador básico usando palabras clave mejoradas"""
    text_lower = text.lower()
    
    # ✅ PALABRAS POSITIVAS EXPANDIDAS (español + inglés)
    positive_words = [
        # Español
        'excelente', 'bueno', 'genial', 'perfecto', 'increíble', 'fantástico', 
        'maravilloso', 'feliz', 'contento', 'satisfecho', 'encantado', 'amor', 
        'amar', 'gustar', 'me gusta', 'recomiendo', 'rápido', 'eficiente', 
        'útil', 'fácil', 'cómodo', 'hermoso', 'mejor', 'éxito', 'ganador',
        # Inglés
        'excellent', 'good', 'great', 'perfect', 'amazing', 'fantastic', 
        'wonderful', 'happy', 'satisfied', 'love', 'like', 'recommend', 
        'fast', 'efficient', 'useful', 'easy', 'comfortable', 'beautiful', 
        'best', 'success', 'winner', 'awesome', 'brilliant', 'outstanding'
    ]
    
    # ✅ PALABRAS NEGATIVAS EXPANDIDAS (español + inglés)
    negative_words = [
        # Español
        'malo', 'terrible', 'horrible', 'pésimo', 'odio', 'odiar', 'detesto', 
        'molesto', 'enojado', 'frustrado', 'problema', 'error', 'falla', 
        'lento', 'difícil', 'complicado', 'caro', 'costoso', 'peor', 
        'decepcionante', 'inútil', 'basura', 'no funciona', 'roto',
        # Inglés
        'bad', 'terrible', 'horrible', 'awful', 'hate', 'angry', 'frustrated', 
        'problem', 'error', 'bug', 'slow', 'difficult', 'expensive', 'worst', 
        'disappointing', 'useless', 'trash', 'broken', 'failed', 'sucks',
        'annoying', 'confusing', 'complicated'
    ]
    
    # ✅ CONTADORES
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    # ✅ LÓGICA DE CLASIFICACIÓN
    if positive_count > negative_count:
        confidence = min(0.6 + (positive_count * 0.1), 0.95)
        return {"label": "POS", "score": confidence, "source": "mock_keywords"}
    elif negative_count > positive_count:
        confidence = min(0.6 + (negative_count * 0.1), 0.95)
        return {"label": "NEG", "score": confidence, "source": "mock_keywords"}
    else:
        return {"label": "NEU", "score": 0.5, "source": "mock_default"}

# ✅ FUNCIÓN AUXILIAR PARA INICIALIZACIÓN
def initialize_sentiment_model():
    """Inicializa el modelo de sentimiento al importar el módulo"""
    try:
        success = load_sentiment_model()
        if success:
            logger.info("[ML_Sentiment] ✅ Modelo inicializado correctamente")
        else:
            logger.warning("[ML_Sentiment] ⚠️ Usando modo fallback")
    except Exception as e:
        logger.error(f"[ML_Sentiment] ❌ Error en inicialización: {e}")

# ✅ AUTO-INICIALIZACIÓN DEL MODELO
if __name__ != "__main__":
    initialize_sentiment_model()
