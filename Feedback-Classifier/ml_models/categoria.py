from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import os
import torch
import requests
import zipfile
from pathlib import Path
import warnings
import logging
import sys

# Agregar el directorio padre al path para imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config.settings import settings

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suprimir warnings de transformers
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

print("[ML_Category] Cargando especialista de categoría...")

# ✅ CONFIGURACIÓN DESDE SETTINGS
USE_HUGGINGFACE_API = settings.USE_HUGGINGFACE_API
HF_CATEGORY_MODEL = settings.HF_CATEGORY_MODEL
BACKBLAZE_URL = settings.BACKBLAZE_MODEL_URL
FORCE_CPU = settings.FORCE_CPU
MAX_LENGTH = settings.MAX_TEXT_LENGTH

# ✅ MAPEO DE CATEGORÍAS
LABEL_MAP_CATEGORY = {
    0: "Soporte Técnico",
    1: "Facturación y Pagos", 
    2: "Producto/Sugerencias",
    3: "Logística/Envíos",
    4: "General/Otro"
}

# ✅ CATEGORÍAS PARA ZERO-SHOT (HUGGING FACE)
CATEGORY_LABELS = [
    "soporte técnico y problemas tecnológicos",
    "facturación, pagos y dinero", 
    "sugerencias de producto y mejoras",
    "logística, envíos y entregas",
    "consulta general y otros temas"
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_CATEGORY = os.path.join(BASE_DIR, "..", "model_files", "category_model")

# ✅ VARIABLES GLOBALES
CATEGORY_CLASSIFIER = None
CATEGORY_CLASSIFIER_HF = None

def download_model_if_needed():
    """Descarga automáticamente desde Backblaze B2 (OPTIMIZADO)"""
    config_path = os.path.join(MODEL_PATH_CATEGORY, "config.json")
    if os.path.exists(config_path):
        print("[ML_Category] ✅ Modelo local ya existe - usando cache")
        return True
    
    print("[ML_Category] 📦 Descargando modelo desde Backblaze B2...")
    
    try:
        Path(MODEL_PATH_CATEGORY).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"[ML_Category] 📥 Descargando desde Backblaze...")
        response = requests.get(BACKBLAZE_URL, stream=True, timeout=300)
        response.raise_for_status()
        
        zip_path = os.path.join(BASE_DIR, "..", "temp_model.zip")
        
        # Descarga con progreso
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r[ML_Category] Progreso: {progress:.1f}%", end='')
        
        print()  # Nueva línea
        
        # Verificar descarga
        file_size = os.path.getsize(zip_path)
        print(f"[ML_Category] ✅ Descargado: {file_size/(1024*1024):.1f} MB")
        
        if not zipfile.is_zipfile(zip_path):
            print("[ML_Category] ❌ ERROR: No es ZIP válido")
            os.remove(zip_path)
            return False
        
        # Extraer
        print("[ML_Category] 📂 Extrayendo...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(BASE_DIR, "..", "model_files"))
        
        os.remove(zip_path)  # Limpiar
        print("[ML_Category] ✅ Modelo listo desde Backblaze B2")
        return True
        
    except Exception as e:
        print(f"[ML_Category] ❌ Error descarga: {e}")
        return False

def load_category_models():
    """Carga ambos modelos: local (Backblaze) y Hugging Face"""
    global CATEGORY_CLASSIFIER, CATEGORY_CLASSIFIER_HF
    
    # ✅ 1. INTENTAR CARGAR MODELO LOCAL (BACKBLAZE)
    try:
        download_success = download_model_if_needed()
        
        if download_success:
            print(f"[ML_Category] Cargando modelo local desde: {MODEL_PATH_CATEGORY}")
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH_CATEGORY,
                torch_dtype=torch.int8 if FORCE_CPU else torch.float16,
                device_map="cpu" if FORCE_CPU else "auto",
                low_cpu_mem_usage=True
            )
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_CATEGORY)

            CATEGORY_CLASSIFIER = pipeline(
                task="text-classification",
                model=model,
                tokenizer=tokenizer,
                device=-1 if FORCE_CPU else 0
            )
            logger.info("[ML_Category] ✅ Modelo local cargado desde Backblaze B2")
        else:
            logger.warning("[ML_Category] ⚠️ No se pudo cargar modelo local")
            
    except Exception as e:
        logger.error(f"[ML_Category] ❌ Error cargando modelo local: {e}")
        CATEGORY_CLASSIFIER = None

    # ✅ 2. CARGAR MODELO HUGGING FACE (ZERO-SHOT)
    try:
        print(f"[ML_Category] Cargando modelo HF: {HF_CATEGORY_MODEL}")
        
        CATEGORY_CLASSIFIER_HF = pipeline(
            task="zero-shot-classification",
            model=HF_CATEGORY_MODEL,
            device=-1 if FORCE_CPU else 0,
            model_kwargs={
                "low_cpu_mem_usage": True,
                "torch_dtype": torch.float16 if not FORCE_CPU else torch.float32
            }
        )
        
        # Prueba
        test_result = CATEGORY_CLASSIFIER_HF("test", CATEGORY_LABELS)
        logger.info("[ML_Category] ✅ Modelo Hugging Face cargado correctamente")
        
    except Exception as e:
        logger.error(f"[ML_Category] ❌ Error cargando modelo HF: {e}")
        CATEGORY_CLASSIFIER_HF = None

def classify_category(text: str) -> dict:
    """
    Clasifica la categoría del feedback usando modelo disponible
    
    Args:
        text (str): El feedback del cliente
        
    Returns:
        dict: Diccionario con 'label', 'score' y 'source'
    """
    # ✅ VALIDACIÓN DE ENTRADA
    if not text or not isinstance(text, str):
        return {"label": "General/Otro", "score": 0.5, "source": "default"}
    
    text = text.strip()
    if len(text) == 0:
        return {"label": "General/Otro", "score": 0.5, "source": "default"}
    
    if len(text) > MAX_LENGTH:
        text = text[:MAX_LENGTH]
    
    # ✅ 1. INTENTAR CON MODELO LOCAL (PRIORIDAD)
    if CATEGORY_CLASSIFIER is not None:
        try:
            result_raw = CATEGORY_CLASSIFIER(text)[0]
            label_id_str = result_raw['label']
            label_id_int = int(label_id_str.split('_')[-1]) 
            clean_label = LABEL_MAP_CATEGORY.get(label_id_int, "General/Otro")
            
            return {
                "label": clean_label,
                "score": round(result_raw['score'], 4),
                "source": "backblaze_local"
            }
            
        except Exception as e:
            logger.error(f"[ML_Category] Error modelo local: {e}")
    
    # ✅ 2. FALLBACK A HUGGING FACE (ZERO-SHOT)
    if CATEGORY_CLASSIFIER_HF is not None:
        try:
            result = CATEGORY_CLASSIFIER_HF(text, CATEGORY_LABELS)
            
            # Mapear resultado a nuestras categorías
            best_label = result['labels'][0]
            best_score = result['scores'][0]
            
            # Convertir a nuestro formato
            category_mapping = {
                "soporte técnico y problemas tecnológicos": "Soporte Técnico",
                "facturación, pagos y dinero": "Facturación y Pagos",
                "sugerencias de producto y mejoras": "Producto/Sugerencias", 
                "logística, envíos y entregas": "Logística/Envíos",
                "consulta general y otros temas": "General/Otro"
            }
            
            final_label = category_mapping.get(best_label, "General/Otro")
            
            return {
                "label": final_label,
                "score": round(best_score, 4),
                "source": "huggingface_zeroshot"
            }
            
        except Exception as e:
            logger.error(f"[ML_Category] Error modelo HF: {e}")
    
    # ✅ 3. FALLBACK FINAL (MOCK)
    return mock_category_classification(text)

def mock_category_classification(text: str) -> dict:
    """Clasificador básico usando palabras clave"""
    text_lower = text.lower()
    
    # ✅ PALABRAS CLAVE POR CATEGORÍA
    keywords = {
        "Soporte Técnico": [
            'no funciona', 'error', 'problema', 'bug', 'falla', 'roto', 'clave', 
            'password', 'login', 'acceso', 'cuenta', 'configurar', 'instalar',
            'conectar', 'wifi', 'internet', 'aplicación', 'app', 'sistema',
            'technical', 'support', 'broken', 'fix', 'crash', 'freeze'
        ],
        
        "Facturación y Pagos": [
            'pago', 'factura', 'cobro', 'tarjeta', 'dinero', 'precio', 'costo',
            'descuento', 'reembolso', 'devolución', 'billetear', 'transferencia',
            'paypal', 'visa', 'mastercard', 'billing', 'payment', 'invoice',
            'charge', 'refund', 'money', 'cost', 'subscription', 'plan'
        ],
        
        "Producto/Sugerencias": [
            'mejora', 'sugerencia', 'idea', 'propuesta', 'feature', 'función',
            'nuevo', 'agregar', 'añadir', 'cambiar', 'modificar', 'actualizar',
            'versión', 'diseño', 'interfaz', 'usabilidad', 'suggestion',
            'improvement', 'enhance', 'add', 'update', 'upgrade', 'feedback'
        ],
        
        "Logística/Envíos": [
            'envío', 'entrega', 'paquete', 'correo', 'shipping', 'delivery',
            'llega', 'llegó', 'retraso', 'demorado', 'rápido', 'lento',
            'tracking', 'seguimiento', 'dirección', 'address', 'pedido',
            'order', 'courier', 'dhl', 'fedex', 'ups', 'postal'
        ]
    }
    
    # ✅ CONTEO DE COINCIDENCIAS POR CATEGORÍA
    scores = {}
    for category, words in keywords.items():
        score = sum(1 for word in words if word in text_lower)
        scores[category] = score
    
    # ✅ DETERMINAR LA MEJOR CATEGORÍA
    if all(score == 0 for score in scores.values()):
        return {"label": "General/Otro", "score": 0.5, "source": "mock_default"}
    
    best_category = max(scores, key=scores.get)
    max_score = scores[best_category]
    confidence = min(0.6 + (max_score * 0.1), 0.95)
    
    return {
        "label": best_category,
        "score": confidence,
        "source": "mock_keywords"
    }

# ✅ FUNCIÓN DE INICIALIZACIÓN
def initialize_category_models():
    """Inicializa ambos modelos de categorización"""
    try:
        load_category_models()
        logger.info("[ML_Category] ✅ Modelos de categoría inicializados")
    except Exception as e:
        logger.error(f"[ML_Category] ❌ Error en inicialización: {e}")

# ✅ AUTO-INICIALIZACIÓN
if __name__ != "__main__":
    initialize_category_models()
