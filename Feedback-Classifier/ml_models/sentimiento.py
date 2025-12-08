from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import warnings
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suprimir warnings de transformers
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# 1. EL MODELO POLÍGLOTA (XLM-RoBERTa)
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# 2. MAPEAMOS LAS ETIQUETAS
LABEL_MAP = {
    "negative": "NEG",
    "neutral": "NEU", 
    "positive": "POS"
}

# 3. CARGAMOS EL MODELO UNA SOLA VEZ (VERSION MEJORADA)
print(f"[ML_Sentiment] Cargando modelo políglota '{MODEL_NAME}'...")
SENTIMENT_CLASSIFIER = None

def load_sentiment_model():
    """Carga el modelo de sentiment con manejo robusto de errores"""
    global SENTIMENT_CLASSIFIER
    
    try:
        # ✅ Método 1: Cargar con pipeline (más simple)
        print("[ML_Sentiment] Intentando cargar con pipeline...")
        
        # Detectar dispositivo disponible
        device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if device >= 0 else "CPU"
        
        print(f"[ML_Sentiment] Usando dispositivo: {device_name}")
        
        SENTIMENT_CLASSIFIER = pipeline(
        task="sentiment-analysis",
        model=MODEL_NAME,
        device=-1,                     # ✅ Forzar CPU
        torch_dtype=torch.float16,     # ✅ SOLO ESTA LÍNEA
        model_kwargs={                 # ✅ SOLO ESTE BLOQUE
        "low_cpu_mem_usage": True,
        "use_cache": False
},
    return_all_scores=False,
    truncation=True,
    max_length=512
)
        
        # Prueba rápida para verificar que funciona
        test_result = SENTIMENT_CLASSIFIER("test")
        print(f"[ML_Sentiment] ✅ Modelo cargado exitosamente en {device_name}")
        print(f"[ML_Sentiment] Prueba: {test_result}")
        return True
        
    except Exception as e1:
        print(f"[ML_Sentiment] ❌ Error con pipeline: {e1}")
        
        try:
            # ✅ Método 2: Cargar manualmente (más control)
            print("[ML_Sentiment] Intentando carga manual...")
            
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            
            # Mover a dispositivo apropiado
            if torch.cuda.is_available():
                model = model.to('cuda')
                device = 0
            else:
                device = -1
            
            SENTIMENT_CLASSIFIER = pipeline(
                task="sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=device,
                return_all_scores=False
            )
            
            print("[ML_Sentiment] ✅ Modelo cargado manualmente")
            return True
            
        except Exception as e2:
            print(f"[ML_Sentiment] ❌ Error carga manual: {e2}")
            
            try:
                # ✅ Método 3: Fallback simple (solo CPU, sin optimizaciones)
                print("[ML_Sentiment] Intentando fallback básico...")
                
                SENTIMENT_CLASSIFIER = pipeline(
                    task="sentiment-analysis",
                    model=MODEL_NAME,
                    device=-1  # Forzar CPU
                )
                
                print("[ML_Sentiment] ✅ Modelo cargado en fallback (CPU)")
                return True
                
            except Exception as e3:
                print(f"[ML_Sentiment] ❌ Error total: {e3}")
                print("[ML_Sentiment] ⚠️  Usando modelo mock para desarrollo")
                SENTIMENT_CLASSIFIER = None
                return False

# Cargar el modelo al importar
load_success = load_sentiment_model()

# 4. LA FUNCIÓN CLASIFICADORA (VERSION MEJORADA)
def classify_sentiment(text: str) -> dict:
    """
    Clasifica el sentimiento de un texto (inglés o español).

    Args:
        text (str): El feedback del cliente.

    Returns:
        dict: Un diccionario con 'label' y 'score'.
              Etiquetas posibles: 'POS', 'NEG', 'NEU'.
    """
    # Validación de entrada
    if not text or not isinstance(text, str):
        return {"label": "ERROR_INVALID_INPUT", "score": 0.0}
    
    # Limpiar texto
    text = text.strip()
    if len(text) == 0:
        return {"label": "ERROR_EMPTY_TEXT", "score": 0.0}
    
    # Si el modelo no está cargado, usar mock
    if SENTIMENT_CLASSIFIER is None:
        print("[ML_Sentiment] ⚠️ Usando clasificación mock (modelo no disponible)")
        return mock_sentiment_classification(text)

    try:
        # Limitar longitud del texto
        if len(text) > 512:
            text = text[:512]
            
        # 1. Obtenemos el resultado del modelo
        result_raw = SENTIMENT_CLASSIFIER(text)
        
        # Manejar diferentes formatos de respuesta
        if isinstance(result_raw, list) and len(result_raw) > 0:
            result = result_raw[0]
        else:
            result = result_raw

        # 2. "Traducimos" la etiqueta
        original_label = result.get('label', 'unknown').lower()
        clean_label = LABEL_MAP.get(original_label, "NEU")  # Default a neutral

        # 3. Devolvemos el diccionario limpio
        return {
            "label": clean_label,
            "score": round(result.get('score', 0.5), 4)
        }

    except Exception as e:
        print(f"[ML_Sentiment] Error al clasificar: {e}")
        return {"label": "ERROR_CLASSIFICATION", "score": 0.0}

# 5. CLASIFICADOR MOCK (para desarrollo/fallback)
def mock_sentiment_classification(text: str) -> dict:
    """Clasificador básico usando palabras clave para desarrollo"""
    text_lower = text.lower()
    
    # Palabras positivas
    positive_words = ['excelente', 'bueno', 'genial', 'perfecto', 'amazing', 'great', 'good', 'love', 'excellent']
    # Palabras negativas  
    negative_words = ['malo', 'terrible', 'horrible', 'odio', 'bad', 'awful', 'hate', 'terrible', 'worst']
    
    positive_score = sum(1 for word in positive_words if word in text_lower)
    negative_score = sum(1 for word in negative_words if word in text_lower)
    
    if positive_score > negative_score:
        return {"label": "POS", "score": 0.7}
    elif negative_score > positive_score:
        return {"label": "NEG", "score": 0.7}
    else:
        return {"label": "NEU", "score": 0.6}

# 6. FUNCIÓN DE ESTADO
def get_sentiment_status() -> dict:
    """Retorna el estado del modelo de sentiment"""
    return {
        "model_name": MODEL_NAME,
        "status": "loaded" if SENTIMENT_CLASSIFIER is not None else "error",
        "device": "cuda" if torch.cuda.is_available() and SENTIMENT_CLASSIFIER is not None else "cpu",
        "fallback_active": SENTIMENT_CLASSIFIER is None
    }

# 7. PRUEBA RÁPIDA
if __name__ == "__main__":
    print("\n--- PRUEBA DE SENTIMIENTO (POLÍGLOTA) ---")
    
    # Mostrar estado
    status = get_sentiment_status()
    print(f"Estado del modelo: {status}")
    
    # Pruebas
    tests = [
        "Esto es maravilloso y excelente!",
        "This is wonderful and amazing!",
        "Esto es horrible y terrible",
        "This is awful and bad",
        "Texto neutral sin emociones específicas"
    ]
    
    for test_text in tests:
        result = classify_sentiment(test_text)
        print(f"Texto: '{test_text}'")
        print(f"Resultado: {result}\n")
