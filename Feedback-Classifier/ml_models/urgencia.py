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

print("[ML_Urgency] Cargando especialista de urgencia...")

# ✅ CONFIGURACIÓN DESDE SETTINGS
FORCE_CPU = settings.FORCE_CPU
MAX_LENGTH = settings.MAX_TEXT_LENGTH
USE_HUGGINGFACE = settings.USE_HUGGINGFACE_API

# ✅ MODELO HUGGING FACE PARA URGENCIA (usando sentiment como proxy)
HF_URGENCY_MODEL = settings.HF_SENTIMENT_MODEL

# ✅ KEYWORDS ORIGINALES (MEJORADAS)
KEYWORDS_URGENCIA_ALTA = [
    "roto", "no funciona", "desastre", "caído", "error fatal", "urgente", 
    "inmediato", "no puedo trabajar", "estafa", "fraude", "no acceso",
    "crítico", "emergencia", "bloqueado", "perdí", "robaron", "hackear",
    "urgent", "critical", "emergency", "broken", "crashed", "down", 
    "failed", "disaster", "help", "asap", "immediately"
]

KEYWORDS_URGENCIA_MEDIA = [
    "lento", "tarda mucho", "problema", "ayuda", "duda", "consulta", 
    "pregunta", "mejorar", "sugerencia", "demora", "retraso", "confuso",
    "difícil", "complicado", "issue", "slow", "delay", "question", 
    "help", "support", "confused", "problem", "bug", "error"
]

# ✅ VARIABLE GLOBAL DEL CLASIFICADOR
URGENCY_CLASSIFIER = None

def load_urgency_model():
    """Carga el modelo de urgencia usando Hugging Face"""
    global URGENCY_CLASSIFIER
    
    try:
        if USE_HUGGINGFACE:
            logger.info(f"[ML_Urgency] Cargando modelo HF: {HF_URGENCY_MODEL}")
            
            # ✅ CONFIGURAR DISPOSITIVO
            device = -1 if FORCE_CPU or not torch.cuda.is_available() else 0
            device_name = "CPU" if device == -1 else "GPU"
            
            logger.info(f"[ML_Urgency] Usando dispositivo: {device_name}")
            
            # ✅ CARGAR PIPELINE
            URGENCY_CLASSIFIER = pipeline(
                task="sentiment-analysis",
                model=HF_URGENCY_MODEL,
                device=device,
                model_kwargs={
                    "low_cpu_mem_usage": True,
                    "torch_dtype": torch.float16 if device != -1 else torch.float32
                },
                return_all_scores=True,
                truncation=True,
                max_length=MAX_LENGTH
            )
            
                        # ✅ PRUEBA
            test_result = URGENCY_CLASSIFIER("test")
            logger.info("[ML_Urgency] ✅ Modelo Hugging Face cargado correctamente")
            
        else:
            logger.info("[ML_Urgency] Usando clasificación por keywords (HuggingFace deshabilitado)")
            URGENCY_CLASSIFIER = None
            
    except Exception as e:
        logger.error(f"[ML_Urgency] ❌ Error cargando modelo HF: {e}")
        logger.info("[ML_Urgency] 🔄 Fallback a clasificación por keywords")
        URGENCY_CLASSIFIER = None


def classify_urgency(text: str) -> str:
    """Clasifica urgencia usando modelo ML o keywords como fallback"""
    global URGENCY_CLASSIFIER
    
    if not text or not isinstance(text, str):
        return "baja"
    
    text_clean = text.lower().strip()
    
    # ✅ INTENTAR CON HUGGING FACE PRIMERO
    if URGENCY_CLASSIFIER is not None:
        try:
            # Truncar texto si es muy largo
            if len(text_clean) > MAX_LENGTH:
                text_clean = text_clean[:MAX_LENGTH]
            
            # Obtener predicción
            result = URGENCY_CLASSIFIER(text_clean)
            
            # El modelo de sentiment devuelve POSITIVE/NEGATIVE
            # Mapear a urgencia: NEGATIVE (sentimiento negativo) = ALTA urgencia
            if isinstance(result, list) and len(result) > 0:
                scores = result[0] if isinstance(result[0], list) else result
                
                # Buscar el score más alto
                max_score = 0
                predicted_label = "POSITIVE"
                
                for item in scores:
                    if item['score'] > max_score:
                        max_score = item['score']
                        predicted_label = item['label']
                
                # Mapeo: NEGATIVE sentiment = ALTA urgencia
                if predicted_label == "NEGATIVE" and max_score > 0.7:
                    logger.info(f"[ML_Urgency] HF: {predicted_label} ({max_score:.3f}) -> ALTA")
                    return "alta"
                elif predicted_label == "NEGATIVE" and max_score > 0.5:
                    logger.info(f"[ML_Urgency] HF: {predicted_label} ({max_score:.3f}) -> MEDIA")
                    return "media"
                else:
                    logger.info(f"[ML_Urgency] HF: {predicted_label} ({max_score:.3f}) -> BAJA")
                    # Continuar con keywords como backup
                    
        except Exception as e:
            logger.error(f"[ML_Urgency] Error en HF, usando keywords: {e}")
    
    # ✅ FALLBACK CON KEYWORDS (mejorado)
    text_words = text_clean.split()
    
    # Contar matches de keywords
    alta_matches = sum(1 for keyword in KEYWORDS_URGENCIA_ALTA 
                      if keyword.lower() in text_clean)
    media_matches = sum(1 for keyword in KEYWORDS_URGENCIA_MEDIA 
                       if keyword.lower() in text_clean)
    
    # Lógica de decisión mejorada
    if alta_matches >= 2:
        logger.info(f"[ML_Urgency] Keywords: ALTA ({alta_matches} matches)")
        return "alta"
    elif alta_matches >= 1:
        logger.info(f"[ML_Urgency] Keywords: ALTA (1 match crítico)")
        return "alta"
    elif media_matches >= 2:
        logger.info(f"[ML_Urgency] Keywords: MEDIA ({media_matches} matches)")
        return "media"
    elif media_matches >= 1:
        logger.info(f"[ML_Urgency] Keywords: MEDIA (1 match)")
        return "media"
    else:
        logger.info("[ML_Urgency] Keywords: BAJA (sin matches)")
        return "baja"


def get_urgencia(texto: str) -> str:
    """Función principal para obtener urgencia de un texto"""
    if not texto:
        return "baja"
    
    try:
        result = classify_urgency(texto)
        logger.info(f"[ML_Urgency] Resultado final para '{texto[:50]}...': {result}")
        return result
    except Exception as e:
        logger.error(f"[ML_Urgency] Error en get_urgencia: {e}")
        return "baja"


def reload_urgency_model():
    """Recarga el modelo de urgencia (útil para debugging)"""
    global URGENCY_CLASSIFIER
    URGENCY_CLASSIFIER = None
    load_urgency_model()
    logger.info("[ML_Urgency] Modelo recargado")


def get_model_info():
    """Retorna información sobre el estado actual del modelo"""
    global URGENCY_CLASSIFIER
    
    info = {
        "huggingface_enabled": USE_HUGGINGFACE,
        "model_loaded": URGENCY_CLASSIFIER is not None,
        "model_name": HF_URGENCY_MODEL if USE_HUGGINGFACE else "Keywords only",
        "device": "CPU" if FORCE_CPU else "Auto",
        "max_length": MAX_LENGTH,
        "keywords_alta": len(KEYWORDS_URGENCIA_ALTA),
        "keywords_media": len(KEYWORDS_URGENCIA_MEDIA)
    }
    
    return info


def test_urgency_classifier():
    """Función para probar el clasificador con casos de ejemplo"""
    test_cases = [
        ("Mi aplicación está rota y no puedo trabajar", "alta"),
        ("Tengo una pregunta sobre cómo usar la función", "media"), 
        ("Todo funciona perfectamente, solo quería comentar", "baja"),
        ("URGENTE: El sistema se cayó completamente", "alta"),
        ("La página carga un poco lento", "media"),
        ("Critical error - cannot access my account", "alta"),
        ("How do I change my password?", "media"),
        ("Thanks for the great service!", "baja"),
        ("El sitio está caído desde hace horas, perdí datos importantes", "alta"),
        ("¿Podrían mejorar la interfaz?", "baja")
    ]
    
    logger.info("[ML_Urgency] 🧪 Iniciando pruebas del clasificador...")
    
    correct_predictions = 0
    total_tests = len(test_cases)
    
    for i, (text, expected) in enumerate(test_cases, 1):
        try:
            predicted = get_urgencia(text)
            is_correct = predicted == expected
            status = "✅" if is_correct else "❌"
            
            if is_correct:
                correct_predictions += 1
                
            logger.info(f"[ML_Urgency] Caso {i}/{total_tests} {status}")
            logger.info(f"  Texto: '{text[:50]}...'")
            logger.info(f"  Esperado: {expected} | Obtenido: {predicted}")
            
        except Exception as e:
            logger.error(f"[ML_Urgency] Error en caso {i}: {e}")
    
    accuracy = (correct_predictions / total_tests) * 100
    logger.info(f"[ML_Urgency] 📊 Precisión: {correct_predictions}/{total_tests} ({accuracy:.1f}%)")
    
    return {
        "total_tests": total_tests,
        "correct_predictions": correct_predictions,
        "accuracy": accuracy,
        "model_info": get_model_info()
    }


# ✅ INICIALIZACIÓN AL IMPORTAR EL MÓDULO
try:
    load_urgency_model()
except Exception as e:
    logger.error(f"[ML_Urgency] Error en inicialización: {e}")
    URGENCY_CLASSIFIER = None


# ✅ FUNCIÓN MAIN PARA TESTING DIRECTO
if __name__ == "__main__":
    """Ejecutar cuando se corre directamente el archivo"""
    print("=" * 60)
    print("🚀 TESTING CLASIFICADOR DE URGENCIA")
    print("=" * 60)
    
    # Mostrar información del modelo
    model_info = get_model_info()
    print(f"📋 Configuración:")
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    print()
    
    # Ejecutar pruebas
    results = test_urgency_classifier()
    
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE PRUEBAS")
    print("=" * 60)
    print(f"Total de casos: {results['total_tests']}")
    print(f"Predicciones correctas: {results['correct_predictions']}")
    print(f"Precisión: {results['accuracy']:.1f}%")
    
    # Probar casos específicos interactivos
    print("\n" + "=" * 60)
    print("🔍 MODO INTERACTIVO")
    print("=" * 60)
    print("Ingresa texto para clasificar (escribe 'exit' para salir):")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'salir', 'q']:
                print("¡Hasta luego! 👋")
                break
            
            if not user_input:
                print("❌ Por favor ingresa un texto válido")
                continue
            
            print(f"🔍 Analizando: '{user_input}'")
            urgencia = get_urgencia(user_input)
            
            # Mostrar resultado con colores/emojis
            emoji_map = {
                "alta": "🔴 ",
                "media": "🟡 MEDIA", 
                "baja": "🟢 BAJA"
            }
            
            print(f"📊 Urgencia detectada: {emoji_map.get(urgencia, urgencia.upper())}")
            
        except KeyboardInterrupt:
            print("\n\n❌ Interrumpido por el usuario. ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


# ✅ EXPORTACIÓN DE LA FUNCIÓN PRINCIPAL
# Esta es la función que será importada por otros módulos
__all__ = ['get_urgencia', 'classify_urgency', 'test_urgency_classifier', 'get_model_info', 'reload_urgency_model']

print("[ML_Urgency] ✅ Módulo de urgencia cargado correctamente")
