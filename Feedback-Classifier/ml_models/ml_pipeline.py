import time
import json
import os
import sys

# --- [PASO 0] CONFIGURACIÓN DE RUTAS ---
# Le dice a Python que mire en esta misma carpeta
# para encontrar 'sentimiento.py', 'urgencia.py', etc.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# --- [PASO 1] Importar a los Especialistas ---
# Python ejecutará los otros 3 archivos .py ahora
# y cargará los modelos en memoria.
print("[ML_Pipeline] Cargando especialistas...")
try:
    from sentimiento import classify_sentiment
    from urgencia import classify_urgency
    from categoria import classify_category
    print("[ML_Pipeline] ¡Todos los especialistas están listos!")
except ImportError as e:
    print(f"[ML_Pipeline] ERROR: No se pudieron importar los módulos.")
    print(f"Detalle: {e}")
    sys.exit(1)

# --- [PASO 2] Definir el Pipeline Principal ---
def analyze_feedback(text: str) -> dict:
    """
    Ejecuta el pipeline de ML completo sobre un texto.
    """
    print(f"\n[ML_Pipeline] Analizando texto: '{text[:50]}...'")
    start_time = time.time()
    
    # Llamamos a cada especialista importado
    sentiment_result = classify_sentiment(text)
    urgency_result = classify_urgency(text)
    category_result = classify_category(text)

    # Unimos todo
    final_analysis = {
        "text_input": text,
        "analysis": {
            "sentiment": {
                "label": sentiment_result.get("label"),
                "score": sentiment_result.get("score")
            },
            "urgency": {
                "label": urgency_result
            },
            "category": {
                "label": category_result.get("label"),
                "score": category_result.get("score")
            }
        }
    }
    
    end_time = time.time()
    print(f"[ML_Pipeline] Análisis completado en {end_time - start_time:.2f} segundos.")
    return final_analysis

# --- [PASO 3] PRUEBA DEL PIPELINE COMPLETO ---
if __name__ == "__main__":
    # Esto se ejecuta si corremos: python ml_models/ml_pipeline.py
    print("\n\n--- PRUEBA DEL PIPELINE COMPLETO (Local) ---")
    
    test_1 = "No me funciona la clave para entrar, es un desastre!"
    test_2 = "El repartidor nunca llegó a mi casa, ¿cuándo lo recibiré?"
    test_3 = "Me cobraron de más en la boleta, necesito ayuda."
    test_4 = "Me encanta el nuevo color azul, gran producto."

    print("\n--- Test 1 ---")
    analysis_1 = analyze_feedback(test_1)
    print(json.dumps(analysis_1, indent=2, ensure_ascii=False))

    print("\n--- Test 2 ---")
    analysis_2 = analyze_feedback(test_2)
    print(json.dumps(analysis_2, indent=2, ensure_ascii=False))
    
    print("\n--- Test 3 ---")
    analysis_3 = analyze_feedback(test_3)
    print(json.dumps(analysis_3, indent=2, ensure_ascii=False))

    print("\n--- Test 4 ---")
    analysis_4 = analyze_feedback(test_4)
    print(json.dumps(analysis_4, indent=2, ensure_ascii=False))
# ✅  Función para verificar estado de modelos ML
def get_ml_status() -> dict:
    """Verifica el estado de todos los modelos ML"""
    from sentimiento import SENTIMENT_CLASSIFIER
    from categoria import CATEGORY_CLASSIFIER  
    from urgencia import classify_urgency
    
    return {
        "sentiment_model": {
            "status": "loaded" if SENTIMENT_CLASSIFIER is not None else "error",
            "model_name": "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        },
        "category_model": {
            "status": "loaded" if CATEGORY_CLASSIFIER is not None else "error", 
            "model_path": "local_category_model"
        },
        "urgency_model": {
            "status": "loaded",  # Siempre funciona (reglas)
            "type": "rule_based"
        },
        "pipeline_status": "ready" if all([
            SENTIMENT_CLASSIFIER is not None,
            CATEGORY_CLASSIFIER is not None
        ]) else "partial"
    }