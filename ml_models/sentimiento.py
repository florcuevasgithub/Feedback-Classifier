from transformers import pipeline
import torch

# 1. EL MODELO POLÍGLOTA (XLM-RoBERTa)
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# 2. MAPEAMOS LAS ETIQUETAS
LABEL_MAP = {
    "negative": "NEG",
    "neutral": "NEU",
    "positive": "POS"
}

# 3. CARGAMOS EL MODELO UNA SOLA VEZ
print(f"[ML_Sentiment] Cargando modelo políglota '{MODEL_NAME}'...")
try:
    SENTIMENT_CLASSIFIER = pipeline(
        task="sentiment-analysis",
        model=MODEL_NAME
    )
    print("[ML_Sentiment] Modelo Políglota cargado exitosamente.")
except Exception as e:
    print(f"[ML_Sentiment] ERROR cargando modelo: {e}")
    SENTIMENT_CLASSIFIER = None

# 4. LA FUNCIÓN CLASIFICADORA (entregable)
def classify_sentiment(text: str) -> dict:
    """
    Clasifica el sentimiento de un texto (inglés o español).

    Args:
        text (str): El feedback del cliente.

    Returns:
        dict: Un diccionario con 'label' y 'score'.
              Etiquetas posibles: 'POS', 'NEG', 'NEU'.
    """
    if SENTIMENT_CLASSIFIER is None:
        return {"label": "ERROR_MODEL_NOT_LOADED", "score": 0.0}

    if not text or not isinstance(text, str):
        return {"label": "ERROR_INVALID_INPUT", "score": 0.0}

    try:
        # 1. Obtenemos el resultado crudo del modelo
        result_raw = SENTIMENT_CLASSIFIER(text)[0]

        # 2. "Traducimos" la etiqueta
        original_label = result_raw['label']
        clean_label = LABEL_MAP.get(original_label, "UNKNOWN")

        # 3. Devolvemos el diccionario limpio
        return {
            "label": clean_label,
            "score": result_raw['score']
        }

    except Exception as e:
        print(f"[ML_Sentiment] Error al clasificar texto: {e}")
        return {"label": "ERROR_CLASSIFICATION", "score": 0.0}

# 5. PRUEBA RÁPIDA
if __name__ == "__main__":
    print("\n--- PRUEBA DE SENTIMIENTO (POLÍGLOTA) ---")

    test_es = "Esto es maravilloso."
    test_en = "This is wonderful."

    print(f"Texto: '{test_es}'")
    print(f"Resultado: {classify_sentiment(test_es)}")

    print(f"\nTexto: '{test_en}'")
    print(f"Resultado: {classify_sentiment(test_en)}")