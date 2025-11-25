from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import os
import torch

# --- [PASO 1] Cargar Especialista de Categoría ---
print("[ML_Category] Cargando especialista de categoría...")

# 1. MAPEAMOS LAS ETIQUETAS (DEBE COINCIDIR CON EL ENTRENAMIENTO)
LABEL_MAP_CATEGORY = {
    0: "Soporte Técnico",
    1: "Facturación y Pagos",
    2: "Producto/Sugerencias",
    3: "Logística/Envíos",
    4: "General/Otro"
}

# 2. RUTA RELATIVA AL MODELO
# Busca la carpeta "arriba" de este archivo, en 'model_files/category_model'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_CATEGORY = os.path.join(BASE_DIR, "..", "model_files", "category_model")

# 3. CARGAMOS EL MODELO (EL QUE ENTRENAMOS EN COLAB)
try:
    print(f"[ML_Category] Cargando componentes desde: {MODEL_PATH_CATEGORY}")
    # Cargamos manualmente para asegurarnos de que encuentre los archivos locales
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH_CATEGORY)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_CATEGORY)
    
    CATEGORY_CLASSIFIER = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer
    )
    print("[ML_Category] ¡Modelo de categoría cargado exitosamente!")
except Exception as e:
    print(f"[ML_Category] ERROR cargando modelo: {e}")
    print("[ML_Category] Asegúrate de que los archivos del modelo (safetensors, config.json...) están en la carpeta 'model_files/category_model/'")
    CATEGORY_CLASSIFIER = None

# 4. LA FUNCIÓN CLASIFICADORA (entregable)
def classify_category(text: str) -> dict:
    """Clasifica la categoría usando nuestro modelo entrenado."""
    if CATEGORY_CLASSIFIER is None:
        return {"label": "ERROR_MODEL_NOT_LOADED", "score": 0.0}
    if not text or not isinstance(text, str):
        return {"label": "ERROR_INVALID_INPUT", "score": 0.0}
        
    try:
        # 1. Obtenemos el resultado crudo del modelo
        result_raw = CATEGORY_CLASSIFIER(text)[0]
        
        # 2. "Traducimos" la etiqueta (ej: "LABEL_1" -> 1)
        label_id_str = result_raw['label']
        label_id_int = int(label_id_str.split('_')[-1]) 
        
        # 3. Mapeamos el ID a nuestro nombre 
        clean_label = LABEL_MAP_CATEGORY.get(label_id_int, "UNKNOWN") 
        
        # 4. Devolvemos el diccionario limpio
        return {"label": clean_label, "score": result_raw['score']}
    except Exception as e:
        print(f"[ML_Category] Error al clasificar: {e}")
        return {"label": "ERROR_CLASSIFICATION", "score": 0.0}

# 5. PRUEBA RÁPIDA (Para probar el archivo)
if __name__ == "__main__":
    # Esto solo se ejecuta si corremos: python ml_models/categoria.py
    print("\n--- PRUEBA INDIVIDUAL DE CATEGORÍA ---")
    
    test_1 = "No me funciona la clave para entrar"
    test_2 = "El repartidor nunca llegó a mi casa"
    
    print(f"Texto: '{test_1}' -> Resultado: {classify_category(test_1)}")
    print(f"Texto: '{test_2}' -> Resultado: {classify_category(test_2)}")