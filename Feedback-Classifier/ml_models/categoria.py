from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import os
import torch
import requests
import zipfile
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

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

# 2. RUTA RELATIVA AL MODELO Y AUTO-DESCARGA
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_CATEGORY = os.path.join(BASE_DIR, "..", "model_files", "category_model")

def download_model_if_needed():
    """Descarga automáticamente el modelo desde Dropbox si no existe"""
    config_path = os.path.join(MODEL_PATH_CATEGORY, "config.json")
    if os.path.exists(config_path):
        return True  # Ya existe
    
    print("[ML_Category] Modelo no encontrado. Descargando desde Dropbox...")
    
    MODEL_URL = os.getenv("MODEL_DOWNLOAD_URL")
    if not MODEL_URL:
        print("[ML_Category] ❌ ERROR: Falta variable MODEL_DOWNLOAD_URL en .env")
        return False
    
    try:
        # Crear carpetas
        Path(MODEL_PATH_CATEGORY).mkdir(parents=True, exist_ok=True)
        
        print("[ML_Category] 📥 Descargando modelo desde Dropbox...")
        
        # Dropbox con dl=1 es descarga directa, sin complicaciones
        response = requests.get(MODEL_URL, stream=True, allow_redirects=True)
        response.raise_for_status()
        
        zip_path = os.path.join(BASE_DIR, "..", "temp_model.zip")
        
        # Guardar archivo
        total_size = 0
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
        
        # Verificar tamaño
        file_size = os.path.getsize(zip_path)
        print(f"[ML_Category] Archivo descargado: {file_size / (1024*1024):.2f} MB")
        
        if file_size < 1024:  # Menos de 1KB = error
            print("[ML_Category] ERROR: Archivo muy pequeño - probablemente HTML")
            with open(zip_path, 'r', errors='ignore') as f:
                print(f"Contenido: {f.read(200)}")
            os.remove(zip_path)
            return False
        
        # Verificar que es un ZIP válido
        if not zipfile.is_zipfile(zip_path):
            print("[ML_Category] ❌ ERROR: El archivo no es un ZIP válido")
            os.remove(zip_path)
            return False
        
        # Extraer archivos
        print("[ML_Category] Extrayendo modelo...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(BASE_DIR, "..", "model_files"))
        
        # Limpiar archivo temporal
        os.remove(zip_path)
        print("[ML_Category] Modelo descargado y extraído exitosamente!")
        return True
        
    except Exception as e:
        print(f"[ML_Category] Error descargando modelo: {e}")
        return False

# 3. DESCARGAR MODELO SI ES NECESARIO
download_success = download_model_if_needed()

# 4. CARGAMOS EL MODELO (EL QUE ENTRENAMOS EN COLAB)
try:
    if download_success:
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
    else:
        raise Exception("No se pudo descargar el modelo")
except Exception as e:
    print(f"[ML_Category] ERROR cargando modelo: {e}")
    print("[ML_Category] Verifica que el link de Dropbox sea correcto y público")
    CATEGORY_CLASSIFIER = None

# 5. LA FUNCIÓN CLASIFICADORA (entregable)
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

# 6. PRUEBA RÁPIDA (Para probar el archivo)
if __name__ == "__main__":
    # Esto solo se ejecuta si corremos: python ml_models/categoria.py
    print("\n--- PRUEBA INDIVIDUAL DE CATEGORÍA ---")
    
    test_1 = "No me funciona la clave para entrar"
    test_2 = "El repartidor nunca llegó a mi casa"
    
    print(f"Texto: '{test_1}' -> Resultado: {classify_category(test_1)}")
    print(f"Texto: '{test_2}' -> Resultado: {classify_category(test_2)}")