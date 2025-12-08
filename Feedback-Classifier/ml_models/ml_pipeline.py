import time
import json
import os
import sys

# ✅ CAMBIO: Variable específica para modelos cuantizados
USE_QUANTIZED_MODELS = os.getenv("USE_QUANTIZED_MODELS", "true").lower() == "true"
RENDER_FREE_MODE = os.getenv("RENDER_FREE_MODE", "false").lower() == "true"
DISABLE_ML_MODELS = os.getenv("DISABLE_ML_MODELS", "false").lower() == "true"
ML_FALLBACK_MODE = os.getenv("ML_FALLBACK_MODE", "false").lower() == "true"

print(f"[ML_Pipeline] 🚀 Render Free Mode: {RENDER_FREE_MODE}")
print(f"[ML_Pipeline] 🔧 Use Quantized Models: {USE_QUANTIZED_MODELS}")
print(f"[ML_Pipeline] 🚫 ML Models Disabled: {DISABLE_ML_MODELS}")

# ✅ CAMBIO: Nueva lógica para usar modelos cuantizados CON IMPORTS RELATIVOS
if USE_QUANTIZED_MODELS:
    print("[ML_Pipeline] Cargando especialistas CUANTIZADOS...")
    try:
        from .sentimiento import classify_sentiment  # ✅ IMPORT RELATIVO CORREGIDO
        from .urgencia import classify_urgency      # ✅ IMPORT RELATIVO CORREGIDO
        from .categoria import classify_category    # ✅ IMPORT RELATIVO CORREGIDO
        print("[ML_Pipeline] ✅ Especialistas CUANTIZADOS cargados correctamente")
        
    except ImportError as e:
        print(f"[ML_Pipeline] ❌ Error cargando modelos cuantizados: {e}")
        print("[ML_Pipeline] 🔄 Fallback a modo LITE...")
        
        # Funciones LITE como fallback
        def classify_sentiment(text: str) -> dict:
            """Clasificador de sentimiento LITE usando palabras clave"""
            if not text: return {"label": "NEU", "score": 0.5}
            text_lower = text.lower()
            
            pos_words = ['excelente', 'bueno', 'perfecto', 'amor', 'genial', 'fantastic', 'great', 'excellent', 'amazing', 'love', 'wonderful']
            neg_words = ['malo', 'horrible', 'terrible', 'odio', 'pésimo', 'awful', 'bad', 'hate', 'worst', 'disgusting']
            
            pos_count = sum(1 for word in pos_words if word in text_lower)
            neg_count = sum(1 for word in neg_words if word in text_lower)
            
            if pos_count > neg_count:
                return {"label": "POS", "score": 0.8}
            elif neg_count > pos_count:
                return {"label": "NEG", "score": 0.8}
            else:
                return {"label": "NEU", "score": 0.6}

        def classify_category(text: str) -> dict:
            """Clasificador de categoría LITE usando palabras clave"""
            if not text: return {"label": "General/Otro", "score": 0.5}
            text_lower = text.lower()
            
            # Soporte Técnico
            if any(word in text_lower for word in ['login', 'contraseña', 'error', 'problema', 'no funciona', 'falla', 'bug']):
                return {"label": "Soporte Técnico", "score": 0.8}
            # Logística/Envíos
            elif any(word in text_lower for word in ['envío', 'entrega', 'delivery', 'llegó', 'repartidor', 'shipping']):
                return {"label": "Logística/Envíos", "score": 0.8}
            # Facturación y Pagos
            elif any(word in text_lower for word in ['factura', 'pago', 'cobro', 'dinero', 'precio', 'billing', 'payment']):
                return {"label": "Facturación y Pagos", "score": 0.8}
            # Producto/Sugerencias
            elif any(word in text_lower for word in ['producto', 'calidad', 'sugerencia', 'mejora', 'feature', 'quality']):
                return {"label": "Producto/Sugerencias", "score": 0.8}
            else:
                return {"label": "General/Otro", "score": 0.6}

        def classify_urgency(text: str) -> str:
            """Clasificador de urgencia LITE usando palabras clave"""
            if not text: return "Normal"
            text_lower = text.lower()
            
            urgent_words = ['urgente', 'emergency', 'inmediato', 'ya', 'ahora', 'grave', 'critical', 'asap']
            return "Alta" if any(word in text_lower for word in urgent_words) else "Normal"
        
        print("[ML_Pipeline] ✅ Fallback LITE activado")

else:
    # Modo LITE explícito
    print("[ML_Pipeline] Cargando especialistas LITE (sin transformers)...")
    
    # Funciones ligeras integradas
    def classify_sentiment(text: str) -> dict:
        """Clasificador de sentimiento LITE usando palabras clave"""
        if not text: return {"label": "NEU", "score": 0.5}
        text_lower = text.lower()
        
        pos_words = ['excelente', 'bueno', 'perfecto', 'amor', 'genial', 'fantastic', 'great', 'excellent', 'amazing', 'love', 'wonderful']
        neg_words = ['malo', 'horrible', 'terrible', 'odio', 'pésimo', 'awful', 'bad', 'hate', 'worst', 'disgusting']
        
        pos_count = sum(1 for word in pos_words if word in text_lower)
        neg_count = sum(1 for word in neg_words if word in text_lower)
        
        if pos_count > neg_count:
            return {"label": "POS", "score": 0.8}
        elif neg_count > pos_count:
            return {"label": "NEG", "score": 0.8}
        else:
            return {"label": "NEU", "score": 0.6}

    def classify_category(text: str) -> dict:
        """Clasificador de categoría LITE usando palabras clave"""
        if not text: return {"label": "General/Otro", "score": 0.5}
        text_lower = text.lower()
        
        # Soporte Técnico
        if any(word in text_lower for word in ['login', 'contraseña', 'error', 'problema', 'no funciona', 'falla', 'bug']):
            return {"label": "Soporte Técnico", "score": 0.8}
        # Logística/Envíos
        elif any(word in text_lower for word in ['envío', 'entrega', 'delivery', 'llegó', 'repartidor', 'shipping']):
            return {"label": "Logística/Envíos", "score": 0.8}
        # Facturación y Pagos
        elif any(word in text_lower for word in ['factura', 'pago', 'cobro', 'dinero', 'precio', 'billing', 'payment']):
            return {"label": "Facturación y Pagos", "score": 0.8}
        # Producto/Sugerencias
        elif any(word in text_lower for word in ['producto', 'calidad', 'sugerencia', 'mejora', 'feature', 'quality']):
            return {"label": "Producto/Sugerencias", "score": 0.8}
        else:
            return {"label": "General/Otro", "score": 0.6}

    def classify_urgency(text: str) -> str:
        """Clasificador de urgencia LITE usando palabras clave"""
        if not text: return "Normal"
        text_lower = text.lower()
        
        urgent_words = ['urgente', 'emergency', 'inmediato', 'ya', 'ahora', 'grave', 'critical', 'asap']
        return "Alta" if any(word in text_lower for word in urgent_words) else "Normal"
    
    print("[ML_Pipeline] ✅ Especialistas LITE cargados correctamente")

def analyze_feedback(text: str) -> dict:
    """Ejecuta el pipeline de ML completo sobre un texto."""
    print(f"[ML_Pipeline] Analizando: '{text[:50]}...'")
    start_time = time.time()
    
    try:
        sentiment_result = classify_sentiment(text)
        urgency_result = classify_urgency(text)
        category_result = classify_category(text)

        final_analysis = {
            "text_input": text,
            "analysis": {
                "sentiment": {
                    "label": sentiment_result.get("label"),
                    "score": sentiment_result.get("score")
                },
                "urgency": {
                    "label": urgency_result if isinstance(urgency_result, str) else urgency_result.get("label", "Normal")
                },
                "category": {
                    "label": category_result.get("label"),
                    "score": category_result.get("score")
                }
            },
            "mode": "quantized" if USE_QUANTIZED_MODELS else "lite"
        }
        
        end_time = time.time()
        print(f"[ML_Pipeline] ✅ Análisis completado en {end_time - start_time:.2f}s")
        return final_analysis
        
    except Exception as e:
        print(f"[ML_Pipeline] ERROR en análisis: {e}")
        return {
            "text_input": text,
            "analysis": {
                "sentiment": {"label": "ERROR", "score": 0.0},
                "urgency": {"label": "Normal"},
                "category": {"label": "ERROR", "score": 0.0}
            },
            "error": str(e)
        }

def get_ml_status() -> dict:
    """Verifica el estado de todos los modelos ML"""
    return {
        "mode": "quantized" if USE_QUANTIZED_MODELS else "lite",
        "use_quantized_models": USE_QUANTIZED_MODELS,
        "render_free_mode": RENDER_FREE_MODE,
        "transformers_disabled": DISABLE_ML_MODELS,
        "status": "ready"
    }

# Para testing rápido
if __name__ == "__main__":
    test_text = "El producto está excelente, muy buena calidad"
    result = analyze_feedback(test_text)
    print(f"Resultado: {json.dumps(result, indent=2, ensure_ascii=False)}")
