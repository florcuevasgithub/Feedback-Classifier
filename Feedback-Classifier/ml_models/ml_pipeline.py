import time
import json
import os
import requests
from typing import Dict, Any, Optional

# ✅ CONFIGURACIÓN HUGGING FACE API
USE_HUGGINGFACE_API = os.getenv("USE_HUGGINGFACE_API", "true").lower() == "true"
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
USE_LITE_MODE = os.getenv("USE_LITE_MODE", "false").lower() == "true"

# URLs de modelos en Hugging Face
HF_SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
HF_CATEGORY_MODEL = "facebook/bart-large-mnli"

print(f"[ML_Pipeline] 🤖 Using Hugging Face API: {USE_HUGGINGFACE_API}")
print(f"[ML_Pipeline] 🔑 API Key configured: {'Yes' if HUGGINGFACE_API_KEY else 'No'}")
print(f"[ML_Pipeline] 💡 Lite Mode: {USE_LITE_MODE}")

def call_huggingface_api(model_name: str, text: str, task_params: Dict = None) -> Optional[Dict[Any, Any]]:
    """Llama a la API de Hugging Face con reintentos y manejo de errores"""
    if not HUGGINGFACE_API_KEY:
        print("[HF_API] ⚠️ No API key configured")
        return None
    
    url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {"inputs": text}
    if task_params:
        payload.update(task_params)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"[HF_API] Calling {model_name} (attempt {attempt + 1})")
            response = requests.post(url, json=payload, headers=headers, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                print(f"[HF_API] ✅ Success")
                return result
            elif response.status_code == 503:
                print(f"[HF_API] ⏳ Model loading, waiting 5s...")
                time.sleep(5)
                continue
            elif response.status_code == 429:
                print(f"[HF_API] ⏳ Rate limited, waiting 10s...")
                time.sleep(10)
                continue
            else:
                print(f"[HF_API] ❌ HTTP {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"[HF_API] ❌ Request error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(3)
    
    return None

def classify_sentiment(text: str) -> Dict[str, Any]:
    """Clasificar sentimiento usando Hugging Face API o fallback"""
    if not text or not isinstance(text, str):
        return {"label": "NEU", "score": 0.5, "source": "default"}
    
    text = text.strip()[:512]
    
    # Intentar Hugging Face API primero
    if USE_HUGGINGFACE_API and HUGGINGFACE_API_KEY and not USE_LITE_MODE:
        try:
            result = call_huggingface_api(HF_SENTIMENT_MODEL, text)
            
            if result and isinstance(result, list) and len(result) > 0:
                hf_result = result[0]
                hf_label = hf_result.get('label', '').lower()
                score = hf_result.get('score', 0.5)
                
                label_mapping = {
                    'negative': 'NEG',
                    'neutral': 'NEU',
                    'positive': 'POS',
                    'label_0': 'NEG',
                    'label_1': 'NEU', 
                    'label_2': 'POS'
                }
                
                clean_label = label_mapping.get(hf_label, 'NEU')
                print(f"[HF_API] Sentiment: {hf_label} → {clean_label} (score: {score:.3f})")
                
                return {
                    "label": clean_label,
                    "score": round(score, 4),
                    "source": "huggingface_api"
                }
            
        except Exception as e:
            print(f"[HF_API] Error in sentiment: {e}")
    
    return classify_sentiment_lite(text)

def classify_sentiment_lite(text: str) -> Dict[str, Any]:
    """Clasificador de sentimiento usando palabras clave mejoradas"""
    if not text:
        return {"label": "NEU", "score": 0.5, "source": "lite"}
    
    text_lower = text.lower()
    
    positive_words = [
        'excelente', 'bueno', 'perfecto', 'genial', 'increíble', 'fantástico', 'maravilloso',
        'feliz', 'contento', 'satisfecho', 'encantado', 'alegre', 'emocionado',
        'great', 'excellent', 'amazing', 'awesome', 'perfect', 'wonderful', 'fantastic',
        'love', 'happy', 'satisfied', 'pleased', 'delighted', 'impressed', 'outstanding'
    ]
    
    negative_words = [
        'malo', 'horrible', 'terrible', 'pésimo', 'disgusto', 'odio', 'furioso',
        'enojado', 'molesto', 'triste', 'decepcionado', 'frustrado', 'disgustado',
        'bad', 'awful', 'hate', 'terrible', 'worst', 'disgusting', 'angry',
        'disappointed', 'frustrated', 'annoyed', 'upset', 'sad', 'horrible'
    ]
    
    pos_matches = sum(1 for word in positive_words if word in text_lower)
    neg_matches = sum(1 for word in negative_words if word in text_lower)
    
    if pos_matches > neg_matches:
        confidence = min(0.9, 0.6 + (pos_matches * 0.1))
        return {"label": "POS", "score": confidence, "source": "lite"}
    elif neg_matches > pos_matches:
        confidence = min(0.9, 0.6 + (neg_matches * 0.1))
        return {"label": "NEG", "score": confidence, "source": "lite"}
    else:
        return {"label": "NEU", "score": 0.6, "source": "lite"}

def classify_category(text: str) -> Dict[str, Any]:
    """Clasificar categoría usando zero-shot o fallback"""
    if not text or not isinstance(text, str):
        return {"label": "General/Otro", "score": 0.5, "source": "default"}
    
    text = text.strip()[:512]
    
    # Intentar Hugging Face zero-shot classification
    if USE_HUGGINGFACE_API and HUGGINGFACE_API_KEY and not USE_LITE_MODE:
        try:
            candidate_labels = [
                "Soporte Técnico",
                "Facturación y Pagos", 
                "Producto/Sugerencias",
                "Logística/Envíos",
                "General/Otro"
            ]
            
            task_params = {
                "parameters": {
                    "candidate_labels": candidate_labels
                }
            }
            
            result = call_huggingface_api(HF_CATEGORY_MODEL, text, task_params)
            
            if result and 'labels' in result and 'scores' in result:
                best_label = result['labels'][0]
                best_score = result['scores'][0]
                
                print(f"[HF_API] Category: {best_label} (score: {best_score:.3f})")
                
                return {
                    "label": best_label,
                    "score": round(best_score, 4),
                    "source": "huggingface_api"
                }
            
        except Exception as e:
            print(f"[HF_API] Error in category: {e}")
    
    return classify_category_lite(text)

def classify_category_lite(text: str) -> Dict[str, Any]:
    """Clasificador de categoría usando palabras clave avanzadas"""
    if not text:
        return {"label": "General/Otro", "score": 0.5, "source": "lite"}
    
    text_lower = text.lower()
    
    categories = {
        "Soporte Técnico": {
            "keywords": [
                'login', 'contraseña', 'password', 'error', 'problema', 'bug', 'falla', 
                'no funciona', 'técnico', 'sistema', 'app', 'aplicación', 'website',
                'crash', 'loading', 'access', 'account', 'authentication', 'código'
            ],
            "weight": 0.9
        },
        "Logística/Envíos": {
            "keywords": [
                'envío', 'entrega', 'delivery', 'llegó', 'paquete', 'enviar', 
                'repartidor', 'courier', 'shipping', 'transport', 'arrived',
                'package', 'box', 'delay', 'tracking', 'address', 'dirección'
            ],
            "weight": 0.85
        },
        "Facturación y Pagos": {
            "keywords": [
                'factura', 'pago', 'cobro', 'dinero', 'precio', 'costo', 
                'billing', 'payment', 'tarjeta', 'cuenta', 'charge',
                'invoice', 'refund', 'money', 'card', 'bank', 'descuento'
            ],
            "weight": 0.85
        },
        "Producto/Sugerencias": {
            "keywords": [
                'producto', 'calidad', 'sugerencia', 'mejora', 'feature', 
                'funcionalidad', 'diseño', 'usabilidad', 'recommendation',
                'suggestion', 'improve', 'quality', 'design', 'experiencia'
            ],
            "weight": 0.8
        }
    }
    
    # Buscar la mejor categoría
    best_category = "General/Otro"
    best_score = 0.6
    max_matches = 0
    
    for category, data in categories.items():
        matches = sum(1 for keyword in data["keywords"] if keyword in text_lower)
        if matches > max_matches:
            max_matches = matches
            best_category = category
            best_score = min(0.95, data["weight"] + (matches * 0.05))
    
    return {
        "label": best_category,
        "score": best_score,
        "source": "lite"
    }

def classify_urgency(text: str) -> str:
    """Clasificar urgencia usando palabras clave"""
    if not text or not isinstance(text, str):
        return "Normal"
    
    text_lower = text.lower()
    
    # Palabras de alta urgencia expandidas
    urgent_keywords = [
        'urgente', 'emergency', 'inmediato', 'ya', 'ahora', 'grave',
        'crítico', 'importante', 'asap', 'rápido', 'pronto', 'emergencia',
        'immediately', 'urgent', 'critical', 'serious', 'help', 'ayuda'
    ]
    
    urgent_matches = sum(1 for keyword in urgent_keywords if keyword in text_lower)
    
    if urgent_matches >= 2:
        return "Crítica"
    elif urgent_matches >= 1:
        return "Alta"
    else:
        return "Normal"

def analyze_feedback(text: str) -> Dict[str, Any]:
    """Pipeline completo de análisis de feedback"""
    if not text or not isinstance(text, str):
        return {
            "text_input": "",
            "analysis": {
                "sentiment": {"label": "NEU", "score": 0.5, "source": "default"},
                "category": {"label": "General/Otro", "score": 0.5, "source": "default"},
                "urgency": {"label": "Normal"}
            },
            "processing_time": 0.0,
            "error": "Invalid input"
        }
    
    print(f"[ML_Pipeline] Analyzing feedback: '{text[:50]}...'")
    start_time = time.time()
    
    try:
        # Ejecutar clasificaciones
        sentiment_result = classify_sentiment(text)
        category_result = classify_category(text)
        urgency_result = classify_urgency(text)
        
        processing_time = time.time() - start_time
        
        result = {
            "text_input": text,
            "analysis": {
                "sentiment": {
                    "label": sentiment_result.get("label"),
                    "score": sentiment_result.get("score"),
                    "source": sentiment_result.get("source", "unknown")
                },
                "category": {
                    "label": category_result.get("label"),
                    "score": category_result.get("score"),
                    "source": category_result.get("source", "unknown")
                },
                "urgency": {
                    "label": urgency_result if isinstance(urgency_result, str) else urgency_result.get("label", "Normal")
                }
            },
            "processing_time": round(processing_time, 3),
            "api_mode": "huggingface" if USE_HUGGINGFACE_API and HUGGINGFACE_API_KEY else "lite"
        }
        
        print(f"[ML_Pipeline] ✅ Analysis completed in {processing_time:.3f}s")
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"[ML_Pipeline] ❌ Error in analysis: {e}")
        
        return {
            "text_input": text,
            "analysis": {
                "sentiment": {"label": "ERROR", "score": 0.0, "source": "error"},
                "category": {"label": "ERROR", "score": 0.0, "source": "error"},
                "urgency": {"label": "Normal"}
            },
            "processing_time": round(processing_time, 3),
            "error": str(e)
        }

def get_ml_status() -> Dict[str, Any]:
    """Verifica el estado del sistema ML"""
    return {
        "huggingface_api": {
            "enabled": USE_HUGGINGFACE_API,
            "api_key_configured": bool(HUGGINGFACE_API_KEY),
            "models": {
                "sentiment": HF_SENTIMENT_MODEL,
                "category": HF_CATEGORY_MODEL
            }
        },
        "lite_mode": USE_LITE_MODE,
        "fallback_active": not (USE_HUGGINGFACE_API and HUGGINGFACE_API_KEY),
        "status": "ready",
        "timestamp": time.time()
    }

# Para testing rápido
if __name__ == "__main__":
    print("\n🧪 TESTING ML_PIPELINE\n")
    
    # Mostrar estado
    status = get_ml_status()
    print(f"Status: {json.dumps(status, indent=2)}\n")
    
    # Pruebas
    test_cases = [
        "El producto está excelente, muy buena calidad!",
        "No puedo hacer login, me sale error 500",
        "El delivery nunca llegó a mi casa",
        "Quiero un reembolso de mi pago de $100",
        "URGENTE: necesito ayuda inmediata con mi cuenta"
    ]
    
    # ✅ AQUÍ CONTINÚA EL CÓDIGO:
    
    print("=" * 60)
    print("🧪 EJECUTANDO PRUEBAS DE CLASIFICACIÓN")
    print("=" * 60)
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n📝 Prueba {i}/5:")
        print(f"Texto: '{test_text}'")
        print("-" * 50)
        
        # Analizar el feedback
        result = analyze_feedback(test_text)
        
        # Mostrar resultados
        analysis = result["analysis"]
        print(f"🎭 Sentimiento: {analysis['sentiment']['label']} ({analysis['sentiment']['score']:.3f})")
        print(f"📁 Categoría: {analysis['category']['label']} ({analysis['category']['score']:.3f})")
        print(f"⚡ Urgencia: {analysis['urgency']['label']}")
        print(f"⏱️ Tiempo: {result['processing_time']}s")
        print(f"🔧 Modo: {result.get('api_mode', 'unknown')}")
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
    
    print("\n" + "=" * 60)
    print("✅ PRUEBAS COMPLETADAS")
    print("=" * 60)
    
    # Estadísticas finales
    print(f"\n📊 Configuración actual:")
    print(f"   - Hugging Face API: {'✅ Activa' if USE_HUGGINGFACE_API else '❌ Desactivada'}")
    print(f"   - Modo LITE: {'✅ Activo' if USE_LITE_MODE else '❌ Desactivado'}")
    print(f"   - Fallback: {'✅ Activo' if not (USE_HUGGINGFACE_API and HUGGINGFACE_API_KEY) else '❌ Desactivado'}")
    
    print(f"\n🎯 Recomendación para Render Free:")
    print(f"   - Usar variables: USE_HUGGINGFACE_API=false, USE_LITE_MODE=true")
    print(f"   - Memoria estimada: ~50-80 MB")
    print(f"   - Velocidad: ~0.001-0.005s por análisis")

    print("\n🚀 ¡Listo para producción!)
