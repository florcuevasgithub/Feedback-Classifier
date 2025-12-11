import time
import json
import os
import requests
from typing import Dict, Any, Optional, Tuple

# ============================================================
# ✅ CONFIGURACIÓN HUGGING FACE API
# ============================================================

USE_HUGGINGFACE_API = os.getenv("USE_HUGGINGFACE_API", "true").lower() == "true"
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "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")
USE_LITE_MODE = os.getenv("USE_LITE_MODE", "false").lower() == "true"

# Modelos
HF_SENTIMENT_MODEL = "foxbell/Feedback-Sentiment-Classifier"
HF_CATEGORY_MODEL = "facebook/bart-large-mnli"

print(f"[ML_Pipeline] 🤖 Using Hugging Face API: {USE_HUGGINGFACE_API}")
print(f"[ML_Pipeline] 🔑 API Key configured: {'Yes' if bool(HUGGINGFACE_API_KEY) else 'No'}")
print(f"[ML_Pipeline] 💡 Lite Mode: {USE_LITE_MODE}")
print(f"[ML_Pipeline] 🎯 Sentiment Model: {HF_SENTIMENT_MODEL}")


# ============================================================
# 🔌 FUNCIÓN BASE PARA LLAMAR A HUGGINGFACE API (robusta)
# ============================================================

def call_huggingface_api(model_name: str, text: str, task_params: Dict = None, max_retries: int = 3, timeout: int = 15) -> Optional[Any]:
    """
    Llama a la API de Hugging Face de forma robusta.
    `task_params` puede ser:
      - {"parameters": {...}}  (ya armada)
      - o un dict con keys custom que se añadirá bajo "parameters"
    Devuelve el JSON tal cual lo retorna HF (list o dict) o None en fallo.
    """
    if not HUGGINGFACE_API_KEY:
        print("[HF_API] ⚠️ No API key configured")
        return None

    url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {"inputs": text}
    # Acepta task_params ya formateado o como "parameters"
    if task_params:
        if "parameters" in task_params:
            payload["parameters"] = task_params["parameters"]
        else:
            payload["parameters"] = task_params

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[HF_API] Calling {model_name} (attempt {attempt}/{max_retries})")
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)

            # Siempre parseamos JSON si podemos (incluso en 200)
            try:
                jj = response.json()
            except ValueError:
                jj = None

            # Manejo por status codes
            if response.status_code == 200:
                # HF a veces devuelve {"error": "... model is loading ..."} con 200
                if isinstance(jj, dict) and jj.get("error"):
                    err_msg = str(jj.get("error"))
                    print(f"[HF_API] ⚠️ HF returned error payload: {err_msg}")
                    # si es loading, esperar y reintentar
                    if "loading" in err_msg.lower() or "model is loading" in err_msg.lower():
                        time.sleep(5)
                        continue
                    # otro error: no retry
                    return None
                # respuesta válida
                return jj
            elif response.status_code == 503:
                print("[HF_API] ⏳ Model loading (503), waiting 5s...")
                time.sleep(5)
                continue
            elif response.status_code == 429:
                print("[HF_API] ⏳ Rate limited (429), waiting 10s...")
                time.sleep(10)
                continue
            else:
                text_snip = response.text[:300]
                print(f"[HF_API] ❌ HTTP {response.status_code}: {text_snip}")
                # si hay JSON con error, mostrarlo
                if jj and isinstance(jj, dict) and jj.get("error"):
                    print(f"[HF_API] ❌ Error body: {jj.get('error')}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"[HF_API] ❌ RequestException (attempt {attempt}): {e}")
            if attempt < max_retries:
                time.sleep(3)
            else:
                return None

    return None


# ============================================================
# 🎭 CLASIFICADOR DE SENTIMIENTO (HF + fallback)
# ============================================================

def _parse_hf_sentiment_response(resp: Any) -> Optional[Tuple[str, float]]:
    """
    Acepta las distintas formas que devuelve HF:
    - Lista de dicts: [{"label":"POSITIVE","score":0.99}, ...]
    - Dict simple: {"label":"positive","score":0.95}
    Devuelve (label_normalizado, score) o None.
    """
    try:
        if resp is None:
            return None
        # Si es lista
        if isinstance(resp, list) and len(resp) > 0 and isinstance(resp[0], dict):
            label = resp[0].get("label", "")
            score = float(resp[0].get("score", 0.0))
            return label, score
        # Si es dict con label
        if isinstance(resp, dict) and ("label" in resp and "score" in resp):
            return resp.get("label"), float(resp.get("score", 0.0))
        # A veces HF devuelve {'error':...} ya filtrado antes, retornará None
    except Exception as e:
        print(f"[HF_API] Error parsing sentiment response: {e}")
    return None


def classify_sentiment(text: str) -> Dict[str, Any]:
    if not text or not isinstance(text, str):
        return {"label": "NEU", "score": 0.5, "source": "default"}

    text = text.strip()[:512]

    if USE_HUGGINGFACE_API and HUGGINGFACE_API_KEY and not USE_LITE_MODE:
        try:
            resp = call_huggingface_api(HF_SENTIMENT_MODEL, text)
            parsed = _parse_hf_sentiment_response(resp)
            if parsed:
                raw_label, score = parsed
                # normalizar label: puede venir 'POSITIVE', 'positive', 'pos', 'LABEL_2', etc.
                l = str(raw_label).lower()
                mapping = {
                    "negative": "NEG", "neg": "NEG", "label_0": "NEG", "0": "NEG",
                    "neutral": "NEU", "neu": "NEU", "label_1": "NEU", "1": "NEU",
                    "positive": "POS", "pos": "POS", "label_2": "POS", "2": "POS",
                    "positive_class": "POS"
                }
                clean_label = mapping.get(l, None)
                if not clean_label:
                    # intentar detectar palabras clave
                    if "pos" in l or "positivo" in l or "good" in l:
                        clean_label = "POS"
                    elif "neg" in l or "negativo" in l or "bad" in l:
                        clean_label = "NEG"
                    elif "neu" in l or "neutral" in l:
                        clean_label = "NEU"
                    else:
                        clean_label = "NEU"

                return {"label": clean_label, "score": round(score, 4), "source": "foxbell_model"}
            else:
                print("[HF_API] ❗ Sentiment: response couldn't be parsed, falling back to lite")

        except Exception as e:
            print(f"[HF_API] Error sentiment call: {e}")

    return classify_sentiment_lite(text)


def classify_sentiment_lite(text: str) -> Dict[str, Any]:
    text_lower = text.lower()

    positive_words = [
        "excelente", "bueno", "fantástico", "increíble", "genial", "perfecto",
        "encantado", "feliz", "satisfecho", "happy", "amazing", "awesome", "love", "excellent"
    ]
    negative_words = [
        "malo", "terrible", "pésimo", "horrible", "odio", "error", "frustrado",
        "enojado", "angry", "worst", "bad", "awful", "disappointed"
    ]

    pos = sum(1 for w in positive_words if w in text_lower)
    neg = sum(1 for w in negative_words if w in text_lower)

    if pos > neg:
        confidence = min(0.95, 0.6 + 0.1 * pos)
        return {"label": "POS", "score": round(confidence, 4), "source": "lite"}
    if neg > pos:
        confidence = min(0.95, 0.6 + 0.1 * neg)
        return {"label": "NEG", "score": round(confidence, 4), "source": "lite"}

    return {"label": "NEU", "score": 0.6, "source": "lite"}


# ============================================================
# 📂 CLASIFICACIÓN DE CATEGORÍA (HF Zero-shot + fallback)
# ============================================================

def _parse_hf_nli_response(resp: Any) -> Optional[Tuple[str, float]]:
    """
    Espera la forma de respuesta de zero-shot: dict con "labels" y "scores".
    Devuelve (best_label, best_score) o None.
    """
    try:
        if resp is None:
            return None
        # respuesta típica: {"sequence": "...", "labels": [...], "scores": [...]}
        if isinstance(resp, dict) and "labels" in resp and "scores" in resp:
            labels = resp.get("labels", [])
            scores = resp.get("scores", [])
            if labels and scores and len(labels) == len(scores):
                return labels[0], float(scores[0])
        # A veces HF devuelve lista de dicts (menos común para NLI) -> no manejamos
    except Exception as e:
        print(f"[HF_API] Error parsing NLI response: {e}")
    return None


def classify_category(text: str) -> Dict[str, Any]:
    if not text or not isinstance(text, str):
        return {"label": "General/Otro", "score": 0.5, "source": "default"}

    text = text.strip()[:1024]

    if USE_HUGGINGFACE_API and HUGGINGFACE_API_KEY and not USE_LITE_MODE:
        try:
            candidate_labels = [
                "Soporte Técnico",
                "Facturación y Pagos",
                "Logística/Envíos",
                "Producto/Sugerencias",
                "General/Otro"
            ]
            # Pasamos los candidate_labels dentro de parameters (estructura esperada)
            task_params = {"parameters": {"candidate_labels": candidate_labels, "multi_class": False}}
            resp = call_huggingface_api(HF_CATEGORY_MODEL, text, task_params)
            parsed = _parse_hf_nli_response(resp)
            if parsed:
                best_label, best_score = parsed
                return {"label": best_label, "score": round(best_score, 4), "source": "huggingface_api"}
            else:
                print("[HF_API] ❗ Category: couldn't parse HF response, using lite fallback")

        except Exception as e:
            print(f"[HF_API] Error category call: {e}")

    return classify_category_lite(text)


def classify_category_lite(text: str) -> Dict[str, Any]:
    text_lower = text.lower()

    categories = {
        "Soporte Técnico": {
            "keywords": ["login", "contraseña", "password", "error", "problema", "bug", "falla", "no funciona", "técnico", "sistema", "app", "aplicación", "website", "crash", "loading", "access", "account", "authentication", "código"],
            "weight": 0.9
        },
        "Logística/Envíos": {
            "keywords": ["envío", "entrega", "delivery", "llegó", "paquete", "enviar", "repartidor", "courier", "shipping", "transport", "arrived", "package", "box", "delay", "tracking", "address", "dirección"],
            "weight": 0.85
        },
        "Facturación y Pagos": {
            "keywords": ["factura", "pago", "cobro", "dinero", "precio", "costo", "billing", "payment", "tarjeta", "cuenta", "charge", "invoice", "refund", "money", "card", "bank", "descuento"],
            "weight": 0.85
        },
        "Producto/Sugerencias": {
            "keywords": ["producto", "calidad", "sugerencia", "mejora", "feature", "funcionalidad", "diseño", "usabilidad", "recommendation", "suggestion", "improve", "quality", "design", "experiencia"],
            "weight": 0.8
        }
    }

    best_category = "General/Otro"
    best_score = 0.6
    max_matches = 0

    for cat, data in categories.items():
        matches = sum(1 for kw in data["keywords"] if kw in text_lower)
        if matches > max_matches:
            max_matches = matches
            best_category = cat
            best_score = min(0.95, data["weight"] + matches * 0.05)
        elif matches == max_matches and matches > 0:
            # empate: elegimos la categoría con mayor weight
            if data["weight"] > best_score:
                best_category = cat
                best_score = min(0.95, data["weight"] + matches * 0.05)

    return {"label": best_category, "score": round(best_score, 4), "source": "lite"}


# ============================================================
# ⚡ CLASIFICADOR DE URGENCIA
# ============================================================

def classify_urgency(text: str) -> str:
    if not text or not isinstance(text, str):
        return "Normal"

    urgent_keywords = [
        "urgente", "emergency", "inmediato", "ya", "ahora", "grave",
        "crítico", "importante", "asap", "rápido", "pronto", "emergencia",
        "immediately", "urgent", "critical", "serious", "help", "ayuda"
    ]

    text_lower = text.lower()
    urgent_matches = sum(1 for k in urgent_keywords if k in text_lower)

    if urgent_matches >= 2:
        return "Crítica"
    elif urgent_matches >= 1:
        return "Alta"
    else:
        return "Normal"


# ============================================================
# 🚀 PIPELINE COMPLETO
# ============================================================

def analyze_feedback(text: str) -> Dict[str, Any]:
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

    print(f"[ML_Pipeline] Analyzing feedback: '{text[:60]}...'")
    start_time = time.time()

    try:
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
                    "label": urgency_result
                }
            },
            "processing_time": round(processing_time, 4),
            "api_mode": "foxbell_huggingface" if (USE_HUGGINGFACE_API and HUGGINGFACE_API_KEY and not USE_LITE_MODE) else "lite"
        }

        print(f"[ML_Pipeline] ✅ Analysis completed in {processing_time:.4f}s")
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
            "processing_time": round(processing_time, 4),
            "error": str(e)
        }


# ============================================================
# 📊 ESTADO DEL SISTEMA
# ============================================================

def get_ml_status() -> Dict[str, Any]:
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
        "fallback_active": not (USE_HUGGINGFACE_API and HUGGINGFACE_API_KEY and not USE_LITE_MODE),
        "status": "ready",
        "timestamp": time.time()
    }


# ============================================================
# 🧪 TESTS DE DEBUG
# ============================================================

if __name__ == "__main__":

    print("\n🧪 TESTING ML_PIPELINE\n")
    status = get_ml_status()
    print(f"Status: {json.dumps(status, indent=2)}\n")

    test_cases = [
        "El producto está excelente, muy buena calidad!",
        "No puedo hacer login, me sale error 500",
        "El delivery nunca llegó a mi casa",
        "Quiero un reembolso de mi pago de $100",
        "URGENTE: necesito ayuda inmediata con mi cuenta"
    ]

    print("=" * 60)
    print("🧪 EJECUTANDO PRUEBAS DE CLASIFICACIÓN")
    print("=" * 60)

    for i, test_text in enumerate(test_cases, 1):
        print(f"\n📝 Prueba {i}/{len(test_cases)}:")
        print(f"Texto: '{test_text}'")
        print("-" * 50)

        result = analyze_feedback(test_text)
        analysis = result["analysis"]
        print(f"🎭 Sentimiento: {analysis['sentiment']['label']} ({analysis['sentiment']['score']}) — Fuente: {analysis['sentiment'].get('source')}")
        print(f"📁 Categoría: {analysis['category']['label']} ({analysis['category']['score']}) — Fuente: {analysis['category'].get('source')}")
        print(f"⚡ Urgencia: {analysis['urgency']['label']}")
        print(f"⏱️ Tiempo: {result['processing_time']}s")
        print(f"🔧 Modo: {result.get('api_mode', 'unknown')}")
        if "error" in result:
            print(f"❌ Error: {result['error']}")

    print("\n" + "=" * 60)
    print("✅ PRUEBAS COMPLETADAS")
    print("=" * 60)

    # Test específico de tu modelo
    print("\n🎯 PROBANDO TU MODELO ESPECÍFICAMENTE:")
    test_sentiment = "Me encanta este producto, es fantástico!"
    result = classify_sentiment(test_sentiment)
    print(f"   Texto: '{test_sentiment}'")
    print(f"   Resultado: {result}")
    if result.get("source") == "foxbell_model":
        print("   ✅ ¡Tu modelo de Hugging Face respondió correctamente!")
    else:
        print("   ⚠️ Se usó fallback - verifica la configuración de HF")
