# 1. DEFINIMOS LOS TEMAS Y SUS KEYWORDS
# El orden ya NO importa, pues la clasificación se basa en la puntuación.
TOPICS_KEYWORDS = {
    # --- TEMAS DE SOPORTE TÉCNICO ---
    "FALLO_LOGIN_ACCESO": [
        "no puedo entrar", "clave", "contraseña", "usuario", "no abre", "sesión"
    ],
    "FALLO_SOFTWARE_APP_WEB": [
        "error", "bug", "se cierra", "caído", "cuelga", "web no funciona", "app"
    ],
    "NECESIDAD_DE_AYUDA": [
        "ayuda", "llamar", "necesito contactar", "soporte", "asistencia"
    ],

    # --- TEMAS DE FACTURACIÓN Y PAGOS ---
    "FALLO_DE_COBRO_PAGO": [
        "cobro", "factura", "pago no pasa", "me cobraron mal", "doble cobro", "pago duplicado"
    ],
    "EXPERIENCIA_POSITIVA_PAGO": [
        "pago realizado a tiempo", "suscripción exitosa", "cobro correcto", "pago a tiempo", 
        "se realizó el pago", "pago sin problemas"
    ],
    "POLITICA_DE_PRECIOS": [
        "caro", "precio", "costo", "valor", "subida", "descuento", "promoción", "rebaja"
    ],
    "DEVOLUCION_Y_REEMBOLSO": [
        "devolver", "reembolso", "devolución", "retorno de dinero"
    ],


    # --- TEMAS DE LOGÍSTICA/ENVÍOS ---
    "RETRASO_DE_ENVIO": [
        "tarda", "demora", "retraso", "lento", "cuando llega", "no llega", "espera"
    ],
    "PROBLEMA_CON_REPARTIDOR": [
        "repartidor", "transportista", "mal servicio de entrega", "no trajo"
    ],
    "DAÑO_DE_PRODUCTO": [
        "roto", "golpeado", "abierto", "dañado", "falta una pieza", "quebrado"
    ],
    "EXPERIENCIA_POSITIVA_ENVIO": [
        "rápido", "a tiempo", "puntual", "llego bien", "repartidor amable"
    ],

    # --- TEMAS DE PRODUCTO/SUGERENCIAS ---
    "CALIDAD_DEL_PRODUCTO": [
        "mala calidad", "calidad baja", "material", "se rompió", "dura poco"
    ],
    "SOLICITUD_DE_FEATURE": [
        "sugerencia", "sería bueno", "me gustaría", "quiero que añadan", "nueva función"
    ]
}

# 2. CONSTANTE PARA ASIGNAR SI NO SE ENCUENTRA NADA
TOPIC_DEFAULT = "OTRO_TEMA_NO_CLASIFICADO"

# Definimos un umbral mínimo de puntuación 
SCORE_THRESHOLD = 0.5 

# 3. LA FUNCIÓN CLASIFICADORA
def classify_topics(text: str) -> dict:
    """
    Clasifica el tema recurrente de un texto usando un sistema de puntuación
    basado en el número de coincidencias de keywords (el que tenga más gana).
    """
    if not text or not isinstance(text, str):
        return {"label": TOPIC_DEFAULT, "score": 0.0}

    text_lower = text.lower()
    scores = {}
    
    # 1. CONTAR COINCIDENCIAS PARA CADA TEMA
    for topic, keywords in TOPICS_KEYWORDS.items():
        score = 0
        # Contamos cuántas palabras clave ÚNICAS de ese tema están en el texto
        for keyword in keywords:
            if keyword in text_lower:
                score += 1
        scores[topic] = score

    # 2. ENCONTRAR EL TEMA GANADOR
    
    # El tema con la mayor puntuación
    best_topic = max(scores, key=scores.get)
    best_score = scores[best_topic]
    
    # 3. APLICAR EL UMBRAL Y DEVOLVER EL RESULTADO
    
    if best_score > SCORE_THRESHOLD: # Si encontramos al menos 1 coincidencia
        # Devolvemos el tema ganador y su puntuación (número de coincidencias)
        return {"label": best_topic, "score": float(best_score)}
    else:
        # Si no hay coincidencias fuertes (score <= 0.5), devolvemos el valor por defecto
        return {"label": TOPIC_DEFAULT, "score": 0.0}

# 4. PRUEBA RÁPIDA
if __name__ == "__main__":
    print("\n--- PRUEBA DE CLASIFICACIÓN DE TEMAS POR PUNTUACIÓN ---")
    
    test_1 = "Se realizó el pago de la suscripción a tiempo."
    test_2 = "El envío llegó rápido y a tiempo. No hay problemas de pago."

    print(f"Texto 1 (Pago): '{test_1}' -> Tema: {classify_topics(test_1)['label']} (Score: {classify_topics(test_1)['score']})")
    print(f"Texto 2 (Envío): '{test_2}' -> Tema: {classify_topics(test_2)['label']} (Score: {classify_topics(test_2)['score']})")