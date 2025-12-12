# 1. DEFINIMOS LAS KEYWORDS
# Estas son las listas que el sistema usará para
# clasificar la urgencia de forma instantánea, las definimos nosotros.
KEYWORDS_URGENCIA_ALTA = [
    "roto", "no funciona", "desastre", "caído", "error fatal",
    "urgente", "inmediato", "no puedo trabajar", "estafa",
    "fraude", "no acceso"
]

KEYWORDS_URGENCIA_MEDIA = [
    "lento", "tarda mucho", "problema", "ayuda", "duda",
    "consulta", "pregunta", "mejorar", "sugerencia"
]


# 2. LA FUNCIÓN CLASIFICADORA (entregable)
def classify_urgency(text: str) -> str:
    """
    Clasifica la urgencia de un texto basado en keywords predefinidas.

    Args:
        text (str): El feedback del cliente.

    Returns:
        str: La etiqueta de urgencia ('ALTA', 'MEDIA', 'BAJA').
    """
    if not text or not isinstance(text, str):
        return "BAJA"

    # Convertimos a minúscula para una búsqueda 'case-insensitive'
    text_lower = text.lower()

    # Buscamos primero las keywords de alta urgencia
    if any(keyword in text_lower for keyword in KEYWORDS_URGENCIA_ALTA):
        return "ALTA"

    # Si no, buscamos las de media urgencia
    if any(keyword in text_lower for keyword in KEYWORDS_URGENCIA_MEDIA):
        return "MEDIA"

    # Si no encuentra nada, es urgencia baja
    return "BAJA"

# 3. PRUEBA RÁPIDA (Para probar el archivo)
if __name__ == "__main__":
    print("\n--- PRUEBA DE URGENCIA (REGLAS) ---")

    test_1 = "¡Es un desastre! ¡¡No funciona!!"
    test_2 = "Hola, tengo una pregunta sobre el envío."
    test_3 = "Todo bien, gracias."

    print(f"Texto: '{test_1}'")
    print(f"Resultado: {classify_urgency(test_1)}")

    print(f"\nTexto: '{test_2}'")
    print(f"Resultado: {classify_urgency(test_2)}")

    print(f"\nTexto: '{test_3}'")
    print(f"Resultado: {classify_urgency(test_3)}")