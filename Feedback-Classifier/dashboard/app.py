import streamlit as st
import requests
import os
import json

# ✅ URL CORREGIDA - El endpoint correcto de tu API
API_URL = os.getenv("API_URL", "https://feedback-classifier-2i6k.onrender.com/api/feedback/submit")
HEALTH_URL = os.getenv("HEALTH_URL", "https://feedback-classifier-2i6k.onrender.com/health")

# 🔥 Mapeo real de fuentes → IDs que tu API espera
SOURCE_MAP = {
    "web": 1,
    "whatsapp": 2,
    "app": 3,
    "email": 4,
    "survey": 5,
    "test": 99
}

def display_submit_form():
    st.title("🎯 Feedback Classifier - Dashboard")
    st.markdown("Clasifica feedback usando tu modelo de Hugging Face")
    st.markdown("---")

    # 🔗 INFO DE URLS
    with st.expander("🔗 Información de Conexión"):
        st.code(f"API URL: {API_URL}")
        st.code(f"Health URL: {HEALTH_URL}")

    # FORMULARIO
    with st.form(key='feedback_form'):
        message_text = st.text_area(
            "📝 Texto del Feedback:",
            height=200,
            key='message_input',
            placeholder="Ejemplo: 'Me encanta este producto, es increíble y funciona perfecto!'"
        )
        
        source = st.selectbox(
            "📍 Fuente del Mensaje:",
            options=list(SOURCE_MAP.keys()),
            key='source_select',
            help="Selecciona la fuente del feedback"
        )
        
        submit_button = st.form_submit_button(label='🚀 Clasificar Feedback')

    # SUBMIT
    if submit_button:
        if not message_text or len(message_text.strip()) < 5:
            st.error("❌ Por favor, introduce un mensaje de feedback de al menos 5 caracteres.")
            return

        # 🚀 PAYLOAD CORRECTO PARA TU API
        payload = {
            "text": message_text.strip(),
            "source_id": SOURCE_MAP[source],
            "external_id": "streamlit-ui"
        }

        # DEBUG
        with st.expander("🔍 Datos enviados (Debug)"):
            st.json(payload)

        try:
            with st.spinner('🤖 Procesando con tu modelo foxbell/Feedback-Sentiment-Classifier...'):
                response = requests.post(
                    API_URL, 
                    json=payload,
                    timeout=30,
                    headers={
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    }
                )
            
            st.write(f"**Status Code:** {response.status_code}")
            
            # 200 OK
            if response.status_code == 200:
                try:
                    data = response.json()
                    st.success("✅ ¡Clasificación completada!")

                    analysis = data.get('analysis', {})

                    col1, col2, col3 = st.columns(3)

                    # Sentimiento
                    with col1:
                        sentiment = analysis.get('sentiment', {})
                        st.metric(
                            label="🎭 Sentimiento",
                            value=sentiment.get('label', 'N/A'),
                            delta=f"{sentiment.get('score', 0):.3f}"
                        )
                        st.caption(f"Fuente: {sentiment.get('source', 'N/A')}")

                    # Categoría
                    with col2:
                        category = analysis.get('category', {})
                        st.metric(
                            label="📁 Categoría",
                            value=category.get('label', 'N/A'),
                            delta=f"{category.get('score', 0):.3f}"
                        )
                        st.caption(f"Fuente: {category.get('source', 'N/A')}")

                    # Urgencia
                    with col3:
                        urgency = analysis.get('urgency', {})
                        st.metric(
                            label="⚡ Urgencia",
                            value=urgency.get('label', 'Normal')
                        )

                    # Respuesta completa
                    with st.expander("📊 Respuesta Completa"):
                        st.json(data)

                    processing_time = data.get('processing_time', 0)
                    api_mode = data.get('api_mode', 'unknown')
                    st.info(f"⏱️ Tiempo: {processing_time}s | 🔧 Modo: {api_mode}")

                except json.JSONDecodeError as e:
                    st.error(f"❌ Error decodificando JSON: {e}")
                    st.text("Respuesta raw:")
                    st.code(response.text)

            # 422 validation error
            elif response.status_code == 422:
                st.error("❌ Error de validación de datos (422)")
                try:
                    st.json(response.json())
                except:
                    st.code(response.text)

            # 404
            elif response.status_code == 404:
                st.error("❌ Endpoint no encontrado (404)")
                st.error(f"Verifica que la URL sea correcta: {API_URL}")

            # Otro error
            else:
                st.error(f"❌ Error del servidor ({response.status_code})")
                st.code(response.text[:500])
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Error de conexión: No se puede conectar a la API")
            st.error(f"URL intentada: {API_URL}")

        except requests.exceptions.Timeout:
            st.error("⏰ Timeout: La API tardó más de 30 segundos en responder")

        except Exception as e:
            st.error(f"❌ Error inesperado: {e}")

def test_api_connection():
    """Función para probar la conectividad con la API"""
    st.sidebar.markdown("## 🔍 Estado de la API")
    
    if st.sidebar.button("Probar Conexión"):
        with st.sidebar:
            with st.spinner("Probando..."):
                try:
                    response = requests.get(HEALTH_URL, timeout=10)
                    
                    if response.status_code == 200:
                        st.success("✅ API Online")
                        try:
                            st.json(response.json())
                        except:
                            st.text("Health check OK")
                    else:
                        st.error(f"❌ API Error: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"❌ Conexión falló")
                    st.caption(str(e))
    
    if st.sidebar.button("Estado ML"):
        with st.sidebar:
            with st.spinner("Consultando..."):
                try:
                    ml_status_url = API_URL.replace('/submit', '/ml-status')
                    response = requests.get(ml_status_url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success("✅ ML System")
                        st.json(data)
                    else:
                        st.error(f"❌ ML Status: {response.status_code}")
                        
                except Exception as e:
                    st.error("❌ ML Status falló")
                    st.caption(str(e))

def main():
    st.set_page_config(
        page_title="Feedback Classifier",
        page_icon="🎯",
        layout="wide"
    )
    
    test_api_connection()
    display_submit_form()
    
    st.markdown("---")
    st.markdown("**🚀 Powered by:** Hugging Face + FastAPI + Streamlit")

if __name__ == '__main__':
    main()
