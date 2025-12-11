import streamlit as st
import requests
import os
import json

# ✅ URL CORREGIDA - El endpoint correcto de tu API
API_URL = os.getenv("API_URL", "https://feedback-classifier-2i6k.onrender.com/api/feedback/submit")
HEALTH_URL = os.getenv("HEALTH_URL", "https://feedback-classifier-2i6k.onrender.com/health")

def display_submit_form():
    st.title("🎯 Feedback Classifier - Dashboard")
    st.markdown("Clasifica feedback usando tu modelo de Hugging Face")
    st.markdown("---")

    # ✅ INFORMACIÓN DE CONEXIÓN
    with st.expander("🔗 Información de Conexión"):
        st.code(f"API URL: {API_URL}")
        st.code(f"Health URL: {HEALTH_URL}")

    with st.form(key='feedback_form'):
        message_text = st.text_area(
            "📝 Texto del Feedback:",
            height=200,
            key='message_input',
            placeholder="Ejemplo: 'Me encanta este producto, es increíble y funciona perfecto!'"
        )
        
        source = st.selectbox(
            "📍 Fuente del Mensaje:",
            options=['web', 'whatsapp', 'app', 'email', 'survey', 'test'],
            key='source_select',
            help="Selecciona la fuente del feedback"
        )
        
        submit_button = st.form_submit_button(label='🚀 Clasificar Feedback')

    if submit_button:
        if not message_text or len(message_text.strip()) < 5:
            st.error("❌ Por favor, introduce un mensaje de feedback de al menos 5 caracteres.")
            return

        # ✅ PAYLOAD CORRECTO según tu API
        payload = {
            "feedback_text": message_text.strip(),
            "source": source
        }
        
        # ✅ MOSTRAR PAYLOAD PARA DEBUG
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
            
            # ✅ MOSTRAR DETALLES DE RESPUESTA
            st.write(f"**Status Code:** {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    st.success("✅ ¡Clasificación completada!")
                    
                    # ✅ MOSTRAR RESULTADOS DE FORMA BONITA
                    col1, col2, col3 = st.columns(3)
                    
                    analysis = data.get('analysis', {})
                    
                    with col1:
                        sentiment = analysis.get('sentiment', {})
                        st.metric(
                            label="🎭 Sentimiento",
                            value=sentiment.get('label', 'N/A'),
                            delta=f"{sentiment.get('score', 0):.3f}"
                        )
                        st.caption(f"Fuente: {sentiment.get('source', 'N/A')}")
                    
                    with col2:
                        category = analysis.get('category', {})
                        st.metric(
                            label="📁 Categoría", 
                            value=category.get('label', 'N/A'),
                            delta=f"{category.get('score', 0):.3f}"
                        )
                        st.caption(f"Fuente: {category.get('source', 'N/A')}")
                    
                    with col3:
                        urgency = analysis.get('urgency', {})
                        st.metric(
                            label="⚡ Urgencia",
                            value=urgency.get('label', 'Normal')
                        )
                    
                    # ✅ MOSTRAR DETALLES COMPLETOS
                    with st.expander("📊 Respuesta Completa"):
                        st.json(data)
                    
                    # ✅ MOSTRAR INFORMACIÓN DE PROCESAMIENTO
                    processing_time = data.get('processing_time', 0)
                    api_mode = data.get('api_mode', 'unknown')
                    st.info(f"⏱️ Tiempo: {processing_time}s | 🔧 Modo: {api_mode}")
                    
                    # ✅ VERIFICAR SI SE USÓ TU MODELO
                    if sentiment.get('source') == 'foxbell_model':
                        st.success("🎉 ¡Se usó tu modelo personalizado de Hugging Face!")
                    else:
                        st.warning(f"⚠️ Se usó fallback: {sentiment.get('source')}")
                        
                except json.JSONDecodeError as e:
                    st.error(f"❌ Error decodificando JSON: {e}")
                    st.text("Respuesta raw:")
                    st.code(response.text)
                    
            elif response.status_code == 422:
                st.error("❌ Error de validación de datos (422)")
                try:
                    error_data = response.json()
                    st.json(error_data)
                except:
                    st.code(response.text)
                    
            elif response.status_code == 404:
                st.error("❌ Endpoint no encontrado (404)")
                st.error(f"Verifica que la URL sea correcta: {API_URL}")
                st.info("💡 Asegúrate de que la API esté desplegada y el endpoint exista")
                
            else:
                st.error(f"❌ Error del servidor ({response.status_code})")
                st.code(response.text[:500])
                
        except requests.exceptions.ConnectionError as e:
            st.error(f"❌ Error de conexión: No se puede conectar a la API")
            st.error(f"URL intentada: {API_URL}")
            st.info("💡 Verifica que la API esté funcionando y la URL sea correcta")
            
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
                    # Test del endpoint de health
                    response = requests.get(HEALTH_URL, timeout=10)
                    
                    if response.status_code == 200:
                        st.success("✅ API Online")
                        try:
                            health_data = response.json()
                            st.json(health_data)
                        except:
                            st.text("Health check OK")
                    else:
                        st.error(f"❌ API Error: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"❌ Conexión falló")
                    st.caption(str(e))
    
    # ✅ MOSTRAR INFORMACIÓN DE ML STATUS
    if st.sidebar.button("Estado ML"):
        with st.sidebar:
            with st.spinner("Consultando..."):
                try:
                    ml_status_url = API_URL.replace('/submit', '/ml-status')
                    response = requests.get(ml_status_url, timeout=10)
                    
                    if response.status_code == 200:
                        status_data = response.json()
                        st.success("✅ ML System")
                        
                        hf_config = status_data.get('huggingface_api', {})
                        if hf_config.get('enabled'):
                            st.write("🤖 HF API: Activa")
                            models = hf_config.get('models', {})
                            st.write(f"🎯 Sentiment: {models.get('sentiment', 'N/A')}")
                        else:
                            st.write("🤖 HF API: Desactiva")
                            
                    else:
                        st.error(f"❌ ML Status: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"❌ ML Status falló")
                    st.caption(str(e))

def main():
    # ✅ CONFIGURACIÓN DE PÁGINA
    st.set_page_config(
        page_title="Feedback Classifier",
        page_icon="🎯",
        layout="wide"
    )
    
    # ✅ SIDEBAR CON TESTS
    test_api_connection()
    
    # ✅ CONTENIDO PRINCIPAL
    display_submit_form()
    
    # ✅ FOOTER
    st.markdown("---")
    st.markdown("**🚀 Powered by:** Hugging Face + FastAPI + Streamlit")

if __name__ == '__main__':
    main()
