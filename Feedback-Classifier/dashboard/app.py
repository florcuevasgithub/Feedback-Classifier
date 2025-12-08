import streamlit as st
import requests
import os
import json

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/feedback/submit")

def display_submit_form():
    st.title("Carga Manual de Feedback")
    st.markdown("---")

    with st.form(key='feedback_form'):
        message_text = st.text_area(
            "Mensaje de Feedback (WhatsApp, Web, etc.):",
            height=200,
            key='message_input',
            placeholder="Escribe el mensaje aquí..."
        )
        
        source = st.selectbox(
            "Fuente del Mensaje:",
            options=['WhatsApp', 'Formulario Web', 'Encuesta', 'Otro'],
            key='source_select'
        )
        
        submit_button = st.form_submit_button(label='Clasificar y Guardar Feedback')

    if submit_button:
        if not message_text:
            st.error("Por favor, introduce un mensaje de feedback.")
            return

        # Mapeo de fuentes
        source_mapping = {
            'WhatsApp': 1,
            'Formulario Web': 2, 
            'Encuesta': 3,
            'Otro': 4
        }

        payload = {
            "text": message_text,
            "source_id": source_mapping[source],
            "external_id": message_text[:20]  # ✅ Usar parte del texto como ID
        }
        
        # ✅ DEBUGGING: Mostrar información de conexión
        st.info(f"🔗 Conectando a API: {API_URL}")
        
        try:
            with st.spinner('Procesando feedback...'):
                response = requests.post(
                    API_URL, 
                    json=payload,
                    timeout=30,  # ✅ Timeout de 30 segundos
                    headers={'Content-Type': 'application/json'}
                )
            
            # ✅ DEBUGGING: Mostrar detalles de respuesta
            st.write(f"**Status Code:** {response.status_code}")
            st.write(f"**Response Headers:** {dict(response.headers)}")
            st.write(f"**Raw Response:** `{response.text[:200]}...`")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    st.success("¡Feedback guardado con éxito!")
                    st.subheader("Resultados de la Clasificación:")
                    st.json(data)
                except json.JSONDecodeError as e:
                    st.error(f"❌ Error decodificando JSON: {e}")
                    st.text(f"Respuesta raw: {response.text}")
                    
            elif response.status_code == 422:
                st.error("❌ Error de validación de datos")
                st.code(response.text)
                
            else:
                st.error(f"❌ Error del servidor: {response.status_code}")
                st.code(response.text)
                
        except requests.exceptions.ConnectionError as e:
            st.error(f"❌ Error de conexión: {e}")
            st.info("Verifica que la API esté funcionando en: " + API_URL)
            
        except requests.exceptions.Timeout:
            st.error("⏰ Timeout: La API tardó más de 30 segundos en responder")
            
        except Exception as e:
            st.error(f"❌ Error inesperado: {e}")

# ✅ NUEVO: Página de prueba de conectividad
def test_api_connection():
    st.sidebar.markdown("## 🔍 Test API")
    if st.sidebar.button("Probar Conexión"):
        try:
            # Test del endpoint de health
            health_url = API_URL.replace('/submit', '').rstrip('/') + '/health'
            response = requests.get(health_url, timeout=10)
            
            if response.status_code == 200:
                st.sidebar.success("✅ API conectada")
            else:
                st.sidebar.error(f"❌ API error: {response.status_code}")
                
        except Exception as e:
            st.sidebar.error(f"❌ No se pudo conectar: {e}")

if __name__ == '__main__':
    test_api_connection()
    display_submit_form()
