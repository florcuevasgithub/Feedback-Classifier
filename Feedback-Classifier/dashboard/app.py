import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/feedback/submit"

def display_submit_form():
    
    st.title("🗣️ Carga Manual de Feedback")
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

        payload = {
            "message_text": message_text,
            "source": source
        }
        
        try:
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 201:
                data = response.json()
                st.success("¡Feedback guardado con éxito!")
                
                st.subheader("Resultados de la Clasificación:")
                st.json(data)
                
            else:
                st.error(f"Error al procesar la solicitud: {response.status_code}")
                st.json(response.json())
                
        except requests.exceptions.ConnectionError:
            st.error("Error de conexión.")


if __name__ == '__main__':
    display_submit_form()
  
