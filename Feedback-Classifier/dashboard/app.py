import streamlit as st
import requests
import pandas as pd
from typing import List, Dict, Any

st.set_page_config(
    page_title="Feedback Classifier Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
API_URL = "http://127.0.0.1:8000"
SUBMIT_ENDPOINT = f"{API_URL}/feedback/submit"
DATA_ENDPOINT = f"{API_URL}/feedback/data"

def display_submit_form():
    """Muestra el formulario de carga manual de feedback en la barra lateral."""
    st.sidebar.title("Carga Manual de Feedback")
    st.sidebar.markdown("---")

    with st.sidebar.form(key='feedback_form'): 
        
        message_text = st.text_area(
            "Mensaje de Feedback:",
            height=150,
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
            st.sidebar.error("Por favor, introduce un mensaje de feedback.")
            return

        payload = {
            "message_text": message_text,
            "source": source
        }
        
        try:
            response = requests.post(SUBMIT_ENDPOINT, json=payload)
            
            if response.status_code == 201:
                data = response.json()
                st.sidebar.success("¡Feedback guardado con éxito!")
                st.sidebar.json(data)
                st.experimental_rerun()
                
            else:
                st.sidebar.error(f"Error al procesar la solicitud: {response.status_code}")
                st.sidebar.json(response.json())
                
        except requests.exceptions.ConnectionError:
            st.sidebar.error("Error de conexión. Servidor FastAPI no responde en :8000.")


def fetch_and_display_data_table():
    """Obtiene y muestra todos los feedbacks clasificados en una tabla."""
    st.header("📋 Feedbacks Clasificados (Tabla Filtrable)")
    
    try:
        response = requests.get(DATA_ENDPOINT)
        
        if response.status_code == 200:
            data = response.json()
            
            if data:
                df = pd.DataFrame(data)
                
                column_order = ['id', 'created_at', 'sentiment', 'category', 'urgency', 'topics', 'message_text', 'source']
                df = df.reindex(columns=column_order, fill_value='')

                st.dataframe(df, use_container_width=True)

            else:
                st.info("Aún no hay feedbacks clasificados en la base de datos.")
        
        else:
            st.error(f"Error al obtener datos de la API: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        st.warning("No se pudo conectar a la API.")


if __name__ == '__main__':
    st.title("Dashboard de Análisis de Feedback")
 
    display_submit_form()

    fetch_and_display_data_table()
