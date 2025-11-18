# Feedback-Classifier
Agente IA que realiza un análisis automático para clasificar, resumir y detectar patrones en feedback textual proveniente de múltiples fuentes (WhatsApp, formulario web, encuestas), clasifica el feedback en categorías y sentimiento para priorizar mejoras y presenta los insights en un dashboard interactivo.
# Estructura del Proyecto

Feedback-Classifier/,Raíz del Proyecto. Contiene la configuración principal y los scripts de ejecución.
main.py,"Script de Ejecución Principal. Punto de entrada para iniciar la aplicación, el entrenamiento o las tareas principales."
app/,Lógica de la Aplicación (API/Servicio). Contiene el código para exponer el modelo como un servicio backend (API REST).
auth/,"Autenticación y Autorización. Módulos para manejar la seguridad de la API, roles de usuario y tokens de acceso."
config/,"Configuración del Proyecto. Archivos de configuración (YAML, JSON, .env) para bases de datos, settings de la aplicación y parámetros del modelo."
crud/,"Operaciones de Base de Datos. Lógica para la creación, lectura, actualización y eliminación de registros (e.g., gestión de datos de entrenamiento o feedback clasificado)."
metrics/,"Evaluación y Monitoreo. Scripts para calcular y registrar métricas de rendimiento del modelo (precisión, recall, F1-score) y métricas de negocio."
dashboard/,"Visualización de Resultados. Scripts o archivos necesarios para desplegar un dashboard interactivo (ej. Streamlit, Dash) que muestre el rendimiento del modelo en tiempo real."
notebooks/,"Exploración y Experimentación. Jupyter notebooks utilizados para el Análisis Exploratorio de Datos (EDA), prototyping del modelo y pruebas rápidas."
resources/,"Archivos Estáticos y Datos Pequeños. Contiene recursos no-código como stopwords, diccionarios, fixtures de prueba o modelos pre-entrenados ligeros."
tests/,"Pruebas Unitarias y de Integración. Código para verificar que las funciones individuales (crud, metrics) y el modelo se comporten según lo esperado."
.gitignore,"Control de Versiones. Especifica archivos y carpetas que Git debe ignorar (ej. variables de entorno, modelos pesados, carpetas temporales)."
README.md,"Documentación Principal. Este archivo, que proporciona una visión general del proyecto, la instalación, la estructura y el uso."
