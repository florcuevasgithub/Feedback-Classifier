# Feedback-Classifier
Agente IA que realiza un análisis automático para clasificar, resumir y detectar patrones en feedback textual proveniente de múltiples fuentes (WhatsApp, formulario web, encuestas), clasifica el feedback en categorías y sentimiento para priorizar mejoras y presenta los insights en un dashboard interactivo.

#Estructura de carpetas
Feedback-Classifier/

├── app/                   Contiene el código fuente del Backend de FastAPI. Aquí va toda la lógica de negocio, endpoints, base de datos y la modularización.
│   ├── config/            Configuración de la IA y Constantes. Aquí se definen los valores fijos de tu modelo: la lista exacta de Categorías, Sentimientos y Urgencias aceptadas por el clasificador. También guarda la clave de la API.
│   ├── crud/              Operaciones de la Base de Datos. Contiene las funciones para guardar los resultados clasificados (Mensaje, Sentimiento, Categoría, Urgencia y Temas) en la DB.
│   ├── metrics/           Cálculo de Insights. Contiene la lógica para agregar los datos clasificados. Se enfoca en calcular la distribución de Sentimiento, el conteo de Urgencia y los Temas más frecuentes.
│   └── auth/              Contiene toda la lógica de la API para verificar la identidad de los usuarios. Aquí se definen los endpoints de login y registro, y la creación y validación de Tokens JWT para proteger el resto de los datos clasificados.
│
├── dashboard/             Contiene el código del Frontend o Panel de Visualización. Es donde se construye la interfaz interactiva que muestra los insights clasificados por la API.
│
├── notebooks/             Área para pruebas rápidas y desarrollo exploratorio. Se usa para probar el modelo de IA o los prompts antes de integrarlos.
│
├── resources/             Almacena los artefactos grandes o binarios del Agente IA. Aquí irían los pesos de modelos NLP, tokenizers o clasificadores entrenados.
│
└── tests/                 Contiene todos los scripts de pruebas automatizadas (unitarias y de integración) para asegurar la calidad y estabilidad de la API y sus módulos.
