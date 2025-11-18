# Feedback-Classifier
Agente IA que realiza un análisis automático para clasificar, resumir y detectar patrones en feedback textual proveniente de múltiples fuentes (WhatsApp, formulario web, encuestas), clasifica el feedback en categorías y sentimiento para priorizar mejoras y presenta los insights en un dashboard interactivo.
# Estructura del Proyecto

### I. Archivos Raíz

| Archivo/Carpeta | Concepto |
| :--- | :--- |
| `Feedback-Classifier/` | **Carpeta Raíz.** Contiene todos los módulos y la configuración del proyecto. |
| `main.py` | **Script de Ejecución Principal.** Punto de entrada para inicializar la API, el entrenamiento o tareas del sistema. |
| `.gitignore` | **Control de Versiones.** Define los archivos y carpetas que Git debe ignorar (logs, modelos entrenados, claves de entorno). |
| `README.md` | **Documentación Principal.** Este archivo. |

### II. Componentes de la Aplicación 

| Carpeta | Contenido y Propósito |
| :--- | :--- |
| `app/` | **Lógica de la API.** Contiene los módulos principales para exponer el modelo de clasificación como un servicio *backend* (API REST). |
| ├── `auth/` | **Seguridad.** Gestión de autenticación, autorización y roles de usuario. |
| ├── `crud/` | **Operaciones de BD.** Lógica para interactuar con la base de datos (guardar feedback clasificado, gestionar datos de entrenamiento). |
| └── `config/` | **Ajustes del Sistema.** Archivos de configuración para la conexión a la base de datos, variables de entorno y *settings* del servidor. |

---

### III. Data  y Análisis

| Carpeta | Contenido y Propósito |
| :--- | :--- |
| `notebooks/` | **Exploración.** Entornos para el Análisis Exploratorio de Datos (EDA) y la experimentación con nuevos modelos o algoritmos. |
| `metrics/` | **Evaluación del Modelo.** *Scripts* dedicados a calcular y registrar métricas de rendimiento del modelo (precisión, *recall*, etc.) después del entrenamiento o en producción. |
| `resources/` | **Activos Estáticos.** Almacenamiento de archivos auxiliares necesarios para el modelo (ej. diccionarios de *stopwords*, *vocabulario* o *fixtures* de prueba). |

---

### IV. Presentación y Calidad

| Carpeta | Contenido y Propósito |
| :--- | :--- |
| `tests/` | **Pruebas de Software.** Pruebas unitarias y de integración para asegurar la funcionalidad correcta de los módulos (`crud`, `metrics`, etc.). |
| `dashboard/` | **Visualización.** Archivos necesarios para el *dashboard* interactivo que monitorea el rendimiento y las tendencias del feedback clasificado en tiempo real. |
