from fastapi import FastAPI


app = FastAPI(
    title="Agente IA de Análisis de Feedback",
    version="1.0.0",
    description="Backend para clasificar, almacenar y exportar feedback de usuarios."
)


@app.get("/", tags=["Status"])
def read_root():
    return {"message": "Bienvenido al Agente IA de Análisis de Feedback. Visita /docs para ver los endpoints."}

@app.get("/health", tags=["Status"])
def health_check():
    
    return {"status": "ok", "service": "Feedback-Classifier API"}

