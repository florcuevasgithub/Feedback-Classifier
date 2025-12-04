# app/config/settings.py
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "Feedback Classifier API"
    VERSION: str = "1.0.0"

    # Base de datos
    DATABASE_URL: str

    # ✅ URL para modelos ML desde Backblaze B2
    BACKBLAZE_MODEL_URL: str = "https://f005.backblazeb2.com/file/Modelosml/modelo_categoria_final.zip"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()