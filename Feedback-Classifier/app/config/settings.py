# app/config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Feedback Classifier API"
    VERSION: str = "1.0.0"

    # Cargado desde .env
    DATABASE_URL: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
