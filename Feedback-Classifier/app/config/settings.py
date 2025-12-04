# app/config/settings.py
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "Feedback Classifier API"
    VERSION: str = "1.0.0"

    # Base de datos
    DATABASE_URL: str

    # ✅ URLs para modelos ML (Dropbox)
    MODEL_DOWNLOAD_URL: Optional[str] = "https://www.dropbox.com/scl/fi/your-default-url"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()