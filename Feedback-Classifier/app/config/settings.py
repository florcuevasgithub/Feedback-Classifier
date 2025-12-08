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
    
    # ✅ CONFIGURACIONES HUGGING FACE (NUEVAS - OBLIGATORIAS)
    USE_HUGGINGFACE_API: bool = True
    FORCE_CPU: bool = False
    MAX_TEXT_LENGTH: int = 512
    ENABLE_QUANTIZATION: bool = True
    
    # ✅ MODELOS HUGGING FACE
    HF_SENTIMENT_MODEL: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    HF_CATEGORY_MODEL: str = "facebook/bart-large-mnli"
    
    # ✅ CONFIGURACIONES AVANZADAS
    HF_CACHE_DIR: Optional[str] = None
    HF_TOKEN: Optional[str] = None
    MODEL_BATCH_SIZE: int = 1
    MODEL_TIMEOUT: int = 30
    
    # ✅ CONFIGURACIONES DE LOGGING
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
