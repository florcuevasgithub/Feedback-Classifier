# app/models/feedback.py
from sqlalchemy import Column, Integer, String, Float, Text, TIMESTAMP, JSON, Boolean
from app.config.db import Base

class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)  # 👈 cambio

    source_id = Column(Integer, nullable=False)
    external_id = Column(String)
    channel = Column(String)
    text_original = Column(Text, nullable=False)
    text_clean = Column(Text)
    received_at = Column(TIMESTAMP)
    analyzed_at = Column(TIMESTAMP)
    sentiment_label = Column(String)
    sentiment_score = Column(Float)
    urgency_label = Column(String)
    category_label = Column(String)
    category_score = Column(Float)
    is_training_sample = Column(Boolean, default=False)
    extra_metadata = Column(JSON)
    created_by = Column(Integer)
