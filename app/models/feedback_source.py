# app/models/feedback_source.py
from sqlalchemy import Column, Integer, String, Text
from app.config.db import Base

class FeedbackSource(Base):
    __tablename__ = "feedback_source"

    id = Column(Integer, primary_key=True)
    code = Column(String, nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text)
