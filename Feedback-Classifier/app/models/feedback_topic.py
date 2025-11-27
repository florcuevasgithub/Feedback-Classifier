# app/models/feedback_topic.py
from sqlalchemy import Column, BigInteger, String, Float, ForeignKey
from app.config.db import Base

class FeedbackTopic(Base):
    __tablename__ = "feedback_topic"

    id = Column(BigInteger, primary_key=True)
    feedback_id = Column(BigInteger, ForeignKey("feedback.id"))
    topic_name = Column(String, nullable=False)
    topic_score = Column(Float)
    topic_group = Column(String)
