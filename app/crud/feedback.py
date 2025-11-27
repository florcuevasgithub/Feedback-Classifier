# app/crud/feedback.py
from datetime import datetime
from app.models.feedback import Feedback
from app.config.db import SessionLocal

def save_feedback(data, ml):
    db = SessionLocal()

    fb = Feedback(
        text_original=data.text,
        source_id=data.source_id,
        external_id=data.external_id,
        sentiment_label=ml["analysis"]["sentiment"]["label"],
        sentiment_score=ml["analysis"]["sentiment"]["score"],
        urgency_label=ml["analysis"]["urgency"]["label"],
        category_label=ml["analysis"]["category"]["label"],
        category_score=ml["analysis"]["category"]["score"],
        analyzed_at=datetime.utcnow()
    )

    db.add(fb)
    db.commit()
    db.refresh(fb)
    return fb
