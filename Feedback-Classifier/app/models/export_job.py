# app/models/export_job.py
from sqlalchemy import Column, BigInteger, String, TIMESTAMP, Integer, Text, JSON
from app.config.db import Base

class ExportJob(Base):
    __tablename__ = "export_job"

    id = Column(BigInteger, primary_key=True)
    created_at = Column(TIMESTAMP)
    format = Column(String, nullable=False)
    filter_params = Column(JSON)
    row_count = Column(Integer)
    file_path = Column(Text)
    user_id = Column(BigInteger)
