# app/models/app_user.py
from sqlalchemy import Column, BigInteger, Boolean, Text, String, TIMESTAMP
from app.config.db import Base

class AppUser(Base):
    __tablename__ = "app_user"

    id = Column(BigInteger, primary_key=True)
    email = Column(String, nullable=False, unique=True)
    password_hash = Column(Text, nullable=False)
    full_name = Column(String)
    created_at = Column(TIMESTAMP, nullable=False)
    is_active = Column(Boolean, default=True)
