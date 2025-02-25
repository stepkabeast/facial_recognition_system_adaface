from sqlalchemy import Column, String, Integer
from sqlalchemy.dialects.postgresql import BYTEA
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class PortfolioEmbeddingInfo(Base):
    __tablename__ = "face_embedding_info"

    id = Column(Integer, primary_key=True, autoincrement=True)  # index in faiss + 1
    portfolio_id = Column(String, primary_key=True)  # face index
    vector = Column(BYTEA)

