# -*- encoding: utf-8 -*-
"""ORM models for the SQLAlchemy Postgres Plugin."""

import uuid

import pgvector.sqlalchemy
from sqlalchemy import Column, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from nendo.library import model

Base = model.Base


class NendoEmbeddingDB(Base):
    __tablename__ = "embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    track_id = Column(UUID(as_uuid=True), ForeignKey("tracks.id"))
    user_id = Column(UUID(as_uuid=True))
    plugin_name = Column(String)
    plugin_version = Column(String)
    text = Column(String)
    embedding = Column(pgvector.sqlalchemy.Vector())

    # Relationship to NendoTrack
    track = relationship("NendoTrackDB")
