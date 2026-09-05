from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class UsageEventRecord(Base):
    __tablename__ = "usage_events"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    event_id: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        nullable=False,
        index=True,
    )

    request_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )

    capability: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    provider: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    model: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    tokens_in: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    tokens_out: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    estimated_cost: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )

    latency_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )
