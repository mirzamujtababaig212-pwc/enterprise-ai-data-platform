from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ai_platform.llm_gateway.config.settings import settings


DATABASE_URL = (
    "postgresql+psycopg2://"
    f"{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
    f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}"
    f"/{settings.POSTGRES_DB}"
)


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    future=True,
)


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()
