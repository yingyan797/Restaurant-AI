"""
Database Configuration and Session Management.

This module sets up the SQLite database connection, session management,
and declarative base for the restaurant booking mock API.

Author: AI Assistant
"""

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# SQLite database URL - creates file in project root
SQLALCHEMY_DATABASE_URL = "sqlite:///./restaurant_booking.db"

# Create SQLAlchemy engine with SQLite-specific configuration
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}  # Required for SQLite threading
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create declarative base for all models
Base = declarative_base()

def get_db() -> Generator[Session, None, None]:
    """
    Database session dependency for FastAPI.

    Creates a new database session for each request and ensures
    it's properly closed after the request completes.

    Yields:
        Session: SQLAlchemy database session

    Example:
        Use as a FastAPI dependency:
        ```python
        @app.get("/example")
        def example_endpoint(db: Session = Depends(get_db)):
            # Use db session here
            pass
        ```
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
