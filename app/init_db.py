"""
Database Initialization Module.

This module handles database table creation and population with sample data
for the restaurant booking mock API. It sets up realistic test data including
restaurants, availability slots, and cancellation reasons.

Author: AI Assistant
"""

import random
from datetime import time, datetime, timedelta

from app.database import engine, SessionLocal
from app.models import Base, Restaurant, AvailabilitySlot, CancellationReason


def create_tables() -> None:
    """
    Create all database tables based on SQLAlchemy models.

    This function creates the database schema by calling SQLAlchemy's
    metadata.create_all() method.
    """
    Base.metadata.create_all(bind=engine)


def init_sample_data() -> None:
    """
    Initialize database with sample data for testing.

    Creates a sample restaurant with availability slots and cancellation reasons.
    This function is idempotent - it will skip initialization if data already exists.

    Sample data includes:
    - A restaurant named "TheHungryUnicorn"
    - 30 days of availability slots with lunch and dinner times
    - 5 predefined cancellation reasons

    Raises:
        Exception: If database operations fail (logged and rolled back)
    """
    db = SessionLocal()

    try:
        # Check if data already exists
        if db.query(Restaurant).first():
            print("Sample data already exists, skipping initialization")
            return

        # Create sample restaurant
        restaurant = Restaurant(
            name="TheHungryUnicorn",
            microsite_name="TheHungryUnicorn"
        )
        db.add(restaurant)
        db.commit()
        db.refresh(restaurant)

        # Create sample availability slots for the next 30 days
        sample_times = [
            time(12, 0),   # 12:00 PM
            time(12, 30),  # 12:30 PM
            time(13, 0),   # 1:00 PM
            time(13, 30),  # 1:30 PM
            time(19, 0),   # 7:00 PM
            time(19, 30),  # 7:30 PM
            time(20, 0),   # 8:00 PM
            time(20, 30),  # 8:30 PM
        ]

        start_date = datetime.now().date()

        for i in range(30):  # Next 30 days
            current_date = start_date + timedelta(days=i)
            for slot_time in sample_times:
                # Randomly make some slots unavailable
                available = random.random() > 0.2  # 80% availability

                slot = AvailabilitySlot(
                    restaurant_id=restaurant.id,
                    date=current_date,
                    time=slot_time,
                    max_party_size=8,
                    available=available
                )
                db.add(slot)

        # Create sample cancellation reasons
        cancellation_reasons = [
            {
                "id": 1,
                "reason": "Customer Request",
                "description": "Customer requested cancellation"
            },
            {
                "id": 2,
                "reason": "Restaurant Closure",
                "description": "Restaurant temporarily closed"
            },
            {
                "id": 3,
                "reason": "Weather",
                "description": "Cancelled due to weather conditions"
            },
            {"id": 4, "reason": "Emergency", "description": "Emergency cancellation"},
            {"id": 5, "reason": "No Show", "description": "Customer did not show up"}
        ]

        for reason_data in cancellation_reasons:
            reason = CancellationReason(**reason_data)
            db.add(reason)

        db.commit()
        print("Database initialized with sample data successfully!")

    except Exception as e:
        print(f"Error initializing database: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    print("Creating database tables...")
    create_tables()
    print("Initializing sample data...")
    init_sample_data()
    print("Database setup complete!")
