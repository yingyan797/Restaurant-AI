"""
SQLAlchemy Database Models for Restaurant Booking System.

This module defines the database schema and relationships for the restaurant
booking mock API. All models inherit from the declarative base and include
proper relationships and constraints.

Author: AI Assistant
"""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, Date, Time, Text, ForeignKey
)
from sqlalchemy.orm import relationship

from app.database import Base

if TYPE_CHECKING:
    # Import for type hints only - avoids circular imports
    pass


class Restaurant(Base):
    """
    Restaurant model representing individual restaurant entities.

    Each restaurant has a unique name and microsite identifier, along with
    associated bookings and availability slots.

    Attributes:
        id (int): Primary key identifier
        name (str): Unique restaurant name
        microsite_name (str): Unique microsite identifier for the restaurant
        created_at (datetime): Timestamp when restaurant was created
        bookings: Related booking records
        availability_slots: Related availability slot records
    """

    __tablename__ = "restaurants"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    microsite_name = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    bookings = relationship("Booking", back_populates="restaurant")
    availability_slots = relationship("AvailabilitySlot", back_populates="restaurant")


class Customer(Base):
    """
    Customer model storing customer information and marketing preferences.

    Stores all customer details including contact information and consent
    for various marketing communications.

    Attributes:
        id (int): Primary key identifier
        title (str): Customer title (Mr/Mrs/Ms/Dr)
        first_name (str): Customer's first name
        surname (str): Customer's surname
        email (str): Customer's email address (indexed)
        mobile (str): Customer's mobile phone number
        phone (str): Customer's landline phone number
        created_at (datetime): Timestamp when customer was created
        bookings: Related booking records
    """

    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    first_name = Column(String)
    surname = Column(String)
    mobile_country_code = Column(String)
    mobile = Column(String)
    phone_country_code = Column(String)
    phone = Column(String)
    email = Column(String, index=True)
    receive_email_marketing = Column(Boolean, default=False)
    receive_sms_marketing = Column(Boolean, default=False)
    group_email_marketing_opt_in_text = Column(Text)
    group_sms_marketing_opt_in_text = Column(Text)
    receive_restaurant_email_marketing = Column(Boolean, default=False)
    receive_restaurant_sms_marketing = Column(Boolean, default=False)
    restaurant_email_marketing_opt_in_text = Column(Text)
    restaurant_sms_marketing_opt_in_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    bookings = relationship("Booking", back_populates="customer")

class Booking(Base):
    """
    Booking model representing restaurant reservations.

    Central model that links customers to restaurants with specific
    date/time slots and booking details.

    Attributes:
        id (int): Primary key identifier
        booking_reference (str): Unique booking reference code
        restaurant_id (int): Foreign key to restaurant
        customer_id (int): Foreign key to customer
        visit_date (date): Date of the booking
        visit_time (time): Time of the booking
        party_size (int): Number of people in the booking
        status (str): Booking status (confirmed/cancelled/completed)
        created_at (datetime): Timestamp when booking was created
        updated_at (datetime): Timestamp when booking was last updated
    """

    __tablename__ = "bookings"

    id = Column(Integer, primary_key=True, index=True)
    booking_reference = Column(String, unique=True, index=True, nullable=False)
    restaurant_id = Column(Integer, ForeignKey("restaurants.id"), nullable=False)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    visit_date = Column(Date, nullable=False)
    visit_time = Column(Time, nullable=False)
    party_size = Column(Integer, nullable=False)
    channel_code = Column(String, nullable=False)
    special_requests = Column(Text)
    is_leave_time_confirmed = Column(Boolean, default=False)
    room_number = Column(String)
    status = Column(String, default="confirmed")  # confirmed, cancelled, completed
    cancellation_reason_id = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    restaurant = relationship("Restaurant", back_populates="bookings")
    customer = relationship("Customer", back_populates="bookings")


class AvailabilitySlot(Base):
    """
    Availability slot model defining when restaurants accept bookings.

    Represents specific date/time combinations when a restaurant
    can accept bookings, with capacity constraints.

    Attributes:
        id (int): Primary key identifier
        restaurant_id (int): Foreign key to restaurant
        date (date): Date of availability
        time (time): Time slot
        max_party_size (int): Maximum party size for this slot
        available (bool): Whether the slot is available for booking
        created_at (datetime): Timestamp when slot was created
    """

    __tablename__ = "availability_slots"

    id = Column(Integer, primary_key=True, index=True)
    restaurant_id = Column(Integer, ForeignKey("restaurants.id"), nullable=False)
    date = Column(Date, nullable=False)
    time = Column(Time, nullable=False)
    max_party_size = Column(Integer, default=8)
    available = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    restaurant = relationship("Restaurant", back_populates="availability_slots")


class CancellationReason(Base):
    """
    Cancellation reason model for tracking why bookings are cancelled.

    Provides predefined reasons for booking cancellations to maintain
    consistent data and enable reporting.

    Attributes:
        id (int): Primary key identifier
        reason (str): Short reason description
        description (str): Detailed reason description
    """

    __tablename__ = "cancellation_reasons"

    id = Column(Integer, primary_key=True, index=True)
    reason = Column(String, nullable=False)
    description = Column(Text)
