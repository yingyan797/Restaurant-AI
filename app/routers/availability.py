"""
Availability Router for Restaurant Booking API.

This module handles restaurant availability searching functionality,
including time slot availability checks and booking constraint validation.

Author: AI Assistant
"""

from datetime import date
from typing import Dict, Any

from fastapi import APIRouter, Form, Depends, HTTPException, Header
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Restaurant, AvailabilitySlot, Booking

router = APIRouter(prefix="/api/ConsumerApi/v1/Restaurant", tags=["availability"])

# Fixed mock bearer token for authentication
MOCK_BEARER_TOKEN = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFwcGVsbGErYXBpQHJlc2"
    "RpYXJ5LmNvbSIsIm5iZiI6MTc1NDQzMDgwNSwiZXhwIjoxNzU0NTE3MjA1LCJpYXQiOjE3NTQ0MzA4"
    "MDUsImlzcyI6IlNlbGYiLCJhdWQiOiJodHRwczovL2FwaS5yZXNkaWFyeS5jb20ifQ.g3yLsufdk8Fn"
    "2094SB3J3XW-KdBc0DY9a2Jiu_56ud8"
)


def verify_token(authorization: str = Header(...)) -> str:
    """
    Verify the bearer token in the Authorization header.

    Args:
        authorization: The Authorization header value

    Returns:
        str: The validated token

    Raises:
        HTTPException: If token is invalid or header format is wrong
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header format"
        )

    token = authorization.replace("Bearer ", "")
    if token != MOCK_BEARER_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )

    return token


@router.post(
    "/{restaurant_name}/AvailabilitySearch",
    summary="Search Available Time Slots",
    response_description="Available booking slots with availability status"
)
async def availability_search(
    restaurant_name: str,
    VisitDate: date = Form(..., description="Visit date in YYYY-MM-DD format"),
    PartySize: int = Form(..., description="Number of people in the party"),
    ChannelCode: str = Form(..., description="Booking channel (e.g., 'ONLINE')"),
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
) -> Dict[str, Any]:
    """
    Search for available booking slots at a restaurant.

    Retrieves available time slots for a specific restaurant, date, and party size.
    The system checks base availability slots and current booking counts to determine
    real-time availability.

    Args:
        restaurant_name: The name of the restaurant
        VisitDate: The desired visit date
        PartySize: Number of people in the party
        ChannelCode: The booking channel identifier
        db: Database session dependency
        token: Authentication token dependency

    Returns:
        Dict containing restaurant info and available time slots

    Raises:
        HTTPException: 404 if restaurant not found
        HTTPException: 401 if authentication fails
    """
    # Find restaurant by name
    restaurant = db.query(Restaurant).filter(Restaurant.name == restaurant_name).first()
    if not restaurant:
        raise HTTPException(status_code=404, detail="Restaurant not found")

    # Get availability slots for the requested date
    slots = db.query(AvailabilitySlot).filter(
        AvailabilitySlot.restaurant_id == restaurant.id,
        AvailabilitySlot.date == VisitDate,
        AvailabilitySlot.max_party_size >= PartySize
    ).all()

    # Check for existing bookings at each slot time
    available_slots = []
    for slot in slots:
        # Count existing bookings for this time slot
        existing_bookings = db.query(Booking).filter(
            Booking.restaurant_id == restaurant.id,
            Booking.visit_date == VisitDate,
            Booking.visit_time == slot.time,
            Booking.status == "confirmed"
        ).count()

        # Simple logic: allow up to 3 bookings per time slot
        max_bookings_per_slot = 3
        is_available = slot.available and existing_bookings < max_bookings_per_slot

        available_slots.append({
            "time": slot.time.strftime("%H:%M:%S"),
            "available": is_available,
            "max_party_size": slot.max_party_size,
            "current_bookings": existing_bookings
        })

    return {
        "restaurant": restaurant_name,
        "restaurant_id": restaurant.id,
        "visit_date": VisitDate,
        "party_size": PartySize,
        "channel_code": ChannelCode,
        "available_slots": available_slots,
        "total_slots": len(available_slots)
    }
