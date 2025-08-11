"""
Restaurant Booking Mock API Server.

A complete FastAPI-based mock server that simulates a restaurant booking system.
This server provides realistic endpoints for availability checking, booking creation,
booking management, and cancellation operations.

Author: AI Assistant
Version: 1.0.0
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.routers import availability, booking, chatbot
from app.database import engine
from app.models import Base
import app.init_db as init_db

# Create database tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Restaurant Booking Mock API",
    description=(
        "A complete mock restaurant booking management system built with FastAPI "
        "and SQLite. Provides realistic endpoints for testing applications."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Include API routers
app.include_router(availability.router)
app.include_router(booking.router)
app.include_router(chatbot.router)

# Serve static files (HTML, JS, CSS)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.on_event("startup")
async def startup_event() -> None:
    """
    Initialize database with sample data on application startup.

    This function is called once when the FastAPI application starts.
    It ensures the database contains sample restaurant data and availability slots.
    """
    init_db.init_sample_data()


@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the frontend HTML page."""
    return FileResponse("app/static/index.html")

@app.get("/api", summary="API Information", tags=["Root"])
async def root() -> dict:
    """
    Get API information and available endpoints.

    Returns:
        dict: API metadata including version and available endpoint URLs.
    """
    return {
        "message": "Restaurant Booking Mock API",
        "version": "1.0.0",
        "description": "Mock restaurant booking system for testing applications",
        "endpoints": {
            "availability_search": (
                "/api/ConsumerApi/v1/Restaurant/{restaurant_name}/"
                "AvailabilitySearch"
            ),
            "create_booking": (
                "/api/ConsumerApi/v1/Restaurant/{restaurant_name}/"
                "BookingWithStripeToken"
            ),
            "cancel_booking": (
                "/api/ConsumerApi/v1/Restaurant/{restaurant_name}/Booking/"
                "{booking_reference}/Cancel"
            ),
            "get_booking": (
                "/api/ConsumerApi/v1/Restaurant/{restaurant_name}/Booking/"
                "{booking_reference}"
            ),
            "update_booking": (
                "/api/ConsumerApi/v1/Restaurant/{restaurant_name}/Booking/"
                "{booking_reference}"
            ),
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }
