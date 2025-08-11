# Restaurant Booking Mock API Server

A complete mock restaurant booking management system built with FastAPI and SQLite. This server provides realistic restaurant booking endpoints for developers to integrate with their applications.

**Purpose**: This mock server simulates a real restaurant booking system, allowing developers to build and test application integrations without needing access to production restaurant APIs.

## Project Structure

```
GFDE test/
├── app/
│   ├── __init__.py
│   ├── __main__.py          # Module entry point (python -m app)
│   ├── main.py              # Main FastAPI application
│   ├── database.py          # Database configuration
│   ├── models.py            # SQLAlchemy database models
│   ├── init_db.py           # Database initialization script
│   └── routers/
│       ├── __init__.py
│       ├── availability.py  # Availability search endpoints
│       └── booking.py       # Booking management endpoints
├── requirements.txt
├── restaurant_booking.db    # SQLite database (created automatically)
└── README.md
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Initialise database manually:
```bash
python app/init_db.py
```
*Note: The database will be automatically initialized when you start the server for the first time.*

## Running the Server

### Development Mode (Recommended)
```bash
python -m app
```

### Alternative Development Mode
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8547
```

## Database Features

- **SQLite Database**: Lightweight, file-based database (`restaurant_booking.db`)
- **Automatic Setup**: Database tables and sample data created on first run
- **Models**:
  - `Restaurant`: Restaurant information and microsite names
  - `Customer`: Customer details with marketing preferences
  - `Booking`: Booking records with full relationship mapping
  - `AvailabilitySlot`: Time slots for restaurant availability
  - `CancellationReason`: Predefined cancellation reasons
- **Sample Data**: 30 days of availability slots and cancellation reasons

## Authentication

All endpoints require a Bearer token in the Authorization header.

**Required Header:**
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFwcGVsbGErYXBpQHJlc2RpYXJ5LmNvbSIsIm5iZiI6MTc1NDQzMDgwNSwiZXhwIjoxNzU0NTE3MjA1LCJpYXQiOjE3NTQ0MzA4MDUsImlzcyI6IlNlbGYiLCJhdWQiOiJodHRwczovL2FwaS5yZXNkaWFyeS5jb20ifQ.g3yLsufdk8Fn2094SB3J3XW-KdBc0DY9a2Jiu_56ud8
```

**Authentication Errors:**
- **401 Unauthorised**: Missing, invalid, or expired token
- **401 Unauthorised**: Invalid authorisation header format

## API Endpoints

All endpoints use `application/x-www-form-urlencoded` content type for POST/PATCH requests and require the Authorization header.

### 1. Search Available Time Slots
**POST** `/api/ConsumerApi/v1/Restaurant/{restaurant_name}/AvailabilitySearch`

Returns available booking slots for a specific date and party size.

**Parameters:**
- `VisitDate`: Date in YYYY-MM-DD format (required)
- `PartySize`: Number of people (required)
- `ChannelCode`: Booking channel, typically "ONLINE" (required)

**Response:**
```json
{
  "restaurant": "TheHungryUnicorn",
  "restaurant_id": 1,
  "visit_date": "2025-08-06",
  "party_size": 2,
  "channel_code": "ONLINE",
  "available_slots": [
    {
      "time": "12:00:00",
      "available": true,
      "max_party_size": 8,
      "current_bookings": 0
    }
  ],
  "total_slots": 8
}
```

### 2. Create New Booking
**POST** `/api/ConsumerApi/v1/Restaurant/{restaurant_name}/BookingWithStripeToken`

Creates a new restaurant booking with customer information.

**Required Parameters:**
- `VisitDate`: Date in YYYY-MM-DD format
- `VisitTime`: Time in HH:MM:SS format
- `PartySize`: Number of people
- `ChannelCode`: Booking channel (e.g., "ONLINE")

**Optional Parameters:**
- `SpecialRequests`: Special requirements text
- `IsLeaveTimeConfirmed`: Boolean for time confirmation
- `RoomNumber`: Specific room/table number

**Customer Information (all optional):**
- `Customer[Title]`: Mr/Mrs/Ms/Dr
- `Customer[FirstName]`: Customer's first name
- `Customer[Surname]`: Customer's last name
- `Customer[Email]`: Email address
- `Customer[Mobile]`: Mobile phone number
- `Customer[Phone]`: Landline phone number
- `Customer[MobileCountryCode]`: Mobile country code
- `Customer[PhoneCountryCode]`: Phone country code
- `Customer[ReceiveEmailMarketing]`: Boolean for email marketing consent
- `Customer[ReceiveSmsMarketing]`: Boolean for SMS marketing consent

**Response:**
```json
{
  "booking_reference": "ABC1234",
  "booking_id": 1,
  "restaurant": "TheHungryUnicorn",
  "visit_date": "2025-08-06",
  "visit_time": "12:30:00",
  "party_size": 4,
  "status": "confirmed",
  "customer": {
    "id": 1,
    "first_name": "John",
    "surname": "Smith",
    "email": "john@example.com"
  },
  "created_at": "2025-08-06T10:30:00.123456"
}
```

### 3. Get Booking Details
**GET** `/api/ConsumerApi/v1/Restaurant/{restaurant_name}/Booking/{booking_reference}`

Retrieves complete booking information.

**Response:**
```json
{
  "booking_reference": "ABC1234",
  "booking_id": 1,
  "restaurant": "TheHungryUnicorn",
  "visit_date": "2025-08-06",
  "visit_time": "12:30:00",
  "party_size": 4,
  "status": "confirmed",
  "special_requests": "Window table please",
  "customer": {
    "id": 1,
    "first_name": "John",
    "surname": "Smith",
    "email": "john@example.com",
    "mobile": "1234567890"
  },
  "created_at": "2025-08-06T10:30:00.123456",
  "updated_at": "2025-08-06T10:30:00.123456"
}
```

### 4. Update Booking
**PATCH** `/api/ConsumerApi/v1/Restaurant/{restaurant_name}/Booking/{booking_reference}`

Modifies an existing booking. Only provide fields you want to change.

**Optional Parameters:**
- `VisitDate`: New date (YYYY-MM-DD)
- `VisitTime`: New time (HH:MM:SS)
- `PartySize`: New party size
- `SpecialRequests`: Updated special requests
- `IsLeaveTimeConfirmed`: Time confirmation status

**Response:**
```json
{
  "booking_reference": "ABC1234",
  "booking_id": 1,
  "restaurant": "TheHungryUnicorn",
  "updates": {
    "party_size": 6,
    "special_requests": "Updated request"
  },
  "status": "updated",
  "updated_at": "2025-08-06T11:30:00.123456",
  "message": "Booking ABC1234 has been successfully updated"
}
```

### 5. Cancel Booking
**POST** `/api/ConsumerApi/v1/Restaurant/{restaurant_name}/Booking/{booking_reference}/Cancel`

Cancels an existing booking with a reason.

**Parameters:**
- `micrositeName`: Restaurant microsite name (same as restaurant_name)
- `bookingReference`: Booking reference (same as in URL)
- `cancellationReasonId`: Reason ID (1-5, see cancellation reasons below)

**Response:**
```json
{
  "booking_reference": "ABC1234",
  "booking_id": 1,
  "restaurant": "TheHungryUnicorn",
  "cancellation_reason_id": 1,
  "cancellation_reason": "Customer Request",
  "status": "cancelled",
  "cancelled_at": "2025-08-06T12:30:00.123456",
  "message": "Booking ABC1234 has been successfully cancelled"
}
```

## Cancellation Reasons

| ID | Reason | Description |
|----|--------|-------------|
| 1 | Customer Request | Customer requested cancellation |
| 2 | Restaurant Closure | Restaurant temporarily closed |
| 3 | Weather | Cancelled due to weather conditions |
| 4 | Emergency | Emergency cancellation |
| 5 | No Show | Customer did not show up |

## Error Responses

All endpoints return appropriate HTTP status codes:

- **200 OK**: Successful operation
- **400 Bad Request**: Invalid parameters or business rule violation
- **404 Not Found**: Restaurant or booking not found
- **422 Unprocessable Entity**: Validation errors

Error response format:
```json
{
  "detail": "Error description"
}
```

## API Documentation

Once the server is running, you can access:
- Interactive API docs: http://localhost:8547/docs
- Alternative docs: http://localhost:8547/redoc

## Example Requests

### 1. Check Availability
```bash
curl -X POST "http://localhost:8547/api/ConsumerApi/v1/Restaurant/TheHungryUnicorn/AvailabilitySearch" \
     -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFwcGVsbGErYXBpQHJlc2RpYXJ5LmNvbSIsIm5iZiI6MTc1NDQzMDgwNSwiZXhwIjoxNzU0NTE3MjA1LCJpYXQiOjE3NTQ0MzA4MDUsImlzcyI6IlNlbGYiLCJhdWQiOiJodHRwczovL2FwaS5yZXNkaWFyeS5jb20ifQ.g3yLsufdk8Fn2094SB3J3XW-KdBc0DY9a2Jiu_56ud8" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "VisitDate=2025-08-06&PartySize=2&ChannelCode=ONLINE"
```

### 2. Make a Booking
```bash
curl -X POST "http://localhost:8547/api/ConsumerApi/v1/Restaurant/TheHungryUnicorn/BookingWithStripeToken" \
     -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFwcGVsbGErYXBpQHJlc2RpYXJ5LmNvbSIsIm5iZiI6MTc1NDQzMDgwNSwiZXhwIjoxNzU0NTE3MjA1LCJpYXQiOjE3NTQ0MzA4MDUsImlzcyI6IlNlbGYiLCJhdWQiOiJodHRwczovL2FwaS5yZXNkaWFyeS5jb20ifQ.g3yLsufdk8Fn2094SB3J3XW-KdBc0DY9a2Jiu_56ud8" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "VisitDate=2025-08-06&VisitTime=12:30:00&PartySize=4&ChannelCode=ONLINE&SpecialRequests=Window table please&Customer[FirstName]=John&Customer[Surname]=Smith&Customer[Email]=john@example.com&Customer[Mobile]=1234567890"
```

### 3. Get Booking Details
```bash
curl -X GET "http://localhost:8547/api/ConsumerApi/v1/Restaurant/TheHungryUnicorn/Booking/ABC1234" \
     -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFwcGVsbGErYXBpQHJlc2RpYXJ5LmNvbSIsIm5iZiI6MTc1NDQzMDgwNSwiZXhwIjoxNzU0NTE3MjA1LCJpYXQiOjE3NTQ0MzA4MDUsImlzcyI6IlNlbGYiLCJhdWQiOiJodHRwczovL2FwaS5yZXNkaWFyeS5jb20ifQ.g3yLsufdk8Fn2094SB3J3XW-KdBc0DY9a2Jiu_56ud8"
```

### 4. Update Booking
```bash
curl -X PATCH "http://localhost:8547/api/ConsumerApi/v1/Restaurant/TheHungryUnicorn/Booking/ABC1234" \
     -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFwcGVsbGErYXBpQHJlc2RpYXJ5LmNvbSIsIm5iZiI6MTc1NDQzMDgwNSwiZXhwIjoxNzU0NTE3MjA1LCJpYXQiOjE3NTQ0MzA4MDUsImlzcyI6IlNlbGYiLCJhdWQiOiJodHRwczovL2FwaS5yZXNkaWFyeS5jb20ifQ.g3yLsufdk8Fn2094SB3J3XW-KdBc0DY9a2Jiu_56ud8" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "PartySize=6&SpecialRequests=Updated request"
```

### 5. Cancel Booking
```bash
curl -X POST "http://localhost:8547/api/ConsumerApi/v1/Restaurant/TheHungryUnicorn/Booking/ABC1234/Cancel" \
     -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFwcGVsbGErYXBpQHJlc2RpYXJ5LmNvbSIsIm5iZiI6MTc1NDQzMDgwNSwiZXhwIjoxNzU0NTE3MjA1LCJpYXQiOjE3NTQ0MzA4MDUsImlzcyI6IlNlbGYiLCJhdWQiOiJodHRwczovL2FwaS5yZXNkaWFyeS5jb20ifQ.g3yLsufdk8Fn2094SB3J3XW-KdBc0DY9a2Jiu_56ud8" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "micrositeName=TheHungryUnicorn&bookingReference=ABC1234&cancellationReasonId=1"
```

## Database Operations

The API now includes full CRUD operations with SQLite:

- **Create**: New bookings are stored with unique references
- **Read**: Availability checks real booking data and time slots
- **Update**: Booking modifications are tracked with timestamps
- **Delete**: Cancellations are soft-deleted with reason tracking

### Sample Data Included

- Restaurant: "TheHungryUnicorn" with availability slots
- Time slots: 12:00-14:00 and 19:00-21:00 (30-minute intervals)
- 30 days of future availability
- 5 predefined cancellation reasons

## Mock Data & Behaviour

- **Sample Restaurant**: "TheHungryUnicorn" is pre-loaded with availability data
- **Time Slots**: Available lunch (12:00-13:30) and dinner (19:00-20:30) slots
- **Availability**: Some slots randomly marked as unavailable to simulate real conditions
- **Booking References**: Auto-generated 7-character alphanumeric codes
- **Fixed Authentication**: Uses a single mock bearer token for all requests
- **Persistent Data**: All bookings saved to SQLite database
- **Realistic Responses**: All endpoints return realistic restaurant booking data

## Technical Details

- **Database**: SQLite with persistent storage
- **Port**: Server runs on localhost:8547 by default
- **Auto-reload**: Development server watches for code changes
- **CORS**: Enabled for cross-origin requests
- **Validation**: Request validation with helpful error messages

# AI integration
- Framework: LangGraph
## Features
- Assist users in finding their booking reference and restaurant names when using the booking form
- All booking and availablility functions are designed with AI, but some have problem running
- Simulate user sign in with an email, to prevent accessing private information of others
## Usage
- Put a google gemini API key in .env: "gemapi=..."
