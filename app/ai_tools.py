"""
This module provides a unified interface for AI agents to query database tables
with support for generic chained operations and LangGraph workflows.

Author: AI Assistant
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Literal
from dataclasses import dataclass
from langchain_core.tools import tool
import logging, inspect, asyncio, concurrent.futures
from sqlalchemy import text
from datetime import date, time, datetime
from fastapi import HTTPException
from app.database import get_db, Base, Session
from app.models import Restaurant, Customer, Booking, AvailabilitySlot, CancellationReason
from app.routers.availability import availability_search, MOCK_BEARER_TOKEN
from app.routers.booking import create_booking_with_stripe, cancel_booking, get_booking, update_booking

def _serialize_value(value):
    """Serialize database values for JSON compatibility."""
    if hasattr(value, 'isoformat'):  # datetime objects
        return value.isoformat()
    return value

# Helper to run async functions from a sync context
def run_async_in_sync(coro):
    """
    Runs an async coroutine in a synchronous context.
    If an event loop is already running, it executes the coroutine in a new thread.
    Otherwise, it runs it in the current thread.
    """
    try:
        loop = asyncio.get_running_loop()
        # If an event loop is already running, run the coroutine in a new thread
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No event loop running, safe to use asyncio.run in the current thread
        return asyncio.run(coro)

# --- Helper for Database Session Management ---
def get_db_session() -> Optional[Session]:
    """Helper to get a database session. Used by tool wrappers."""
    try:
        db = next(get_db())
        return db
    except Exception as e:
        return None

class AIToolCallingInterface:
    '''Export all functions and API callable by LangGraph AI agents'''

    def get_customer_information(email: str):
        """Get customer information by email"""
        db = get_db_session()
        if not db: return {"error": "Database connection error."}
        try:
            query = db.query(Customer).filter(Customer.email == email)
            customer = query.one()
            customer_info = {}
            for column in customer.__table__.columns:
                value = getattr(customer, column.name)
                customer_info[column.name] = _serialize_value(value)
            return customer_info

        except HTTPException as e: return {"error": e.detail, "status_code": e.status_code}
        except Exception as e: return {"error": str(e)}
        finally: db.close()

    def list_restaurants_tool():
        """
        Lists all available restaurants registered in the system.
        This tool does not take any parameters and returns a comprehensive list of all restaurants.
        """
        db = get_db_session()
        if not db: return {"error": "Database connection error."}
        try:
            query_output = db.execute(text("select * from restaurants")).fetchall()
            restaurants = [{col.name: row[i] for i, col in enumerate(Restaurant.__table__.columns)} for row in query_output]
            return restaurants

        except HTTPException as e: return {"error": e.detail, "status_code": e.status_code}
        except Exception as e: return {"error": str(e)}
        finally: db.close()

    def list_cancellation_reasons_tool():
        """
        Retrieves all predefined cancellation reasons with their ID, title, and description.
        This is useful when a user wishes to cancel a booking and needs to specify a reason ID.
        """
        db = get_db_session()
        if not db: return {"error": "Database connection error."}
        try:
            query = db.query(CancellationReason)
            reasons = []
            for reason in query.all():
                reason_info = {}
                for column in reason.__table__.columns:
                    value = getattr(reason, column.name)
                    reason_info[column.name] = _serialize_value(value)
                reasons.append(reason_info)
            return reasons

        except HTTPException as e: return {"error": e.detail, "status_code": e.status_code}
        except Exception as e: return {"error": str(e)}
        finally: db.close()

    def find_customer_bookings_tool(
            email: str,
            date_range: Optional[List[str]] = None,
            time_range: Optional[List[str]] = None,
            restaurant_name: Optional[str] = None,
            party_size_range: Optional[List[int]] = None,
            status: Literal['cancelled', 'confirmed', 'completed'] = None,
            is_leave_time_confirmed: Optional[bool] = None,
            room_number: Optional[str] = None
        ) -> List[Dict[str, Any]]:
        """
        Retrieves a list of bookings for a specific customer, identified by their email.
        This tool is used to find existing reservations.

        Args:
            email (str, provided): The email address of the customer whose bookings are being searched. This parameter is already known from the user.
            date_range (List[str], optional): A list of two strings representing the start and end dates (ISO format YYYY-MM-DD) to filter bookings by visit date. Example: ["2025-01-01", "2025-01-31"].
            time_range (List[str], optional): A list of two strings representing the start and end times (ISO format HH:MM:SS) to filter bookings by visit time. Example: ["18:00:00", "20:00:00"].
            restaurant_name (str, optional): The name of the restaurant to filter bookings for.
            party_size_range (List[int], optional): A list of two integers representing the minimum and maximum party sizes to filter bookings. Example: [2, 4].
            status (Literal['cancelled', 'confirmed', 'completed'], optional): The current status of the booking to filter by.
            is_leave_time_confirmed (bool, optional): Filter by whether the leave time for the booking is confirmed.
            room_number (str, optional): Filter by the room number associated with the booking.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents a booking and includes details like booking reference, restaurant name, and other booking specifics. Returns an empty list if no bookings are found.
        """
        db = get_db_session()
        if not db: return {"error": "Database connection error."}
        try:
            customer_info = AIToolCallingInterface.get_customer_information(email)
            if "error" in customer_info:
                return {"error": f"Customer with email '{email}' not found or error retrieving info: {customer_info['error']}"}
            customer_id = customer_info["id"]

            statement = (
                "select booking_reference, restaurants.name as restaurant_name "
                "from bookings left join restaurants on bookings.restaurant_id = restaurants.id "
                f"where customer_id={customer_id}"
            )

            def put_value(v):
                if isinstance(v, str):
                    return f"'{v}'"
                # Convert boolean to integer for SQL
                if isinstance(v, bool):
                    return "1" if v else "0"
                return str(v)

            # Apply filters based on provided arguments
            if date_range is not None and len(date_range) == 2:
                start_date, end_date = date_range
                # Ensure date format is correct for SQL and handle None within range
                if start_date and end_date:
                    statement += f" and visit_date between {put_value(start_date)} and {put_value(end_date)}"
                elif start_date:
                    statement += f" and visit_date >= {put_value(start_date)}"
                elif end_date:
                    statement += f" and visit_date <= {put_value(end_date)}"

            if time_range is not None and len(time_range) == 2:
                start_time, end_time = time_range
                # Ensure time format is correct for SQL and handle None within range
                if start_time and end_time:
                    statement += f" and visit_time between {put_value(start_time)} and {put_value(end_time)}"
                elif start_time:
                    statement += f" and visit_time >= {put_value(start_time)}"
                elif end_time:
                    statement += f" and visit_time <= {put_value(end_time)}"

            if restaurant_name is not None:
                statement += f" and restaurants.name={put_value(restaurant_name)}"

            if party_size_range is not None and len(party_size_range) == 2:
                min_party_size, max_party_size = party_size_range
                if min_party_size is not None and max_party_size is not None:
                    statement += f" and party_size between {put_value(min_party_size)} and {put_value(max_party_size)}"
                elif min_party_size is not None:
                    statement += f" and party_size >= {put_value(min_party_size)}"
                elif max_party_size is not None:
                    statement += f" and party_size <= {put_value(max_party_size)}"

            if status is not None:
                statement += f" and status={put_value(status)}"

            if is_leave_time_confirmed is not None:
                statement += f" and is_leave_time_confirmed={put_value(is_leave_time_confirmed)}"

            if room_number is not None:
                statement += f" and room_number={put_value(room_number)}"

            # Execute the constructed query
            bookings_raw = db.execute(text(statement)).fetchall()
            
            # Fetch full booking details using the predefined get_booking method
            detailed_bookings = []
            for booking_ref_tuple in bookings_raw:
                try:
                    # Use the new run_async_in_sync helper
                    booking_details = run_async_in_sync(
                        get_booking(booking_ref_tuple[1], booking_ref_tuple[0], db, MOCK_BEARER_TOKEN)
                    )
                    detailed_bookings.append(booking_details)
                except Exception as e:
                    logging.warning(f"Error retrieving details for booking {booking_ref_tuple[0]}: {str(e)}")
                    continue

            return detailed_bookings
            
        except HTTPException as e:
            logging.error(f"HTTPException in find_customer_bookings: {e.detail}")
            return {"error": e.detail, "status_code": e.status_code}
        except Exception as e:
            logging.exception("An unexpected error occurred in find_customer_bookings")
            return {"error": str(e)}
        finally:
            db.close()

    def search_availability_tool(restaurant_name: str, visit_date: str, party_size: int, channel_code: str = "ONLINE") -> Dict[str, Any]:
        """
        Searches for available booking slots at a specific restaurant on a given date for a certain party size.

        Args:
            restaurant_name (str): The name of the restaurant to check availability for.
            visit_date (str): The desired date for the visit in ISO format (YYYY-MM-DD).
            party_size (int): The number of people in the party.
            channel_code (str, optional): The channel through which the booking is made (default is "ONLINE").

        Returns:
            Dict[str, Any]: A dictionary containing a list of available slots if any, or an empty list if no availability is found.
        """
        db = get_db_session()
        if not db: return {"error": "Database connection error."}
        try:
            # CORRECTED: Use run_async_in_sync to call the async availability_search
            result = run_async_in_sync(
                availability_search(restaurant_name, date.fromisoformat(visit_date), party_size, channel_code, db, MOCK_BEARER_TOKEN)
            )
            return result
        except HTTPException as e: return {"error": e.detail, "status_code": e.status_code}
        except Exception as e: return {"error": str(e)}
        finally: db.close()

    def create_booking_tool(
        restaurant_name: str, visit_date: str, visit_time: str, party_size: int,
        customer_email: str, customer_first_name: str, customer_surname: str, customer_mobile: str,
        channel_code: str = "ONLINE", special_requests: Optional[str] = None,
        is_leave_time_confirmed: Optional[bool] = False, room_number: Optional[str] = None,
        customer_title: Optional[str] = None, customer_mobile_country_code: Optional[str] = None,
        customer_phone_country_code: Optional[str] = None, customer_phone: Optional[str] = None,
        customer_receive_email_marketing: Optional[bool] = False, customer_receive_sms_marketing: Optional[bool] = False,
        customer_group_email_marketing_opt_in_text: Optional[str] = None, customer_group_sms_marketing_opt_in_text: Optional[str] = None,
        customer_receive_restaurant_email_marketing: Optional[bool] = False, customer_receive_restaurant_sms_marketing: Optional[bool] = False,
        customer_restaurant_email_marketing_opt_in_text: Optional[str] = None, customer_restaurant_sms_marketing_opt_in_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Creates a new restaurant booking. This tool requires specific details about the reservation and the customer.

        Args:
            restaurant_name (str): The name of the restaurant for the booking.
            visit_date (str): The date of the visit in ISO format (YYYY-MM-DD).
            visit_time (str): The time of the visit in ISO format (HH:MM:SS).
            party_size (int): The number of people for the reservation.
            customer_email (str): The email address of the customer.
            customer_first_name (str): The first name of the customer.
            customer_surname (str): The surname of the customer.
            customer_mobile (str): The mobile phone number of the customer.
            channel_code (str, optional): The channel through which the booking is made (default is "ONLINE").
            special_requests (str, optional): Any special requests for the booking (e.g., dietary restrictions, seating preferences).
            is_leave_time_confirmed (bool, optional): Indicates if the departure time is confirmed.
            room_number (str, optional): The room number if the customer is staying at a hotel associated with the restaurant.
            customer_title (str, optional): Title of the customer (e.g., Mr., Ms., Dr.).
            customer_mobile_country_code (str, optional): Mobile phone country code for the customer.
            customer_phone_country_code (str, optional): Landline phone country code for the customer.
            customer_phone (str, optional): Landline phone number for the customer.
            customer_receive_email_marketing (bool, optional): Opt-in for general email marketing.
            customer_receive_sms_marketing (bool, optional): Opt-in for general SMS marketing.
            customer_group_email_marketing_opt_in_text (str, optional): Text for group email marketing opt-in.
            customer_group_sms_marketing_opt_in_text (str, optional): Text for group SMS marketing opt-in.
            customer_receive_restaurant_email_marketing (bool, optional): Opt-in for restaurant-specific email marketing.
            customer_receive_restaurant_sms_marketing (bool, optional): Opt-in for restaurant-specific SMS marketing.
            customer_restaurant_email_marketing_opt_in_text (str, optional): Text for restaurant email marketing opt-in.
            customer_restaurant_sms_marketing_opt_in_text (str, optional): Text for restaurant SMS marketing opt-in.

        Returns:
            Dict[str, Any]: A dictionary containing the booking confirmation details, including the booking_reference.
        """
        db = get_db_session()
        if not db: return {"error": "Database connection error."}
        try:
            # CORRECTED: Use run_async_in_sync to call the async create_booking_with_stripe
            result = run_async_in_sync(
                create_booking_with_stripe(
                    restaurant_name=restaurant_name, VisitDate=date.fromisoformat(visit_date), VisitTime=time.fromisoformat(visit_time),
                    PartySize=party_size, ChannelCode=channel_code, SpecialRequests=special_requests,
                    IsLeaveTimeConfirmed=is_leave_time_confirmed, RoomNumber=room_number,
                    Title=customer_title, FirstName=customer_first_name, Surname=customer_surname,
                    MobileCountryCode=customer_mobile_country_code, Mobile=customer_mobile,
                    PhoneCountryCode=customer_phone_country_code, Phone=customer_phone, Email=customer_email,
                    ReceiveEmailMarketing=customer_receive_email_marketing, ReceiveSmsMarketing=customer_receive_sms_marketing,
                    GroupEmailMarketingOptInText=customer_group_email_marketing_opt_in_text,
                    GroupSmsMarketingOptInText=customer_group_sms_marketing_opt_in_text,
                    ReceiveRestaurantEmailMarketing=customer_receive_restaurant_email_marketing,
                    ReceiveRestaurantSmsMarketing=customer_receive_restaurant_sms_marketing,
                    RestaurantEmailMarketingOptInText=customer_restaurant_email_marketing_opt_in_text,
                    RestaurantSmsMarketingOptInText=customer_restaurant_sms_marketing_opt_in_text,
                    db=db, token=MOCK_BEARER_TOKEN
                )
            )
            return result
        except HTTPException as e: return {"error": e.detail, "status_code": e.status_code}
        except Exception as e: return {"error": str(e)}
        finally: db.close()

    def cancel_booking_tool(restaurant_name: str, booking_reference: str, microsite_name: str, cancellation_reason_id: int) -> Dict[str, Any]:
        """
        Cancels an existing restaurant booking using its booking reference and a specified cancellation reason.

        Args:
            restaurant_name (str): The name of the restaurant where the booking was made.
            booking_reference (str): The unique reference ID of the booking to be cancelled.
            microsite_name (str): The microsite name associated with the restaurant (often the same as restaurant_name).
            cancellation_reason_id (int): The ID of the reason for cancellation, obtained from `list_cancellation_reasons`.

        Returns:
            Dict[str, Any]: A dictionary indicating the status of the cancellation (e.g., {"status": "success"}).
        """
        db = get_db_session()
        if not db: return {"error": "Database connection error."}
        try:
            # CORRECTED: Use run_async_in_sync to call the async cancel_booking
            result = run_async_in_sync(
                cancel_booking(restaurant_name, booking_reference, microsite_name, booking_reference, cancellation_reason_id, db, MOCK_BEARER_TOKEN)
            )
            return result
        except HTTPException as e: return {"error": e.detail, "status_code": e.status_code}
        except Exception as e: return {"error": str(e)}
        finally: db.close()

    def update_booking_details_tool(
        restaurant_name: str, booking_reference: str,
        new_visit_date: Optional[str] = None, new_visit_time: Optional[str] = None, new_party_size: Optional[int] = None,
        new_special_requests: Optional[str] = None, new_is_leave_time_confirmed: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Updates specific details of an existing restaurant booking.
        At least one of the optional parameters (visit_date, visit_time, party_size, special_requests, is_leave_time_confirmed) must be provided.

        Args:
            restaurant_name (str): The name of the restaurant where the booking was made.
            booking_reference (str): The unique reference ID of the booking to be updated.
            visit_date (str, optional): The new date for the visit in ISO format (YYYY-MM-DD).
            visit_time (str, optional): The new time for the visit in ISO format (HH:MM:SS).
            party_size (int, optional): The new number of people for the reservation.
            special_requests (str, optional): Updated special requests for the booking.
            is_leave_time_confirmed (bool, optional): Updated status for whether the leave time is confirmed.

        Returns:
            Dict[str, Any]: A dictionary indicating the status of the update (e.g., {"status": "success"}).
        """
        db = get_db_session()
        if not db: return {"error": "Database connection error."}
        try:
            visit_date_obj = date.fromisoformat(new_visit_date) if new_visit_date else None
            visit_time_obj = time.fromisoformat(new_visit_time) if new_visit_time else None
            # CORRECTED: Use run_async_in_sync to call the async update_booking
            result = run_async_in_sync(
                update_booking(restaurant_name, booking_reference, visit_date_obj, visit_time_obj, new_party_size, new_special_requests, new_is_leave_time_confirmed, db, MOCK_BEARER_TOKEN)
            )
            return result
        except HTTPException as e: return {"error": e.detail, "status_code": e.status_code}
        except Exception as e: return {"error": str(e)}
        finally: db.close()

def test_data():
    db = get_db_session()
    db.execute(text("update bookings set status = 'confirmed'"))
    db.commit()

if __name__ == "__main__":
    # Example usage
    test_data()