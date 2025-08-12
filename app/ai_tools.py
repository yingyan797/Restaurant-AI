"""
This module provides a unified interface for AI agents to query database tables
with support for generic chained operations and LangGraph workflows.

Author: AI Assistant
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Literal
from dataclasses import dataclass
from langchain_core.tools import tool
import logging, inspect
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

    @tool
    def list_restaurants():
        """Get information of all restauarants satisfying certain constraints"""
        db = get_db_session()
        if not db: return {"error": "Database connection error."}
        try:
            query_output = db.execute(text("select * from restaurants")).fetchall()
            restaurants = [{col.name: row[i] for i, col in enumerate(Restaurant.__table__.columns)} for row in query_output]
            return restaurants

        except HTTPException as e: return {"error": e.detail, "status_code": e.status_code}
        except Exception as e: return {"error": str(e)}
        finally: db.close()

    @tool
    def list_cancellation_reasons():
        """Get all possible cancellation reasons with ID, title, and description"""
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

    @tool
    def find_customer_bookings(
            email: str,
            date_range: Tuple[str, str] = None,
            time_range: Tuple[str, str] = None,
            restaurant_name: str = None,
            party_size_range: Tuple[int, int] = None,
            status: Literal['cancelled', 'confirmed', 'completed'] = None,
            is_leave_time_confirmed: bool = None,
            room_number: str = None
        ) -> List[Dict[str, Any]]:
        """
        Per-customer conditional booking reference and restaurant retrieval
        
        Args:
            email: User's email address
            ...: Filter bookings and restaurants
        
        Returns:
            bookings: booking reference and restaurant_name
        """
        # Get customer information
        db = get_db_session()
        if not db: return {"error": "Database connection error."}
        try:
            customer_id = AIToolCallingInterface.get_customer_information(email)["id"]
            statement = ("select booking_reference, restaurants.name as restaurant_name "
                        "from bookings left join restaurants on bookings.restaurant_id = restaurants.id "
                        f"where customer_id={customer_id}")
            frame = inspect.currentframe()
            args_info = inspect.signature(AIToolCallingInterface.find_customer_bookings).parameters
            values = frame.f_locals  # dictionary of local vars

            for name in args_info.keys():  # only iterate over parameter names
                value = values[name]
                if name != "email" and value is not None:

                    def put_value(v):
                        if isinstance(v, str):
                            return f"'{v}'"
                        return str(v)

                    if name.endswith("_range") and any(v is not None for v in value):
                        if value[1] is None:
                            statement += f" and {name[:-6]} >= {put_value(value[0])}"
                        elif value[0] is None:
                            statement += f" and {name[:-6]} <= {put_value(value[0])}"
                        else:
                            statement += f" and {name[:-6]} between {put_value(value[0])} and {put_value(value[1])}"
                    else:
                        statement += f" and {name}='{value}'"
            bookings = db.execute(text(statement)).fetchall()
            return [get_booking(booking[1], booking[0], db, MOCK_BEARER_TOKEN) for booking in bookings]
            
        except HTTPException as e: return {"error": e.detail, "status_code": e.status_code}
        except Exception as e: return {"error": str(e)}
        finally: db.close()

    @tool
    def search_availability(restaurant_name: str, VisitDate: str, PartySize: int, ChannelCode: str = "ONLINE") -> Dict[str, Any]:
        """Wrapper for availability_search API."""
        db = get_db_session()
        if not db: return {"error": "Database connection error."}
        try:
            result = availability_search(restaurant_name, date.fromisoformat(VisitDate), PartySize, ChannelCode, db, MOCK_BEARER_TOKEN)
            return result
        except HTTPException as e: return {"error": e.detail, "status_code": e.status_code}
        except Exception as e: return {"error": str(e)}
        finally: db.close()

    @tool
    def create_booking(
        restaurant_name: str, VisitDate: str, VisitTime: str, PartySize: int,
        customer_email: str, customer_first_name: str, customer_surname: str, customer_mobile: str,
        ChannelCode: str = "ONLINE", SpecialRequests: Optional[str] = None,
        IsLeaveTimeConfirmed: Optional[bool] = False, RoomNumber: Optional[str] = None,
        customer_title: Optional[str] = None, customer_mobile_country_code: Optional[str] = None,
        customer_phone_country_code: Optional[str] = None, customer_phone: Optional[str] = None,
        customer_receive_email_marketing: Optional[bool] = False, customer_receive_sms_marketing: Optional[bool] = False,
        customer_group_email_marketing_opt_in_text: Optional[str] = None, customer_group_sms_marketing_opt_in_text: Optional[str] = None,
        customer_receive_restaurant_email_marketing: Optional[bool] = False, customer_receive_restaurant_sms_marketing: Optional[bool] = False,
        customer_restaurant_email_marketing_opt_in_text: Optional[str] = None, customer_restaurant_sms_marketing_opt_in_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Wrapper for create_booking_with_stripe API."""
        db = get_db_session()
        if not db: return {"error": "Database connection error."}
        try:
            result = create_booking_with_stripe(
                restaurant_name=restaurant_name, VisitDate=date.fromisoformat(VisitDate), VisitTime=time.fromisoformat(VisitTime),
                PartySize=PartySize, ChannelCode=ChannelCode, SpecialRequests=SpecialRequests,
                IsLeaveTimeConfirmed=IsLeaveTimeConfirmed, RoomNumber=RoomNumber,
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
            return result
        except HTTPException as e: return {"error": e.detail, "status_code": e.status_code}
        except Exception as e: return {"error": str(e)}
        finally: db.close()

    @tool
    def cancel_booking(restaurant_name: str, booking_reference: str, micrositeName: str, cancellationReasonId: int) -> Dict[str, Any]:
        """Wrapper for cancel_booking API."""
        db = get_db_session()
        if not db: return {"error": "Database connection error."}
        try:
            result = cancel_booking(restaurant_name, booking_reference, micrositeName, booking_reference, cancellationReasonId, db, MOCK_BEARER_TOKEN)
            return result
        except HTTPException as e: return {"error": e.detail, "status_code": e.status_code}
        except Exception as e: return {"error": str(e)}
        finally: db.close()

    @tool
    def update_booking_details(
        restaurant_name: str, booking_reference: str,
        VisitDate: Optional[str] = None, VisitTime: Optional[str] = None, PartySize: Optional[int] = None,
        SpecialRequests: Optional[str] = None, IsLeaveTimeConfirmed: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Wrapper for update_booking API."""
        db = get_db_session()
        if not db: return {"error": "Database connection error."}
        try:
            visit_date_obj = date.fromisoformat(VisitDate) if VisitDate else None
            visit_time_obj = time.fromisoformat(VisitTime) if VisitTime else None
            result = update_booking(restaurant_name, booking_reference, visit_date_obj, visit_time_obj, PartySize, SpecialRequests, IsLeaveTimeConfirmed, db, MOCK_BEARER_TOKEN)
            return result
        except HTTPException as e: return {"error": e.detail, "status_code": e.status_code}
        except Exception as e: return {"error": str(e)}
        finally: db.close()

def test_data():
    for booking in (AIToolCallingInterface.find_customer_bookings("yingyan797@restaurantai.com",
                                                        restaurant_name="TheHungryUnicorn")):
        print(booking)    

if __name__ == "__main__":
    # Example usage
    test_data()
    
