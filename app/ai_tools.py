"""
This module provides a unified interface for AI agents to query database tables
with support for generic chained operations and LangGraph workflows.

Author: AI Assistant
"""

from typing import Dict, List, Any, Optional, Union, Type, Callable
from dataclasses import dataclass

import logging
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

# Set up logging for better visibility during execution
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper for Database Session Management ---
def get_db_session() -> Optional[Session]:
    """Helper to get a database session. Used by tool wrappers."""
    try:
        db = next(get_db())
        return db
    except Exception as e:
        logger.error(f"Error getting DB session: {e}")
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

    def find_restaurants(sql_condition: str):
        """Get information of all restauarants satisfying certain constraints"""
        db = get_db_session()
        if not db: return {"error": "Database connection error."}
        try:
            query_output = db.execute(text("select * from restaurants "+sql_condition)).fetchall()
            restaurants = [{"restaurant_"+col.name: row[i] for i, col in enumerate(Restaurant.__table__.columns)} for row in query_output]
            return restaurants

        except HTTPException as e: return {"error": e.detail, "status_code": e.status_code}
        except Exception as e: return {"error": str(e)}
        finally: db.close()

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

    def customer_bookings_and_restaurants_summary(email: str, sql_conditions:list[str]) -> Dict[str, str]:
        """
        Per-customer conditional booking reference and restaurant retrieval
        
        Args:
            email: User's email address
            booking_conditions: SQL conditions for bookings table, 
            restaurant_conditions: SQL conditions for restaurants table 
        
        Returns:
            - booking_summary: a map from booking reference to restaurant_name and booking details
        """

        # Get customer information
        db = get_db_session()
        if not db: return {"error": "Database connection error."}
        try:
            customer_id = AIToolCallingInterface.get_customer_information(email)["id"]
            statement = ("select booking_reference, restaurants.name as restaurant_name, visit_date, visit_time, party_size "
                        "from bookings left join restaurants on bookings.restaurant_id = restaurants.id "
                        f"where customer_id={customer_id} ")
            for condition in sql_conditions:
                statement += f"and {condition} "
            columns = ['booking_reference', 'restaurant_name', 'visit_date', 'visit_time', 'party_size']
            query_output = db.execute(text(statement)).fetchall()
            return [{col: row[i] for i, col in enumerate(columns)} for row in query_output]


        except HTTPException as e: return {"error": e.detail, "status_code": e.status_code}
        except Exception as e: return {"error": str(e)}
        finally: db.close()

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

    def get_booking_details(restaurant_name: str, booking_reference: str) -> Dict[str, Any]:
        """Wrapper for get_booking API."""
        db = get_db_session()
        if not db: return {"error": "Database connection error."}
        try:
            result = get_booking(restaurant_name, booking_reference, db, MOCK_BEARER_TOKEN)
            return result
        except HTTPException as e: return {"error": e.detail, "status_code": e.status_code}
        except Exception as e: return {"error": str(e)}
        finally: db.close()

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
    db = get_db_session()
    # print(AIToolCallingInterface.find_restaurants(""))
    print(AIToolCallingInterface.customer_bookings_and_restaurants_summary(
        "yingyan797@restaurantai.com", ["party_size > 3"]))

if __name__ == "__main__":
    # Example usage
    test_data()
    
