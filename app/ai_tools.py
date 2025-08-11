"""
This module provides a unified interface for AI agents to query database tables
with support for generic chained operations and LangGraph workflows.

Author: AI Assistant
"""

from typing import Dict, List, Any, Optional, Union, Type, Callable
from dataclasses import dataclass

import logging
from datetime import date, time, datetime
from fastapi import HTTPException
from app.database import get_db, Base, Session
from app.models import Restaurant, Customer, Booking, AvailabilitySlot, CancellationReason
from app.routers.availability import availability_search, MOCK_BEARER_TOKEN
from app.routers.booking import create_booking_with_stripe, cancel_booking, get_booking, update_booking

@dataclass
class FilterCondition:
    """Represents a single filter condition"""
    column: str
    operator: str
    value: Any = None
    
    def apply_to_query(self, query, model):
        """Apply this filter condition to a SQLAlchemy query"""
        if not hasattr(model, self.column):
            raise ValueError(f"Column '{self.column}' not found in model {model.__name__}")
        
        column_attr = getattr(model, self.column)
        
        if self.operator == "eq":
            return query.filter(column_attr == self.value)
        elif self.operator == "ne":
            return query.filter(column_attr != self.value)
        elif self.operator == "lt":
            return query.filter(column_attr < self.value)
        elif self.operator == "lte":
            return query.filter(column_attr <= self.value)
        elif self.operator == "gt":
            return query.filter(column_attr > self.value)
        elif self.operator == "gte":
            return query.filter(column_attr >= self.value)
        elif self.operator == "in":
            return query.filter(column_attr.in_(self.value))
        elif self.operator == "not_in":
            return query.filter(~column_attr.in_(self.value))
        elif self.operator == "like":
            return query.filter(column_attr.like(self.value))
        elif self.operator == "ilike":
            return query.filter(column_attr.ilike(self.value))
        elif self.operator == "is_null":
            return query.filter(column_attr.is_(None))
        elif self.operator == "is_not_null":
            return query.filter(column_attr.isnot(None))
        elif self.operator == "between":
            if not isinstance(self.value, (list, tuple)) or len(self.value) != 2:
                raise ValueError("BETWEEN operator requires a list/tuple of 2 values")
            return query.filter(column_attr.between(self.value[0], self.value[1]))
        else:
            raise ValueError(f"Unsupported filter operator: {self.operator}")

@dataclass
class QueryCondition:
    """Constraints and displayed information for database queries"""
    filters: Optional[List[FilterCondition]] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    order_by: Optional[List[str]] = None  # List of column names to order by

    @staticmethod
    def create_empty_condition():
        return QueryCondition(filters=[])

@dataclass
class QueryResult:
    """Response structure for database queries."""
    success: bool
    data: List[Dict[str, Any]]
    count: int
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class DatabaseInterface:
    """Main database interface with chain query support."""
    
    # Table model mapping
    TABLE_MODELS = {
        'restaurants': Restaurant,
        'customers': Customer,
        'bookings': Booking,
        'availability_slots': AvailabilitySlot,
        'cancellation_reasons': CancellationReason
    }
    
    # Define which tables are accessible for general queries (non-private)
    PUBLIC_TABLES = {'restaurants', 'cancellation_reasons'}
    
    @classmethod
    def _serialize_value(cls, value):
        """Serialize database values for JSON compatibility."""
        if hasattr(value, 'isoformat'):  # datetime objects
            return value.isoformat()
        return value
    
    @classmethod
    def _convert_to_dict(cls, model_instance, columns: Optional[List[str]] = None):
        """Convert SQLAlchemy model instance to dictionary."""
        item_dict = {}
        for column in model_instance.__table__.columns:
            value = getattr(model_instance, column.name)
            item_dict[column.name] = cls._serialize_value(value)
        
        # Add related data for specific models
        if hasattr(model_instance, 'restaurant') and model_instance.restaurant:
            item_dict['restaurant_name'] = model_instance.restaurant.name
        if hasattr(model_instance, 'customer') and model_instance.customer:
            item_dict['customer_email'] = model_instance.customer.email
            item_dict['customer_first_name'] = model_instance.customer.first_name
            item_dict['customer_surname'] = model_instance.customer.surname
        
        # Filter columns if specified
        if columns:
            item_dict = {k: v for k, v in item_dict.items() if k in columns}
        
        return item_dict

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

# Convenience functions for common filter patterns
def create_equals_filter(column: str, value: Any) -> FilterCondition:
    """Create an equals filter condition"""
    return FilterCondition(column=column, operator="eq", value=value)

def create_in_filter(column: str, values: List[Any]) -> FilterCondition:
    """Create an IN filter condition"""
    return FilterCondition(column=column, operator="in", value=values)

def create_range_filter(column: str, lower: Any, upper: Any) -> FilterCondition:
    """Create a date range filter condition"""
    return FilterCondition(column=column, operator="between", value=[lower, upper])

def create_like_filter(column: str, pattern: str, case_sensitive: bool = True) -> FilterCondition:
    """Create a LIKE filter condition"""
    operator = "like" if case_sensitive else "ilike"
    return FilterCondition(column=column, operator=operator, value=pattern)

def create_not_equals_filter(column: str, value: Any) -> FilterCondition:
    """Create a not equals filter condition"""
    return FilterCondition(column=column, operator="ne", value=value)

def create_less_than_filter(column: str, value: Any) -> FilterCondition:
    """Create a less than filter condition"""
    return FilterCondition(column=column, operator="lt", value=value)

def create_less_than_or_equal_filter(column: str, value: Any) -> FilterCondition:
    """Create a less than or equal filter condition"""
    return FilterCondition(column=column, operator="lte", value=value)

def create_greater_than_filter(column: str, value: Any) -> FilterCondition:
    """Create a greater than filter condition"""
    return FilterCondition(column=column, operator="gt", value=value)

def create_greater_than_or_equal_filter(column: str, value: Any) -> FilterCondition:
    """Create a greater than or equal filter condition"""
    return FilterCondition(column=column, operator="gte", value=value)

def create_not_in_filter(column: str, values: List[Any]) -> FilterCondition:
    """Create a NOT IN filter condition"""
    return FilterCondition(column=column, operator="not_in", value=values)

def create_is_null_filter(column: str) -> FilterCondition:
    """Create an IS NULL filter condition"""
    return FilterCondition(column=column, operator="is_null")

def create_is_not_null_filter(column: str) -> FilterCondition:
    """Create an IS NOT NULL filter condition"""
    return FilterCondition(column=column, operator="is_not_null")

def query_table(table_name: str, query_condition: QueryCondition = None) -> QueryResult:
    """
    Generic table query function.
    
    Args:
        table_name: Name of table to query
        query_config: 
         - filters: List of FilterCondition objects
         - limit: Maximum number of records
         - offset: Number of records to skip
         - order_by: List of column names to order by
    
    Returns:
        QueryResult with data and metadata
    """
    try:
        if table_name not in DatabaseInterface.TABLE_MODELS:
            return QueryResult(
                success=False,
                data=[],
                count=0,
                error=f"Table '{table_name}' not found"
            )
        
        db = get_db_session()
        model = DatabaseInterface.TABLE_MODELS[table_name]
        query = db.query(model)
        
        if query_condition is not None:
            # Apply filters
            if query_condition.filters:
                for filter_condition in query_condition.filters:
                    query = filter_condition.apply_to_query(query, model)
            
            # Apply ordering
            if query_condition.order_by:
                for column_name in query_condition.order_by:
                    if hasattr(model, column_name):
                        query = query.order_by(getattr(model, column_name))
            
            # Apply offset
            if query_condition.offset:
                query = query.offset(query_condition.offset)
            
            # Apply limit
            if query_condition.limit:
                query = query.limit(query_condition.limit)
        
        results = query.all()
        
        # Convert to dict format
        result_data = [DatabaseInterface._convert_to_dict(item) for item in results]
        
        return QueryResult(
            success=True,
            data=result_data,
            count=len(result_data),
            metadata={
                'table': table_name,
                'filters': [f"{f.column} {f.operator} {f.value}" for f in query_condition.filters] if query_condition.filters else [],
                'total_results': len(result_data)
            }
        )
        
    except Exception as e:
        return QueryResult(
            success=False,
            data=[],
            count=0,
            error=str(e)
        )

class AIToolCallingInterface:
    '''Export all functions and API callable by LangGraph AI agents'''
    HELPER_CLASSES = [QueryCondition, FilterCondition]

    def get_customer_information(email: str) -> QueryResult:
        """Get customer information by email"""
        filter_condition = FilterCondition(
            column="email",
            operator="eq",
            value=email
        )
        
        return query_table(
            "customers",
            QueryCondition(
                filters=[filter_condition],
                limit=1
            )
        )

    def get_restaurants(query_condition: QueryCondition = None) -> QueryResult:
        """Get information of all restauarants satisfying certain constraints"""
        return query_table("restaurants", query_condition)

    def list_cancellation_reasons() -> QueryResult:
        """Get all possible cancellation reasons with ID, title, and description"""
        return query_table("cancellation_reasons")

    def customer_bookings_and_restaurants_summary(
        email: str,
        booking_conditions: Optional[QueryCondition] = None,
        restaurant_conditions: Optional[QueryCondition] = None
    ) -> Dict[str, str]:
        """
        Per-customer conditional booking reference and restaurant retrieval
        
        Args:
            email: User's email address
            booking_conditions :
            - filters: List of FilterCondition objects
            - limit: Maximum number of records
            - offset: Number of records to skip
            - order_by: List of column names to order by
            restaurant_conditions: same as booking_conditions
        
        Returns:
            Dict including
            - booking_summary: a map of booking reference to restaurant_name
            - booked_restaurants: information of all booked restaurants without repetition
        """

        # Get customer information
        customer_result = AIToolCallingInterface.get_customer_information(email)
        if not customer_result.success or not customer_result.data:
            return QueryResult(
                success=False,
                data=[],
                count=0,
                error="Customer not found"
            )
        
        # Add customer filter to booking config
        customer_id = customer_result.data[0]["id"]
        customer_filter = FilterCondition(
            column="customer_id",
            operator="eq",
            value=customer_id
        )
        if not booking_conditions:
            booking_conditions = QueryCondition.create_empty_condition()
        if not restaurant_conditions:
            restaurant_conditions = QueryCondition.create_empty_condition()

        # Get bookings for customer
        booking_conditions.filters.append(customer_filter)
        bookings = query_table("bookings", booking_conditions)
        
        if bookings.success and bookings.data:
            booking_summary = []
            restaurant_cache = {}
            for booking in bookings.data:
                restaurant_id = booking.get("restaurant_id")
                if restaurant_id not in restaurant_cache:
                    # Get restaurant data
                    restaurant_filter = FilterCondition(
                        column="id",
                        operator="eq",
                        value=restaurant_id
                    )
                    if restaurant_conditions.limit != 1:
                        restaurant_conditions.limit = 1
                    restaurant_conditions.filters.append(restaurant_filter)
                    restaurant_cache[restaurant_id] = query_table("restaurants", restaurant_conditions)
                
                restaurant = restaurant_cache[restaurant_id]
                if restaurant.success and restaurant.data:
                    booking_summary.append({"booking_reference": booking.get("booking_reference"),
                                            "restaurant_name": restaurant.data[0].get("name")})
                    
            return {
                "bookings_summary": booking_summary,
                "booked_restaurants": [restaurant.data[0] for restaurant in restaurant_cache.values() if restaurant.success and restaurant.data]
            }
        return "No booking or restaurant matches the search condition"

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

    TOOL_FUNCTIONS = {
        "SearchAvailabilityTool": search_availability, "CreateBookingTool": create_booking,
        "CancelBookingTool": cancel_booking, "GetBookingDetailsTool": get_booking_details,
        "UpdateBookingDetailsTool": update_booking_details,
        "GetCustomerBookingsAndRestaurantsSummaryTool": customer_bookings_and_restaurants_summary,
        "ListCancellationReasonsTool": list_cancellation_reasons, "GetRestaurantsTool": get_restaurants,

        "CreateEqualsFilterTool": create_equals_filter, "CreateInFilterTool": create_in_filter,
        "CreateRangeFilterTool": create_range_filter, "CreateLikeFilterTool": create_like_filter,
        "CreateNotEqualsFilterTool": create_not_equals_filter, "CreateLessThanFilterTool": create_less_than_filter,
        "CreateLessThanOrEqualFilterTool": create_less_than_or_equal_filter, "CreateGreaterThanFilterTool": create_greater_than_filter,
        "CreateGreaterThanOrEqualFilterTool": create_greater_than_or_equal_filter, "CreateNotInFilterTool": create_not_in_filter,
        "CreateIsNullFilterTool": create_is_null_filter, "CreateIsNotNullFilterTool": create_is_not_null_filter,
    }

def test_data():
    import sqlite3
    _con = sqlite3.connect("restaurant_booking.db")
    print(_con.execute("select * from bookings").fetchall())
    _con.commit()

if __name__ == "__main__":
    # Example usage
    # test_data()
    result = AIToolCallingInterface.customer_bookings_and_restaurants_summary(
        "yingyan797@restaurantai.com",
        QueryCondition([create_range_filter("party_size", 3, 9)])
    )
    
    print(result)