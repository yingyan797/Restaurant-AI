"""
This module provides a unified interface for AI agents to query database tables
with support for generic chained operations and LangGraph workflows.

Author: AI Assistant
"""

from typing import Dict, List, Any, Optional, Union, Type, Callable
from dataclasses import dataclass
from enum import Enum

from app.database import get_db, Base
from app.models import Restaurant, Customer, Booking, AvailabilitySlot, CancellationReason

class FilterOperator(Enum):
    """Supported filter operations"""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    IN = "in"
    NOT_IN = "not_in"
    LIKE = "like"
    ILIKE = "ilike"  # case-insensitive like
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    BETWEEN = "between"

@dataclass
class FilterCondition:
    """Represents a single filter condition"""
    column: str
    operator: FilterOperator
    value: Any = None
    
    def apply_to_query(self, query, model):
        """Apply this filter condition to a SQLAlchemy query"""
        if not hasattr(model, self.column):
            raise ValueError(f"Column '{self.column}' not found in model {model.__name__}")
        
        column_attr = getattr(model, self.column)
        
        if self.operator == FilterOperator.EQUALS:
            return query.filter(column_attr == self.value)
        elif self.operator == FilterOperator.NOT_EQUALS:
            return query.filter(column_attr != self.value)
        elif self.operator == FilterOperator.LESS_THAN:
            return query.filter(column_attr < self.value)
        elif self.operator == FilterOperator.LESS_THAN_OR_EQUAL:
            return query.filter(column_attr <= self.value)
        elif self.operator == FilterOperator.GREATER_THAN:
            return query.filter(column_attr > self.value)
        elif self.operator == FilterOperator.GREATER_THAN_OR_EQUAL:
            return query.filter(column_attr >= self.value)
        elif self.operator == FilterOperator.IN:
            return query.filter(column_attr.in_(self.value))
        elif self.operator == FilterOperator.NOT_IN:
            return query.filter(~column_attr.in_(self.value))
        elif self.operator == FilterOperator.LIKE:
            return query.filter(column_attr.like(self.value))
        elif self.operator == FilterOperator.ILIKE:
            return query.filter(column_attr.ilike(self.value))
        elif self.operator == FilterOperator.IS_NULL:
            return query.filter(column_attr.is_(None))
        elif self.operator == FilterOperator.IS_NOT_NULL:
            return query.filter(column_attr.isnot(None))
        elif self.operator == FilterOperator.BETWEEN:
            if not isinstance(self.value, (list, tuple)) or len(self.value) != 2:
                raise ValueError("BETWEEN operator requires a list/tuple of 2 values")
            return query.filter(column_attr.between(self.value[0], self.value[1]))
        else:
            raise ValueError(f"Unsupported filter operator: {self.operator}")

@dataclass
class QueryConfig:
    """Constraints and displayed information for database queries"""
    filters: Optional[List[FilterCondition]] = None
    columns: Optional[List[str]] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    order_by: Optional[List[str]] = None  # List of column names to order by

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

def query_table(table_name: str, query_config: QueryConfig) -> QueryResult:
    """
    Generic table query function.
    
    Args:
        table_name: Name of table to query
        query_config: 
         - filters: List of FilterCondition objects
         - columns: Specific columns to return
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
        
        db = next(get_db())
        model = DatabaseInterface.TABLE_MODELS[table_name]
        query = db.query(model)
        
        # Apply filters
        if query_config.filters:
            for filter_condition in query_config.filters:
                query = filter_condition.apply_to_query(query, model)
        
        # Apply ordering
        if query_config.order_by:
            for column_name in query_config.order_by:
                if hasattr(model, column_name):
                    query = query.order_by(getattr(model, column_name))
        
        # Apply offset
        if query_config.offset:
            query = query.offset(query_config.offset)
        
        # Apply limit
        if query_config.limit:
            query = query.limit(query_config.limit)
        
        results = query.all()
        
        # Convert to dict format
        result_data = [DatabaseInterface._convert_to_dict(item, query_config.columns) for item in results]
        
        return QueryResult(
            success=True,
            data=result_data,
            count=len(result_data),
            metadata={
                'table': table_name,
                'filters': [f"{f.column} {f.operator.value} {f.value}" for f in query_config.filters] if query_config.filters else [],
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

def get_customer_information(email: str, columns: Optional[List[str]] = None) -> QueryResult:
    """Get customer information by email"""
    filter_condition = FilterCondition(
        column="email",
        operator=FilterOperator.EQUALS,
        value=email
    )
    
    return query_table(
        "customers",
        QueryConfig(
            filters=[filter_condition],
            columns=columns,
            limit=1
        )
    )

def check_customer_bookings_and_restaurants(
    email: str,
    booking_config: Optional[QueryConfig] = None,
    restaurant_config: Optional[QueryConfig] = None
) -> QueryResult:
    """Get customer bookings with optional restaurant information"""
    
    # Initialize configs if not provided
    if booking_config is None:
        booking_config = QueryConfig()
    if booking_config.filters is None:
        booking_config.filters = []
    if restaurant_config is not None and "restaurant_id" not in booking_config.columns:
        booking_config.columns.append("restaurant_id")
    
    # Get customer information
    customer_result = get_customer_information(email, ["id"])
    
    if not customer_result.success or not customer_result.data:
        return QueryResult(
            success=False,
            data=[],
            count=0,
            error="Customer not found"
        )
    
    customer_id = customer_result.data[0]["id"]
    
    # Add customer filter to booking config
    customer_filter = FilterCondition(
        column="customer_id",
        operator=FilterOperator.EQUALS,
        value=customer_id
    )
    booking_config.filters.append(customer_filter)
    
    # Get bookings
    bookings_result = query_table("bookings", booking_config)
    
    if not bookings_result.success:
        return bookings_result
    
    # Enrich with restaurant data if requested
    if restaurant_config is not None and bookings_result.data:
        restaurant_cache = {}
        
        for booking in bookings_result.data:
            restaurant_id = booking.get("restaurant_id")
            if restaurant_id not in restaurant_cache:
                # Get restaurant data
                restaurant_filter = FilterCondition(
                    column="id",
                    operator=FilterOperator.EQUALS,
                    value=restaurant_id
                )
                
                restaurant_query_config = QueryConfig(
                    filters=[restaurant_filter] + (restaurant_config.filters or []),
                    columns=restaurant_config.columns,
                    limit=1
                )
                
                restaurant_cache[restaurant_id] = query_table("restaurants", restaurant_query_config)
            
            restaurant = restaurant_cache[restaurant_id]
            if restaurant.success:
                for key, value in restaurant.data[0].items():
                    booking[f"restaurant_{key}"] = value
        
    return bookings_result

# Convenience functions for common filter patterns
def create_equals_filter(column: str, value: Any) -> FilterCondition:
    """Create an equals filter condition"""
    return FilterCondition(column=column, operator=FilterOperator.EQUALS, value=value)

def create_in_filter(column: str, values: List[Any]) -> FilterCondition:
    """Create an IN filter condition"""
    return FilterCondition(column=column, operator=FilterOperator.IN, value=values)

def create_date_range_filter(column: str, start_date: Any, end_date: Any) -> FilterCondition:
    """Create a date range filter condition"""
    return FilterCondition(column=column, operator=FilterOperator.BETWEEN, value=[start_date, end_date])

def create_like_filter(column: str, pattern: str, case_sensitive: bool = True) -> FilterCondition:
    """Create a LIKE filter condition"""
    operator = FilterOperator.LIKE if case_sensitive else FilterOperator.ILIKE
    return FilterCondition(column=column, operator=operator, value=pattern)

def test_data():
    import sqlite3
    _con = sqlite3.connect("restaurant_booking.db")
    _con.execute("delete from customers")
    _con.execute("delete from bookings")
    _con.commit()

if __name__ == "__main__":
    # Example usage
    # test_data()
    booking_config = QueryConfig(
        columns=[],
        limit=10
    )
    
    result = check_customer_bookings_and_restaurants(
        "yingyan797@restaurantai.com",
        booking_config=booking_config,
        restaurant_config=QueryConfig(columns=["name"])
    )
    
    print(f"Success: {result.success}")
    print(f"Count: {result.count}")
    if result.success:
        for booking in result.data:
            print(booking)
    else:
        print(f"Error: {result.error}")