"""
Generic Database Interface for AI Agents.

This module provides a unified interface for AI agents to query database tables
with support for LangGraph workflows and agent frameworks.

Author: AI Assistant
"""

from typing import Dict, List, Any, Optional, Union, Type
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import text, inspect
from sqlalchemy.exc import SQLAlchemyError
import json

from app.database import get_db, Base
from app.models import Restaurant, Customer, Booking, AvailabilitySlot, CancellationReason


@dataclass
class QueryRequest:
    """Request structure for database queries."""
    table_name: str
    columns: Optional[List[str]] = None  # None = all columns
    conditions: Optional[Dict[str, Any]] = None  # WHERE conditions
    limit: Optional[int] = None
    order_by: Optional[str] = None


@dataclass
class QueryResult:
    """Response structure for database queries."""
    success: bool
    data: List[Dict[str, Any]]
    count: int
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DatabaseInterface:
    """Generic database interface for AI agents."""
    
    # Map table names to SQLAlchemy models
    TABLE_MODELS = {
        'restaurants': Restaurant,
        'customers': Customer,
        'bookings': Booking,
        'availability_slots': AvailabilitySlot,
        'cancellation_reasons': CancellationReason
    }
    
    def __init__(self, db_session: Optional[Session] = None):
        """Initialize with optional database session."""
        self._db_session = db_session
    
    def _get_session(self) -> Session:
        """Get database session."""
        if self._db_session:
            return self._db_session
        return next(get_db())
    
    def query_table(self, request: QueryRequest) -> QueryResult:
        """
        Execute generic table query based on request parameters.
        
        Args:
            request: QueryRequest with table name, columns, conditions, etc.
            
        Returns:
            QueryResult with data and metadata
        """
        try:
            db = self._get_session()
                
            # Validate table exists
            if request.table_name not in self.TABLE_MODELS:
                return QueryResult(
                    success=False,
                    data=[],
                    count=0,
                    error=f"Table '{request.table_name}' not found"
                )
            
            model = self.TABLE_MODELS[request.table_name]
            query = db.query(model)
            
            # Apply WHERE conditions
            if request.conditions:
                for column, value in request.conditions.items():
                    if hasattr(model, column):
                        query = query.filter(getattr(model, column) == value)
            
            # Apply ordering
            if request.order_by and hasattr(model, request.order_by):
                query = query.order_by(getattr(model, request.order_by))
            
            # Apply limit
            if request.limit:
                query = query.limit(request.limit)
            
            # Execute query
            results = query.all()
            
            # Convert to dictionaries
            data = []
            for result in results:
                row_dict = {}
                for column in (request.columns or self._get_table_columns(request.table_name)):
                    if hasattr(result, column):
                        value = getattr(result, column)
                        # Handle datetime serialization
                        if hasattr(value, 'isoformat'):
                            value = value.isoformat()
                        row_dict[column] = value
                data.append(row_dict)
            
            return QueryResult(
                success=True,
                data=data,
                count=len(data),
                metadata={
                    'table': request.table_name,
                    'total_columns': len(self._get_table_columns(request.table_name))
                }
            )
            
        except SQLAlchemyError as esql:
            return QueryResult(
                success=False,
                data=[],
                count=0,
                error=f"Database error: {str(esql)}"
            )
        except Exception as e:
            return QueryResult(
                success=False,
                data=[],
                count=0,
                error=f"Unexpected error: {str(e)}"
            )
    
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get table schema information for AI agents."""
        if table_name not in self.TABLE_MODELS:
            return {"error": f"Table '{table_name}' not found"}
        
        model = self.TABLE_MODELS[table_name]
        columns = self._get_table_columns(table_name)
        
        return {
            "table_name": table_name,
            "columns": columns,
            "model_class": model.__name__,
            "relationships": self._get_relationships(model)
        }
    
    def _get_table_columns(self, table_name: str) -> List[str]:
        """Get column names for a table."""
        if table_name not in self.TABLE_MODELS:
            return []
        
        model = self.TABLE_MODELS[table_name]
        return [column.name for column in model.__table__.columns]
    
    def _get_relationships(self, model: Type[Base]) -> List[str]:
        """Get relationship names for a model."""
        return [rel.key for rel in inspect(model).relationships]


# LangGraph Tool Functions
def create_database_query_tool():
    """Create a database query tool for LangGraph workflows."""
    
    def database_query_tool(
        table_name: str,
        columns: Optional[str] = None,
        conditions: Optional[str] = None,
        limit: Optional[int] = 10,
        order_by: Optional[str] = None
    ) -> str:
        """
        Query database table with specified parameters.
        
        Args:
            table_name: Name of the table to query
            columns: Comma-separated column names (optional, defaults to all)
            conditions: JSON string of WHERE conditions (optional)
            limit: Maximum number of rows to return
            order_by: Column name to order by (optional)
            
        Returns:
            JSON string with query results
        """
        try:
            # Parse inputs
            column_list = [col.strip() for col in columns.split(',')] if columns else None
            condition_dict = json.loads(conditions) if conditions else None
            
            # Create request
            request = QueryRequest(
                table_name=table_name,
                columns=column_list,
                conditions=condition_dict,
                limit=limit,
                order_by=order_by
            )
            
            # Execute query
            db_interface = DatabaseInterface()
            result = db_interface.query_table(request)
            
            return json.dumps({
                "success": result.success,
                "data": result.data,
                "count": result.count,
                "error": result.error,
                "metadata": result.metadata
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "success": False,
                "data": [],
                "count": 0,
                "error": f"Tool error: {str(e)}"
            })
    
    return database_query_tool


def create_schema_info_tool():
    """Create a schema information tool for LangGraph workflows."""
    
    def schema_info_tool(table_name: Optional[str] = None) -> str:
        """
        Get database schema information.
        
        Args:
            table_name: Specific table name (optional, returns all if None)
            
        Returns:
            JSON string with schema information
        """
        try:
            db_interface = DatabaseInterface()
            
            if table_name:
                schema = db_interface.get_table_schema(table_name)
                return json.dumps(schema, indent=2)
            else:
                # Return all table schemas
                all_schemas = {}
                for table in db_interface.TABLE_MODELS.keys():
                    all_schemas[table] = db_interface.get_table_schema(table)
                return json.dumps(all_schemas, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Schema tool error: {str(e)}"
            })
    
    return schema_info_tool


# Agent Helper Functions
class AgentDatabaseHelper:
    """Helper class for AI agents to interact with database."""
    
    def __init__(self):
        self.db_interface = DatabaseInterface()
    
    def search_bookings(self, **filters) -> List[Dict[str, Any]]:
        """Search bookings with flexible filters."""
        request = QueryRequest(
            table_name='bookings',
            conditions=filters,
            limit=50
        )
        result = self.db_interface.query_table(request)
        return result.data if result.success else []
    
    def get_customer_bookings(self, email: str) -> List[Dict[str, Any]]:
        """Get all bookings for a customer by email."""
        # First get customer ID
        customer_request = QueryRequest(
            table_name='customers',
            conditions={'email': email},
            columns=['id']
        )
        customer_result = self.db_interface.query_table(customer_request)
        
        if not customer_result.success or not customer_result.data:
            return []
        
        customer_id = customer_result.data[0]['id']
        
        # Get bookings
        booking_request = QueryRequest(
            table_name='bookings',
            conditions={'customer_id': customer_id},
            order_by='visit_date'
        )
        booking_result = self.db_interface.query_table(booking_request)
        return booking_result.data if booking_result.success else []


# Export tools for LangGraph integration
DATABASE_TOOLS = {
    "database_query": create_database_query_tool(),
    "schema_info": create_schema_info_tool()
}

# Export helper for agents
AGENT_DB_HELPER = AgentDatabaseHelper()