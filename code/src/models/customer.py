from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

class Customer(BaseModel):
    # Basic Customer Info
    customer_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    customer_name: str
    customer_type: str  # Allowed values: "Individual", "Small Business", "Corporate"
    segment_id: str     # Reference to the segment (e.g., Individual, Small Business, Corporate)
    email: EmailStr
    phone_number: str
    
    # Merged Profile Details
    annual_income: float
    credit_score: int
    interests: List[str] = []  # e.g., ["technology", "sports", "finance"]
    
    # Aggregated available balance from all products/accounts
    available_balance: float = 0.0
    
    # Reference to one or more products the customer is using
    product_ids: List[str] = []  # List of product IDs (as strings)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def to_dict(self):
        """Convert the Customer model to a dictionary for MongoDB insertion."""
        return self.dict()
