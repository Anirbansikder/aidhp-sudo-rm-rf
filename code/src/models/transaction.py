from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime
import uuid

class Transaction(BaseModel):
    # Unique Transaction Identifier
    transaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Customer Reference
    customer_id: str  # Foreign Key reference to Customer collection
    
    # Transaction Details
    transaction_date: datetime = Field(default_factory=datetime.utcnow)
    transaction_type: Literal["Debit", "Credit"]  # Enum for transaction type
    amount: float  # Transaction amount
    merchant_category: str  # e.g., "Retail", "Travel", "Insurance"
    description: str  # Transaction details
    
    # Balance & Processing
    balance_after_transaction: float
    is_processed_for_recommendation: bool = False  # Default to False
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def to_dict(self):
        """Convert the Transaction model to a dictionary for MongoDB insertion."""
        return self.dict()
