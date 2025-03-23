from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid

class Product(BaseModel):
    product_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    product_name: str
    product_type: str  # E.g., "Loan", "Credit Card", "Savings Account", etc.
    description: Optional[str] = None
    eligibility_criteria: Optional[str] = None
    segment_id: str  # Reference to the associated segment (Segment.segment_id)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def to_dict(self):
        """Convert the Product model to a dictionary for MongoDB insertion."""
        return self.dict()