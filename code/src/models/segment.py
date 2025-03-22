from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid

class Segment(BaseModel):
    segment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    segment_name: str
    customer_type: str  # Allowed values: "Individual", "Small Business", "Corporate"
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def to_dict(self):
        """Return the model as a dictionary for MongoDB insertion."""
        return self.dict()