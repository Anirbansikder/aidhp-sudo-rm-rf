import sys
import os

# Adjust the Python path to locate 'models' and 'utils'
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.segment import Segment
from utils.db_utils import get_database

def populate_segments():
    db = get_database()
    segment_collection = db["segments"]

    # Define sample segments for Individual, Small Business, and Corporate customers
    sample_segments = [
        Segment(
            segment_name="Individual Customers",
            customer_type="Individual",
            description="Personal banking segment tailored for individual customers with diverse financial needs."
        ),
        Segment(
            segment_name="Small Business Clients",
            customer_type="Small Business",
            description="Financial solutions designed for small business owners to help grow their enterprises."
        ),
        Segment(
            segment_name="Corporate Clients",
            customer_type="Corporate",
            description="Comprehensive banking services and products for large corporations with complex financial requirements."
        )
    ]

    # Convert each segment model to a dictionary for insertion
    segment_docs = [segment.to_dict() for segment in sample_segments]

    try:
        result = segment_collection.insert_many(segment_docs)
        print(f"Inserted segment IDs: {result.inserted_ids}")
    except Exception as e:
        print(f"Error inserting segments: {e}")

if __name__ == "__main__":
    populate_segments()