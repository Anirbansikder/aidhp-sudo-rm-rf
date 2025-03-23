import sys
import os
import csv
import argparse

# Append the project root to sys.path so we can import from models and utils.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.product import Product
from utils.db_utils import get_database

def populate_products(csv_filepath):
    """
    Reads product data from the CSV file and inserts the product documents into the
    products collection. Each product document references a segment from the segments collection.
    """
    db = get_database()
    products_collection = db["products"]
    segments_collection = db["segments"]

    products_to_insert = []

    try:
        with open(csv_filepath, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Extract CSV fields.
                product_name = row["Product Name"]
                product_type = row["Category"]
                customer_segment = row["Customer Segment"]
                key_features = row["Key Features"]
                eligibility_criteria = row["Eligibility Criteria"]

                # Lookup the segment based on the CSV's "Customer Segment" field.
                # The segments collection should have documents with a "customer_type" field.
                segment_doc = segments_collection.find_one({"customer_type": customer_segment})
                if not segment_doc:
                    print(f"Segment not found for customer segment: {customer_segment}. Skipping product '{product_name}'.")
                    continue

                segment_id = segment_doc["segment_id"]

                # Create a Product model instance.
                product = Product(
                    product_name=product_name,
                    product_type=product_type,
                    description=key_features,
                    eligibility_criteria=eligibility_criteria,
                    segment_id=segment_id
                )

                products_to_insert.append(product.to_dict())

        if products_to_insert:
            result = products_collection.insert_many(products_to_insert)
            print(f"Inserted {len(result.inserted_ids)} products successfully.")
        else:
            print("No products to insert.")

    except FileNotFoundError:
        print(f"CSV file not found at {csv_filepath}. Please check the path and try again.")
    except Exception as e:
        print("An error occurred while populating products:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Populate the products collection from a CSV file.")
    populate_products(os.path.join(os.path.dirname(__file__), "datasets/products.csv"))