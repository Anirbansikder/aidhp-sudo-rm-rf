import sys
import os
import csv
import argparse

# Append project root to sys.path so we can import from models and utils.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.customer import Customer
from utils.db_utils import get_database

def parse_interests(interests_str):
    """
    Split the interests string by comma and trim whitespace.
    """
    return [interest.strip() for interest in interests_str.split(",") if interest.strip()]

def fetch_segment_id(db, customer_type):
    """
    Lookup the segment document in the segments collection based on the customer_type.
    Assumes segments have been populated with a 'customer_type' field.
    """
    segment_doc = db["segments"].find_one({"customer_type": customer_type})
    if not segment_doc:
        print(f"Warning: No segment found for customer_type: {customer_type}")
        return None
    return segment_doc["segment_id"]

def fetch_product_ids(db, products_using_str):
    """
    Split the products_using string by semicolon, lookup each product by its name in the products collection,
    and return a list of product_ids.
    """
    product_ids = []
    product_names = [p.strip() for p in products_using_str.split(";") if p.strip()]
    
    for name in product_names:
        product_doc = db["products"].find_one({"product_name": name})
        if product_doc:
            product_ids.append(product_doc["product_id"])
        else:
            print(f"Warning: No product found with product_name: {name}")
    return product_ids

def populate_customers(csv_filepath):
    db = get_database()
    customers_collection = db["customers"]

    customers_to_insert = []

    try:
        with open(csv_filepath, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Lookup the segment_id based on the customer_type from the CSV.
                customer_type = row["customer_type"]
                segment_id = fetch_segment_id(db, customer_type)
                if segment_id is None:
                    print(f"Skipping customer {row['customer_name']} due to missing segment.")
                    continue

                # Lookup product_ids based on the 'products_using' field.
                product_ids = fetch_product_ids(db, row["products_using"])

                # Parse interests (split on comma)
                interests = parse_interests(row["interests"])

                try:
                    # Create the Customer document using the combined schema.
                    customer = Customer(
                        customer_id=row["customer_id"],
                        customer_name=row["customer_name"],
                        customer_type=customer_type,
                        segment_id=segment_id,
                        email=row["email"],
                        phone_number=row["phone_number"],
                        annual_income=float(row["annual_income"]),
                        credit_score=int(row["credit_score"]),
                        interests=interests,
                        available_balance=float(row["available_balance"]),
                        product_ids=product_ids
                    )
                except Exception as e:
                    print(f"Error processing customer {row['customer_name']}: {e}")
                    continue

                customers_to_insert.append(customer.to_dict())

        if customers_to_insert:
            result = customers_collection.insert_many(customers_to_insert)
            print(f"Inserted {len(result.inserted_ids)} customers successfully.")
        else:
            print("No customers to insert.")

    except FileNotFoundError:
        print(f"CSV file not found at {csv_filepath}. Please check the path and try again.")
    except Exception as e:
        print("An error occurred while populating customers:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Populate the customers collection from a CSV file.")
    populate_customers(os.path.join(os.path.dirname(__file__), "datasets/customers.csv"))
