import sys
import os
import csv
import argparse
from datetime import datetime

# Append project root to sys.path to allow imports from models and utils
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.transaction import Transaction
from utils.db_utils import get_database

def parse_transaction_date(date_str: str) -> datetime:
    """
    Parse the date string from the CSV (format: "MM/DD/YYYY") to a datetime object.
    """
    try:
        return datetime.strptime(date_str, "%m/%d/%Y")
    except Exception as e:
        print(f"Error parsing date '{date_str}': {e}")
        raise

def populate_transactions(csv_filepath="transactions.csv"):
    db = get_database()
    transactions_collection = db["transactions"]
    
    transactions_to_insert = []
    
    try:
        with open(csv_filepath, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Parse and convert CSV values
                customer_id = row["customer_id"]
                # Parse transaction_date using the given format
                transaction_date = parse_transaction_date(row["transaction_date"])
                transaction_type = row["transaction_type"]  # Expected "Debit" or "Credit"
                try:
                    amount = float(row["amount"])
                except Exception as e:
                    print(f"Error converting amount '{row['amount']}' for customer {customer_id}: {e}")
                    continue
                merchant_category = row["merchant_category"]
                description = row["description"]
                try:
                    balance_after_transaction = float(row["balance_after_transaction"])
                except Exception as e:
                    print(f"Error converting balance_after_transaction '{row['balance_after_transaction']}' for customer {customer_id}: {e}")
                    continue
                
                # Create a Transaction instance.
                transaction = Transaction(
                    customer_id=customer_id,
                    transaction_date=transaction_date,
                    transaction_type=transaction_type,
                    amount=amount,
                    merchant_category=merchant_category,
                    description=description,
                    balance_after_transaction=balance_after_transaction,
                    is_processed_for_recommendation=False,
                    created_at=transaction_date,
                    updated_at=transaction_date
                )
                
                transactions_to_insert.append(transaction.to_dict())
        
        if transactions_to_insert:
            result = transactions_collection.insert_many(transactions_to_insert)
            print(f"Inserted {len(result.inserted_ids)} transactions successfully.")
        else:
            print("No transactions to insert.")
    
    except FileNotFoundError:
        print(f"CSV file not found at {csv_filepath}. Please check the path and try again.")
    except Exception as e:
        print("An error occurred while populating transactions:", e)

if __name__ == "__main__":
    populate_transactions(os.path.join(os.path.dirname(__file__), "datasets/transactions3.csv"))