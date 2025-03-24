# src/services/transaction_service.py

import json
from datetime import datetime
from pymongo import MongoClient
from utils.db_utils import get_database
from utils.openai_util import get_openai_client

def fetch_transactions_by_date(date_str: str):
    """
    Fetch ALL transactions for a given date (ignoring is_processed_for_recommendation).
    :param date_str: in format 'MM/DD/YYYY' or 'YYYY-MM-DD' (depending on your approach)
    """
    db = get_database()
    transactions_coll = db["transactions"]
    
    date_obj = datetime.strptime(date_str, "%m/%d/%Y")
    start_of_day = datetime(date_obj.year, date_obj.month, date_obj.day, 0, 0, 0)
    end_of_day   = datetime(date_obj.year, date_obj.month, date_obj.day, 23, 59, 59)
    
    query = {
      "transaction_date": {
          "$gte": start_of_day,
          "$lte": end_of_day
      },
      "is_processed_for_recommendation": False
    }

    transactions = list(transactions_coll.find(query))
    for tx in transactions:
        tx["_id"] = str(tx["_id"])  # Convert ObjectID to string if needed
    return transactions

def clean_completion_text(text: str) -> str:
    """
    Clean the text returned by the LLM by stripping markdown code block formatting,
    such as triple backticks and any language hints.
    """
    text = text.strip()
    if text.startswith("```"):
        # Remove the first line (```json or similar) and the last line (```)
        lines = text.splitlines()
        # Remove the first and last lines if they are the triple backticks
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text

def get_recommended_transaction_by_date(date_str: str):
    """
    1) Fetch all transactions by specified date with is_processed_for_recommendation = false.
    2) Build an intelligent prompt to choose ONE transaction and recommend a product.
    3) Call the LLM with chat completions using openai_util, parse JSON response.
    4) Return the chosen transaction_id.
    """
    db = get_database()
    transactions_coll = db["transactions"]

    # Prepare the query for unprocessed transactions on given date
    date_obj = datetime.strptime(date_str, "%m/%d/%Y")
    start_of_day = datetime(date_obj.year, date_obj.month, date_obj.day, 0, 0, 0)
    end_of_day   = datetime(date_obj.year, date_obj.month, date_obj.day, 23, 59, 59)
    
    query = {
      "transaction_date": {
          "$gte": start_of_day,
          "$lte": end_of_day
      },
      "is_processed_for_recommendation": False
    }

    unprocessed_txs = list(transactions_coll.find(query))

    if not unprocessed_txs:
        return {
            "message": "No unprocessed transactions found for this date",
            "date": date_str
        }

    # Build a prompt context from the unprocessed transactions
    tx_descriptions = []
    for tx in unprocessed_txs:
        tx_descriptions.append(
            f"TransactionID: {tx['transaction_id']}, "
            f"Type: {tx['transaction_type']}, "
            f"Balance After Transaction: {tx['balance_after_transaction']}"
            f"Amount: {tx['amount']}, "
            f"Category: {tx['merchant_category']}, "
            f"Desc: {tx['description']}"
        )

    prompt_context = "\n".join(tx_descriptions)

    # Construct a JSON instruction for the LLM
    system_instructions = (
        "You are a Wells Fargo product recommendation system. "
        "Given a list of transactions, pick transactions (transaction_id) "
        "for which a Wells Fargo product can be recommended. "
        "Output a list of JSON objects with the format:\n"
        "[ {\n"
        '  "transaction_id": "<the chosen transaction id>",\n'
        '  "category": "<the chosen transaction category>",\n'
        '  "description": "<the chosen transaction description>",\n'
        '  "type": "<the chosen transaction type>",\n'
        '  "reason": "<short reason why this product suits the transaction>"\n'
        "} ]"
    )

    user_message = f"Transactions:\n{prompt_context}\nWhich transactions do you pick?"

    # Get the configured openai client
    openai_client = get_openai_client()

    # Make the ChatCompletion call
    try:
        response = openai_client.chat.completions.create(
            model="deepseek-reasoner",
            temperature=0.7,
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_message}
            ]
        )
    except Exception as e:
        return {"error": f"OpenAI API call failed: {e}"}

    # Extract response text (should be JSON)
    completion_text = response.choices[0].message.content.strip()
    completion_text = clean_completion_text(completion_text)

    try:
        llm_json = json.loads(completion_text)
    except json.JSONDecodeError:
        return {"error": "Failed to parse LLM response as JSON.", "raw_response": completion_text}

    # Return the parsed response
    return llm_json
