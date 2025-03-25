# src/services/transaction_service.py

import json
from datetime import datetime
from pymongo import MongoClient
from utils.db_utils import get_database
from utils.openai_util import get_openai_client
from datetime import datetime, timedelta, timezone
from bson import ObjectId

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

def analyze_recommendable_transaction_by_date(date_str: str):
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

    unprocessed_txs = list(transactions_coll.find(query))

    if not unprocessed_txs:
        return {
            "message": "No unprocessed transactions found for this date",
            "date": date_str
        }
    
    tx_descriptions = []
    for tx in unprocessed_txs:
        tx_descriptions.append(
            f"TransactionID: {tx['transaction_id']}, "
            f"Transaction Type: {tx['transaction_type']}, "
            f"Balance After Transaction: {tx['balance_after_transaction']}"
            f"Amount: {tx['amount']}, "
            f"Merchant Category: {tx['merchant_category']}, "
            f"Description: {tx['description']}"
        )

    prompt_context = "\n".join(tx_descriptions)

    system_prompt = (
        "You are an AI assistant specializing in financial product recommendations for bank customers. "
        "Your task is to analyze a list of recent transactions and determine which transactions are suitable "
        "for a personalized recommendation. Consider factors such as merchant category, transaction amount, "
        "available balance, transaction type, and description. Select only transactions that indicate potential "
        "interest in relevant banking products (e.g., travel transactions may suggest interest in travel insurance, "
        "large retail purchases may indicate interest in a credit limit increase). "
        "Output a object containing a list of valid transactions strictly maintaining below format:\n"
        "{\"valid_transactions\": [\n"
        "    {\n"
        "      \"transaction_id\": \"<valid transaction id>\",\n"
        "      \"reason\": \"<brief reason why this transaction is suitable for recommendation>\"\n"
        "    }\n"
        "  ]\n"
        "}"
    )

    user_message = f"Transactions:\n{prompt_context}\nWhich transaction do you pick?"

    openai_client = get_openai_client()

    try:
        response = openai_client.chat.completions.create(
            model="deepseek-reasoner",
            temperature=0.7,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
    except Exception as e:
        return {"error": f"OpenAI API call failed: {e}"}
    
    completion_text = response.choices[0].message.content.strip()
    completion_text = clean_completion_text(completion_text)

    try:
        llm_json = json.loads(completion_text)
    except json.JSONDecodeError:
        return {"error": "Failed to parse LLM response as JSON.", "raw_response": completion_text}

    valid_transactions = llm_json.get("valid_transactions") or []

    # Update processed transactions in the database
    transaction_ids = [tx["transaction_id"] for tx in valid_transactions]
    transactions_coll.update_many(
        {"transaction_id": {"$in": transaction_ids}},
        {"$set": {"is_processed_for_recommendation": True}}
    )

    return valid_transactions

def analyze_recommendable_products_for_customer(customer_id: str, start_date: str, end_date: str):
    db = get_database()
    transactions_coll = db["transactions"]
    customers_coll = db["customers"]
    products_coll = db["products"]

    # Find the customer to get the segment_id
    customer = customers_coll.find_one({"customer_id": customer_id})
    if not customer:
        return {"error": "Customer not found"}

    segment_id = customer.get("segment_id")
    if not segment_id:
        return {"error": "Segment ID not found for customer"}

    try:
        start_date = datetime.strptime(start_date, "%m/%d/%Y")
        end_date = datetime.strptime(end_date, "%m/%d/%Y")
    except ValueError:
        return {"error": "Invalid date format. Use MM/DD/YYYY."}

    valid_transactions = transactions_coll.find({
        "transaction_date": {"$gte": start_date, "$lte": end_date},
        "customer_id": customer_id,
        "is_processed_for_recommendation": True      # Only processed transactions
    })
    valid_transactions = [{**tx, "_id": str(tx["_id"])} for tx in valid_transactions]

    eligible_products = products_coll.find({"segment_id": segment_id})
    eligible_products = [{**product, "_id": str(product["_id"])} for product in eligible_products]

    customer_product_ids = customer.get("product_ids")
    subtracted_eligible_rpoducts = [
        product for product in eligible_products if product["product_id"] not in customer_product_ids
    ]

    tx_descriptions = []
    for tx in valid_transactions:
        tx_descriptions.append(
            f"TransactionID: {tx['transaction_id']}, "
            f"Transaction Type: {tx['transaction_type']}, "
            f"Balance After Transaction: {tx['balance_after_transaction']}"
            f"Amount: {tx['amount']}, "
            f"Merchant Category: {tx['merchant_category']}, "
            f"Description: {tx['description']}"
        )
    tx_prompt_context = "\n".join(tx_descriptions)

    tx_descriptions = []
    for tx in valid_transactions:
        tx_descriptions.append(
            f"TransactionID: {tx['transaction_id']}, "
            f"Transaction Type: {tx['transaction_type']}, "
            f"Balance After Transaction: {tx['balance_after_transaction']}"
            f"Amount: {tx['amount']}, "
            f"Merchant Category: {tx['merchant_category']}, "
            f"Description: {tx['description']}"
        )
    tx_prompt_context = "\n".join(tx_descriptions)
    user_message = f"Transactions:\n{tx_prompt_context}\nChoose the most eligible product recommended for the transactions and rank them in order"

    pd_descriptions = []
    for pd in subtracted_eligible_rpoducts:
        pd_descriptions.append(
            f"product_id: {pd['product_id']}, "
            f"Product Name: {pd['product_name']}, "
            f"Product Type: {pd['product_type']}, "
            f"Product Description: {pd['description']}"
            f"Product Eligibility Criteria: {pd['eligibility_criteria']}"
        )
    pd_prompt_context = "\n".join(pd_descriptions)

    customer_interests = " ".join(list(customer.get('interests')))

    system_prompt = (
        "You are a financial AI assistant specializing in recommending personalized banking products based on customer transactions"
        "Your task is to analyze a list of valid transactions and match them with eligible financial products based on the customer's segment."
        "Consider the following key factors while making recommendations:"
        "Transaction Type: Identify patterns such as large purchases, frequent travel expenses, or recurring business transactions."
        "Merchant Category: Recognize spending behaviors that align with specific banking products (e.g., real estate-related payments may indicate interest in commercial real estate financing)."
        "Transaction Amount & Balance: Suggest products that match the customer's financial activity and ensure affordability."
        "Segment-Based Eligibility: Only recommend products that belong to the customer's designated segment."
        "Customer Interest: Take into account any explicit product interests the customer has shown in past interactions, applications, or inquiries."
        "Priority Ranking: Assign a priority to each recommended product based on how well it matches the transaction. A lower number indicates a higher priority (1 = best match)."
        "\n\n Here are the **customer interests**" + "\n" + customer_interests + "\n"
        "Here are the **eligible financial products**" + "\n" + pd_prompt_context + "\n"
        "Output a object containing a list of valid transactions strictly maintaining below format:\n"
        "{\"valid_products\": [\n"
        "    {\n"
        "      \"product_id\": \"<valid product id>\",\n"
        "      \"product_name\": \"<valid product name>\",\n"
        "      \"reason\": \"<descriptive reason why this product is suitable for customer based on transactions>\"\n"
        "      \"priority\": \"<Recommendation priority (1 = highest, increasing number = lower priority)>\"\n"
        "    }\n"
        "  ]\n"
        "}"
    )

    openai_client = get_openai_client()

    try:
        response = openai_client.chat.completions.create(
            model="deepseek-reasoner",
            temperature=0.7,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
    except Exception as e:
        return {"error": f"OpenAI API call failed: {e}"}
    
    completion_text = response.choices[0].message.content.strip()
    completion_text = clean_completion_text(completion_text)

    try:
        llm_json = json.loads(completion_text)
    except json.JSONDecodeError:
        return {"error": "Failed to parse LLM response as JSON.", "raw_response": completion_text}

    valid_products = llm_json.get("valid_products") or []

    return valid_products
