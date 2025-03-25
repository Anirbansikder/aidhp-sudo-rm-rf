# src/controllers/transaction_controller.py

from flask import Blueprint, request, jsonify
import logging

# Assume these are in the same package or an importable package
from services.transaction_service import (
    fetch_transactions_by_date,
    get_recommended_transaction_by_date,
    analyze_recommendable_transaction_by_date,
    analyze_recommendable_products_for_customer
)

transaction_bp = Blueprint('transaction_bp', __name__)
logger = logging.getLogger(__name__)

@transaction_bp.route('/fetch/by_date', methods=['GET'])
def get_transactions_by_date():
    """
    GET /api/transactions/fetch/by_date?date=MM/DD/YYYY
    Fetch ALL transactions for a given date.
    """
    date_str = request.args.get("date")
    if not date_str:
        return jsonify({"error": "Missing 'date' query parameter"}), 400
    
    logger.info(f"Fetching transactions for date: {date_str}")
    transactions = fetch_transactions_by_date(date_str)
    return jsonify({"transactions": transactions, "count": len(transactions)}), 200

@transaction_bp.route('/analyze/by_date', methods=['POST'])
def analyze_transactions_by_date():
    """
    POST /api/transactions/analyze/by_date
    Body: { "date": "MM/DD/YYYY" }
    1) Fetch unprocessed transactions for that date
    2) Use LLM to pick transactions for recommendation
    3) Return JSON with chosen transaction_id, and reason
    """
    data = request.get_json() or {}
    date_str = data.get("date")
    if not date_str:
        return jsonify({"error": "Missing 'date' in request body"}), 400

    logger.info(f"Analyzing transactions for date: {date_str}")
    result = get_recommended_transaction_by_date(date_str)

    # If there's an error key, handle that
    if "error" in result:
        return jsonify(result), 500

    return jsonify(result), 200

@transaction_bp.route('/analyze_recommendable_transactions/by_date', methods=['POST'])
def analyze_recommendable_transactions():
    data = request.get_json() or {}
    date_str = data.get("date")
    if not date_str:
        return jsonify({"error": "Date query parameter is required"}), 400

    logger.info(f"Analyzing transactions for date: {date_str}")
    result = analyze_recommendable_transaction_by_date(date_str)

    # If there's an error key, handle that
    if "error" in result:
        return jsonify(result), 500

    return jsonify({"result" : result}), 200

@transaction_bp.route('/analyze_customer_product', methods=['GET'])
def analyze_recommendable_transactions_for_customer():
    customer_str = request.args.get("customer_id")
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    if not customer_str or not start_date or not end_date:
        return jsonify({"error": "Query parameters are required"}), 400

    logger.info(f"Analyzing products for customer: {customer_str}")
    result = analyze_recommendable_products_for_customer(customer_str, start_date, end_date)

    # If there's an error key, handle that
    if "error" in result:
        return jsonify(result), 500

    return jsonify({"result" : result}), 200
