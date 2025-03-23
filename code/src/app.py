# src/app.py

from flask import Flask
from controllers.transaction_controller import transaction_bp
# from controllers.recommendation_controller import recommendation_bp

def create_app():
    """
    Create and configure the Flask application.
    """
    app = Flask(__name__)

    # Register Blueprints for different controllers
    app.register_blueprint(transaction_bp, url_prefix='/api/transactions')
    # app.register_blueprint(recommendation_bp, url_prefix='/api/recommendations')

    return app
