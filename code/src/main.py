# src/main.py

from app import create_app

if __name__ == "__main__":
    app = create_app()
    # Run the server (in production, consider using Gunicorn or similar)
    app.run(debug=True, host="0.0.0.0", port=3000)
