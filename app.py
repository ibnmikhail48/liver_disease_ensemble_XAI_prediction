"""
Liver Disease Prediction Web System — Flask Backend
Post-hoc Explainable Ensemble ML System
"""

from flask import Flask
from extensions import db
from dotenv import load_dotenv
from utils.cloudinary_config import configure_cloudinary
import os

# Load environment variables from .env
load_dotenv()


def create_app():
    app = Flask(__name__)

    # ── Configuration ──────────────────────────────────────────
    app.config['SECRET_KEY']                  = os.environ.get('SECRET_KEY', 'change-me')
    app.config['SQLALCHEMY_DATABASE_URI']     = os.environ.get('DATABASE_URL', 'sqlite:///liver_predictions.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # ── Extensions ─────────────────────────────────────────────
    db.init_app(app)

    # ── Register Blueprints ────────────────────────────────────
    from routes.main      import main_bp
    from routes.predict   import predict_bp
    from routes.dashboard import dashboard_bp
    from routes.history   import history_bp
    from routes.about     import about_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(predict_bp,   url_prefix='/predict')
    app.register_blueprint(dashboard_bp, url_prefix='/dashboard')
    app.register_blueprint(history_bp,   url_prefix='/history')
    app.register_blueprint(about_bp,     url_prefix='/about')

    # ── Create DB tables ───────────────────────────────────────
    with app.app_context():
        db.create_all()

    # ── Load ML model once at startup ─────────────────────────
    from utils.ml_engine import predictor
    predictor.load()

    configure_cloudinary()

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
