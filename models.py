"""
Database Models — Liver Disease Prediction System
"""

from extensions import db
from datetime import datetime
import json


class Prediction(db.Model):
    """Stores every prediction made through the web system."""
    __tablename__ = 'predictions'

    id                         = db.Column(db.Integer, primary_key=True)
    timestamp                  = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # ── Patient inputs ──────────────────────────────────────────
    patient_name               = db.Column(db.String(120), nullable=True)
    patient_id                 = db.Column(db.String(50),  nullable=True)
    age                        = db.Column(db.Integer,     nullable=False)
    gender                     = db.Column(db.String(10),  nullable=False)
    total_bilirubin            = db.Column(db.Float,       nullable=False)
    direct_bilirubin           = db.Column(db.Float,       nullable=False)
    alkaline_phosphotase       = db.Column(db.Integer,     nullable=False)
    alamine_aminotransferase   = db.Column(db.Integer,     nullable=False)
    aspartate_aminotransferase = db.Column(db.Integer,     nullable=False)
    total_protiens             = db.Column(db.Float,       nullable=False)
    albumin                    = db.Column(db.Float,       nullable=False)
    albumin_globulin_ratio     = db.Column(db.Float,       nullable=False)

    # ── Prediction outputs ──────────────────────────────────────
    prediction                 = db.Column(db.String(30),  nullable=False)  # 'Liver Disease' / 'No Liver Disease'
    probability_disease        = db.Column(db.Float,       nullable=False)
    probability_no_disease     = db.Column(db.Float,       nullable=False)
    confidence                 = db.Column(db.String(20),  nullable=False)  # High / Moderate / Low
    result_color               = db.Column(db.String(10),  nullable=False, default='green')  # green / amber / red
    threshold_used             = db.Column(db.Float,       nullable=False)

    # ── XAI outputs (stored as JSON strings) ────────────────────
    shap_values_json           = db.Column(db.Text,        nullable=True)
    top_features_json          = db.Column(db.Text,        nullable=True)
    shap_plot_path             = db.Column(db.String(500), nullable=True)

    # ── Doctor notes ────────────────────────────────────────────
    notes                      = db.Column(db.Text,        nullable=True)

    def get_shap_values(self):
        if self.shap_values_json:
            return json.loads(self.shap_values_json)
        return {}

    def get_top_features(self):
        if self.top_features_json:
            return json.loads(self.top_features_json)
        return []

    def to_dict(self):
        return {
            'id'                    : self.id,
            'timestamp'             : self.timestamp.strftime('%Y-%m-%d %H:%M'),
            'patient_name'          : self.patient_name or 'Anonymous',
            'patient_id'            : self.patient_id   or '—',
            'age'                   : self.age,
            'gender'                : self.gender,
            'prediction'            : self.prediction,
            'probability_disease'   : round(self.probability_disease * 100, 1),
            'probability_no_disease': round(self.probability_no_disease * 100, 1),
            'confidence'            : self.confidence,
            'result_color'          : self.result_color,
            'top_features'          : self.get_top_features(),
        }

    def __repr__(self):
        return f'<Prediction #{self.id} | {self.patient_name} | {self.prediction}>'
