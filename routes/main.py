from flask import Blueprint, render_template
from models import Prediction
from extensions import db
from sqlalchemy import func

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    # Quick stats for the home page
    total       = Prediction.query.count()
    disease     = Prediction.query.filter_by(prediction='Liver Disease').count()
    no_disease  = Prediction.query.filter_by(prediction='No Liver Disease').count()
    recent      = Prediction.query.order_by(Prediction.timestamp.desc()).limit(5).all()

    stats = {
        'total'     : total,
        'disease'   : disease,
        'no_disease': no_disease,
        'disease_pct': round(disease / total * 100, 1) if total else 0
    }
    return render_template('index.html', stats=stats, recent=recent)
