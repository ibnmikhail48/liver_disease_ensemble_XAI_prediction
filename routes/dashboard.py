from flask import Blueprint, render_template, jsonify
from models import Prediction
from extensions import db
from sqlalchemy import func, case
from datetime import datetime, timedelta
import json

dashboard_bp = Blueprint('dashboard', __name__)


@dashboard_bp.route('/')
def index():
    """Main dashboard with all analytics."""

    # ── Summary cards ──────────────────────────────────────────
    total       = Prediction.query.count()
    disease     = Prediction.query.filter_by(prediction='Liver Disease').count()
    no_disease  = Prediction.query.filter_by(prediction='No Liver Disease').count()
    high_conf   = Prediction.query.filter_by(confidence='High').count()

    # ── Last 7 days predictions ────────────────────────────────
    week_ago    = datetime.utcnow() - timedelta(days=7)
    last_7_days = Prediction.query.filter(Prediction.timestamp >= week_ago).count()


    # ── Daily trend (last 14 days) ─────────────────────────────
    daily_trend = []
    for i in range(13, -1, -1):
        day_start = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0) - timedelta(days=i)
        day_end   = day_start + timedelta(days=1)
        count_d   = Prediction.query.filter(
            Prediction.timestamp >= day_start,
            Prediction.timestamp <  day_end
        ).count()
        count_ld  = Prediction.query.filter(
            Prediction.timestamp >= day_start,
            Prediction.timestamp <  day_end,
            Prediction.prediction == 'Liver Disease'
        ).count()
        daily_trend.append({
            'date'         : day_start.strftime('%b %d'),
            'total'        : count_d,
            'liver_disease': count_ld,
        })

    # ── Gender distribution ────────────────────────────────────
    gender_dist = db.session.query(
        Prediction.gender, func.count(Prediction.id)
    ).group_by(Prediction.gender).all()
    gender_data = {row[0]: row[1] for row in gender_dist}

    # ── Result colour distribution ─────────────────────────────
    color_dist = db.session.query(
        Prediction.result_color, func.count(Prediction.id)
    ).group_by(Prediction.result_color).all()
    color_data = {row[0]: row[1] for row in color_dist}

    # ── Confidence distribution ────────────────────────────────
    conf_dist = db.session.query(
        Prediction.confidence, func.count(Prediction.id)
    ).group_by(Prediction.confidence).all()
    conf_data = {row[0]: row[1] for row in conf_dist}

    # ── Age group distribution ─────────────────────────────────
    age_groups = {'0-30': 0, '31-45': 0, '46-60': 0, '60+': 0}
    all_preds  = Prediction.query.all()
    for p in all_preds:
        if   p.age <= 30: age_groups['0-30']  += 1
        elif p.age <= 45: age_groups['31-45'] += 1
        elif p.age <= 60: age_groups['46-60'] += 1
        else:             age_groups['60+']   += 1

    # ── Recent 10 predictions ──────────────────────────────────
    recent = Prediction.query.order_by(Prediction.timestamp.desc()).limit(10).all()

    stats = {
        'total'        : total,
        'disease'      : disease,
        'no_disease'   : no_disease,
        'high_conf'    : high_conf,
        'last_7_days'  : last_7_days,
        'disease_rate' : round(disease / total * 100, 1) if total else 0,
    }

    return render_template(
        'dashboard.html',
        stats        = stats,
        daily_trend  = json.dumps(daily_trend),
        gender_data  = json.dumps(gender_data),
        color_data   = json.dumps(color_data),
        conf_data    = json.dumps(conf_data),
        age_groups   = json.dumps(age_groups),
        recent       = recent,
    )


@dashboard_bp.route('/api/stats')
def api_stats():
    """JSON endpoint for live dashboard refresh."""
    total     = Prediction.query.count()
    disease   = Prediction.query.filter_by(prediction='Liver Disease').count()
    high_risk = Prediction.query.filter_by(risk_level='High').count()
    return jsonify({
        'total'       : total,
        'disease'     : disease,
        'no_disease'  : total - disease,
        'high_risk'   : high_risk,
        'disease_rate': round(disease / total * 100, 1) if total else 0,
    })
