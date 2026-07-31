from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from models import Prediction
from extensions import db
from sqlalchemy import or_

history_bp = Blueprint('history', __name__)


@history_bp.route('/')
def index():
    """
    Patient records page with search, filter and pagination.
    Query params: page, search, prediction, risk, gender
    """
    page       = request.args.get('page', 1, type=int)
    search     = request.args.get('search', '').strip()
    pred_filter= request.args.get('prediction', '')
    risk_filter= request.args.get('risk', '')
    gender_f   = request.args.get('gender', '')
    per_page   = 15

    query = Prediction.query

    # ── Search by patient name or ID ───────────────────────────
    if search:
        query = query.filter(
            or_(
                Prediction.patient_name.ilike(f'%{search}%'),
                Prediction.patient_id.ilike(f'%{search}%'),
            )
        )

    # ── Filters ────────────────────────────────────────────────
    if pred_filter:
        query = query.filter(Prediction.prediction == pred_filter)
    if risk_filter:
        query = query.filter(Prediction.risk_level == risk_filter)
    if gender_f:
        query = query.filter(Prediction.gender == gender_f)

    # ── Paginate ───────────────────────────────────────────────
    pagination = query.order_by(
        Prediction.timestamp.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)

    return render_template(
        'history.html',
        records    = pagination.items,
        pagination = pagination,
        search     = search,
        pred_filter= pred_filter,
        risk_filter= risk_filter,
        gender_f   = gender_f,
    )


@history_bp.route('/view/<int:prediction_id>')
def view(prediction_id):
    """View full details of a single past prediction."""
    record       = Prediction.query.get_or_404(prediction_id)
    top_features = record.get_top_features()
    return render_template('result.html', record=record,
                           top_features=top_features,
                           shap_values=record.get_shap_values())


@history_bp.route('/delete/<int:prediction_id>', methods=['POST'])
def delete(prediction_id):
    """Delete a single prediction record."""
    record = Prediction.query.get_or_404(prediction_id)
    db.session.delete(record)
    db.session.commit()
    flash('Record deleted successfully.', 'success')
    return redirect(url_for('history.index'))


@history_bp.route('/update-notes/<int:prediction_id>', methods=['POST'])
def update_notes(prediction_id):
    """AJAX endpoint — update doctor notes on a prediction."""
    record = Prediction.query.get_or_404(prediction_id)
    notes  = request.form.get('notes', '').strip()
    record.notes = notes
    db.session.commit()
    return jsonify({'status': 'success', 'message': 'Notes updated.'})


@history_bp.route('/export')
def export():
    """Export all records as CSV download."""
    import csv, io
    from flask import Response

    records = Prediction.query.order_by(Prediction.timestamp.desc()).all()
    output  = io.StringIO()
    writer  = csv.writer(output)

    writer.writerow([
        'ID', 'Timestamp', 'Patient Name', 'Patient ID',
        'Age', 'Gender',
        'Total Bilirubin', 'Direct Bilirubin', 'Alkaline Phosphotase',
        'ALT', 'AST', 'Total Proteins', 'Albumin', 'AG Ratio',
        'Prediction', 'Probability (%)', 'Confidence', 'Result Color', 'Notes'
    ])
    for r in records:
        writer.writerow([
            r.id, r.timestamp.strftime('%Y-%m-%d %H:%M'),
            r.patient_name or '', r.patient_id or '',
            r.age, r.gender,
            r.total_bilirubin, r.direct_bilirubin, r.alkaline_phosphotase,
            r.alamine_aminotransferase, r.aspartate_aminotransferase,
            r.total_protiens, r.albumin, r.albumin_globulin_ratio,
            r.prediction, round(r.probability_disease * 100, 1),
            r.confidence, r.result_color, r.notes or ''
        ])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=liver_predictions.csv'}
    )
