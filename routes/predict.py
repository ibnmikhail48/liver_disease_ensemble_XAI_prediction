from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from models import Prediction
from extensions import db
from utils.ml_engine import predictor
import json

predict_bp = Blueprint('predict', __name__)


@predict_bp.route('/', methods=['GET'])
def form():
    """Render the patient input form."""
    return render_template('predict.html')


@predict_bp.route('/run', methods=['POST'])
def run():
    """
    Receives form data → runs ML prediction → saves to DB → redirects to result.
    """
    try:
        # ── Collect raw inputs ─────────────────────────────────
        raw = {
            'age'                       : request.form.get('age'),
            'gender'                    : request.form.get('gender'),
            'total_bilirubin'           : request.form.get('total_bilirubin'),
            'direct_bilirubin'          : request.form.get('direct_bilirubin'),
            'alkaline_phosphotase'      : request.form.get('alkaline_phosphotase'),
            'alamine_aminotransferase'  : request.form.get('alamine_aminotransferase'),
            'aspartate_aminotransferase': request.form.get('aspartate_aminotransferase'),
            'total_protiens'            : request.form.get('total_protiens'),
            'albumin'                   : request.form.get('albumin'),
            'albumin_globulin_ratio'    : request.form.get('albumin_globulin_ratio'),
        }
        patient_name = request.form.get('patient_name', '').strip() or None
        patient_id   = request.form.get('patient_id',   '').strip() or None
        notes        = request.form.get('notes',        '').strip() or None

        # ── Validate all numeric fields present ────────────────
        for key, val in raw.items():
            if not val:
                flash(f'Missing field: {key.replace("_", " ").title()}', 'error')
                return redirect(url_for('predict.form'))

        # ── Run ML prediction ──────────────────────────────────
        result = predictor.predict(raw)

        # ── Save to database ───────────────────────────────────
        record = Prediction(
            patient_name               = patient_name,
            patient_id                 = patient_id,
            age                        = int(float(raw['age'])),
            gender                     = raw['gender'],
            total_bilirubin            = float(raw['total_bilirubin']),
            direct_bilirubin           = float(raw['direct_bilirubin']),
            alkaline_phosphotase       = int(float(raw['alkaline_phosphotase'])),
            alamine_aminotransferase   = int(float(raw['alamine_aminotransferase'])),
            aspartate_aminotransferase = int(float(raw['aspartate_aminotransferase'])),
            total_protiens             = float(raw['total_protiens']),
            albumin                    = float(raw['albumin']),
            albumin_globulin_ratio     = float(raw['albumin_globulin_ratio']),
            prediction                 = result['prediction'],
            probability_disease        = result['probability_disease'],
            probability_no_disease     = result['probability_no_disease'],
            confidence                 = result['confidence'],
            result_color               = result['result_color'],
            threshold_used             = result['threshold_used'],
            shap_values_json           = json.dumps(result['shap_values']),
            top_features_json          = json.dumps(result['top_features']),
            shap_plot_path             = result['shap_plot_path'],
            notes                      = notes,
        )
        db.session.add(record)
        db.session.commit()

        return redirect(url_for('predict.result', prediction_id=record.id))

    except ValueError as e:
        flash(f'Invalid input value: {e}', 'error')
        return redirect(url_for('predict.form'))
    except Exception as e:
        flash(f'Prediction error: {e}', 'error')
        return redirect(url_for('predict.form'))


@predict_bp.route('/result/<int:prediction_id>')
def result(prediction_id):
    """Display prediction result with SHAP explanation."""
    record = Prediction.query.get_or_404(prediction_id)
    top_features = record.get_top_features()
    shap_values  = record.get_shap_values()
    return render_template(
        'result.html',
        record       = record,
        top_features = top_features,
        shap_values  = shap_values,
    )


@predict_bp.route('/api/predict', methods=['POST'])
def api_predict():
    """
    REST API endpoint — accepts JSON, returns JSON prediction.
    Useful for testing or external integrations.
    """
    try:
        data   = request.get_json(force=True)
        result = predictor.predict(data)
        # Remove numpy arrays from JSON response
        result.pop('feature_map', None)
        result.pop('shap_values', None)
        return jsonify({'status': 'success', 'data': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400
