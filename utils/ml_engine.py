"""
ML Engine — Loads model, runs predictions, generates SHAP explanations
"""

import joblib
import json
import numpy as np
import os
import shap
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import cloudinary.uploader
import io
from sklearn.pipeline import Pipeline


# ── Paths ──────────────────────────────────────────────────────────────────
# ml_engine.py is inside /utils/
# The model folders are in the project root.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, 'liver_disease_models')
XAI_DIR   = os.path.join(BASE_DIR, 'liver_disease_xai')

class LiverDiseasePredictor:
    """
    Wraps the saved ensemble pipeline.
    Handles:
      - Feature engineering from raw inputs
      - Prediction with optimal threshold
      - SHAP local explanations for the result page
    """

    def __init__(self):
        self.model           = None
        self.rf_model        = None          # for TreeExplainer SHAP
        self.xgb_model       = None
        self.threshold       = 0.5
        self.feature_names   = []
        self.shap_explainer  = None
        self._loaded         = False

    def load(self):
        """Load all artifacts from disk. Called once at app startup."""
        try:
            # ── Main ensemble ──────────────────────────────────
            self.model = joblib.load(
                os.path.join(MODEL_DIR, 'final_soft_voting_ensemble.pkl'))

            # ── Individual pipelines for SHAP (RF is fastest) ─
            rf_path  = os.path.join(MODEL_DIR, 'pipeline_random_forest.pkl')
            xgb_path = os.path.join(MODEL_DIR, 'pipeline_xgboost.pkl')
            if os.path.exists(rf_path):
                self.rf_pipeline = joblib.load(rf_path)
                # Extract raw RF from pipeline for TreeExplainer
                self.rf_model = self.rf_pipeline.named_steps['clf']
                # Extract ONLY the scaler for preprocessing before SHAP.
                # NOTE: this pipeline also contains a SMOTEENN step, which
                # is a training-time class-balancing resampler with no
                # .transform() method — it must NEVER be applied at
                # inference/explanation time, only the scaler should be.
                self.rf_preprocessor = self.rf_pipeline.named_steps.get('scaler', None)
            else:
                self.rf_preprocessor = None
            if os.path.exists(xgb_path):
                self.xgb_pipeline = joblib.load(xgb_path)

            # ── Optimal threshold ──────────────────────────────
            thr_path = os.path.join(MODEL_DIR, 'optimal_thresholds.json')
            if os.path.exists(thr_path):
                with open(thr_path) as f:
                    self.threshold = json.load(f).get('ensemble', 0.5)

            # ── Feature names ──────────────────────────────────
            feat_path = os.path.join(MODEL_DIR, 'selected_features.json')
            if os.path.exists(feat_path):
                with open(feat_path) as f:
                    self.feature_names = json.load(f)

            # ── SHAP TreeExplainer on RF ───────────────────────
            if self.rf_model is not None:
                self.shap_explainer = shap.TreeExplainer(self.rf_model)

            self._loaded = True
            print(f"✅ Model loaded | threshold={self.threshold:.4f} "
                  f"| features={len(self.feature_names)}")
        except Exception as e:
            print(f"❌ Model load error: {e}")
            self._loaded = False

    # ── Feature engineering ────────────────────────────────────────────────
    def _engineer_features(self, raw):
        """
        Applies the same feature engineering pipeline used during training.
        raw: dict with keys matching the 10 raw input fields.
        Returns: numpy array shaped (1, n_features) in SELECTED_FEATURES order.
        """
        age                        = float(raw['age'])
        gender                     = 1 if str(raw['gender']).lower() == 'male' else 0
        total_bilirubin            = float(raw['total_bilirubin'])
        direct_bilirubin           = float(raw['direct_bilirubin'])
        alkaline_phosphotase       = float(raw['alkaline_phosphotase'])
        alamine_aminotransferase   = float(raw['alamine_aminotransferase'])
        aspartate_aminotransferase = float(raw['aspartate_aminotransferase'])
        total_protiens             = float(raw['total_protiens'])
        albumin                    = float(raw['albumin'])
        albumin_globulin_ratio     = float(raw['albumin_globulin_ratio'])

        # Engineered features
        bilirubin_ratio       = direct_bilirubin        / (total_bilirubin + 1e-6)
        ast_alt_ratio         = aspartate_aminotransferase / (alamine_aminotransferase + 1e-6)
        globulin              = total_protiens           - albumin
        enzyme_load           = alkaline_phosphotase + alamine_aminotransferase + aspartate_aminotransferase
        bili_ast_interaction  = total_bilirubin          * aspartate_aminotransferase
        albumin_protein_ratio = albumin                  / (total_protiens + 1e-6)
        is_elder              = int(age >= 60)
        alt_albumin_ratio     = alamine_aminotransferase / (albumin + 1e-6)
        bilirubin_protein     = total_bilirubin          * total_protiens

        if   age <= 30: age_group = 0
        elif age <= 45: age_group = 1
        elif age <= 60: age_group = 2
        else:           age_group = 3

        # Full feature map — keys must match SELECTED_FEATURES exactly
        feature_map = {
            'Age'                             : age,
            'Gender'                          : gender,
            'Total_Bilirubin'                 : total_bilirubin,
            'Direct_Bilirubin'                : direct_bilirubin,
            'Alkaline_Phosphotase'            : alkaline_phosphotase,
            'Alamine_Aminotransferase'        : alamine_aminotransferase,
            'Aspartate_Aminotransferase'      : aspartate_aminotransferase,
            'Total_Protiens'                  : total_protiens,
            'Albumin'                         : albumin,
            'Albumin_and_Globulin_Ratio'      : albumin_globulin_ratio,
            'Bilirubin_Ratio'                 : bilirubin_ratio,
            'AST_ALT_Ratio'                   : ast_alt_ratio,
            'Globulin'                        : globulin,
            'Enzyme_Load'                     : enzyme_load,
            'Age_Group'                       : age_group,
            'Bili_AST_Interaction'            : bili_ast_interaction,
            'Albumin_Protein_Ratio'           : albumin_protein_ratio,
            'Is_Elder'                        : is_elder,
            'ALT_Albumin_Ratio'               : alt_albumin_ratio,
            'Bilirubin_Protein'               : bilirubin_protein,
            'Total_Bilirubin_log'             : np.log1p(total_bilirubin),
            'Direct_Bilirubin_log'            : np.log1p(direct_bilirubin),
            'Alkaline_Phosphotase_log'        : np.log1p(alkaline_phosphotase),
            'Alamine_Aminotransferase_log'    : np.log1p(alamine_aminotransferase),
            'Aspartate_Aminotransferase_log'  : np.log1p(aspartate_aminotransferase),
            'Enzyme_Load_log'                 : np.log1p(enzyme_load),
        }

        # Build array in the exact order of SELECTED_FEATURES
        row = [feature_map[f] for f in self.feature_names]
        return np.array(row).reshape(1, -1), feature_map

    # ── Main prediction ────────────────────────────────────────────────────
    def predict(self, raw_inputs):
        """
        Full prediction pipeline.
        Returns dict with prediction, probability, confidence, SHAP explanation.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call predictor.load() first.")

        X, feature_map = self._engineer_features(raw_inputs)

        # ── Probability from ensemble ──────────────────────────
        prob_disease    = float(self.model.predict_proba(X)[0][1])
        prob_no_disease = 1.0 - prob_disease
        prediction      = 'Liver Disease' if prob_disease >= self.threshold else 'No Liver Disease'

        # ── Confidence (how certain is the model about its classification) ──
        # Based on distance from threshold — not from 0.5
        margin = abs(prob_disease - self.threshold)
        if margin > 0.25:
            confidence = 'High'
        elif margin > 0.10:
            confidence = 'Moderate'
        else:
            confidence = 'Low'

        # ── Result colour — drives HCI display on result page ──
        # Green   = No Liver Disease (any confidence)
        # Amber   = Liver Disease, low confidence (prob between threshold and threshold+0.20)
        # Red     = Liver Disease, high confidence (prob > threshold+0.20)
        if prediction == 'No Liver Disease':
            result_color = 'green'
        elif prob_disease <= self.threshold + 0.20:
            result_color = 'amber'
        else:
            result_color = 'red'

        # ── SHAP local explanation ──────────────────────────────
        shap_vals, top_features, plot_path = self._explain(X, feature_map)

        return {
            'prediction'            : prediction,
            'probability_disease'   : round(prob_disease,    4),
            'probability_no_disease': round(prob_no_disease, 4),
            'confidence'            : confidence,
            'result_color'          : result_color,
            'threshold_used'        : round(self.threshold,  4),
            'shap_values'           : shap_vals,
            'top_features'          : top_features,
            'shap_plot_path'        : plot_path,
            'feature_map'           : feature_map,
        }

    # ── SHAP local explanation ─────────────────────────────────────────────
    def _explain(self, X_raw, feature_map):
        """
        Generates SHAP values for a single prediction.
        X_raw: raw, unscaled (1, n_features) array in SELECTED_FEATURES order
               (same array used for the ensemble prediction).
        Returns: shap_dict, top_features list, plot_path
        """
        try:
            if self.shap_explainer is None:
                return {}, [], None

            # ── Apply the SAME preprocessing the RF was trained on ──
            # (e.g. StandardScaler) before handing it to TreeExplainer.
            # Without this, the tree sees raw values on a completely
            # different scale than it was trained on, producing
            # meaningless/uniformly-biased SHAP values.
            if self.rf_preprocessor is not None:
                X_for_shap = self.rf_preprocessor.transform(X_raw)
            else:
                X_for_shap = X_raw

            # SHAP values for the positive class
            explanation = self.shap_explainer(X_for_shap)

            sv = explanation.values
            
            # New SHAP format: (samples, features, classes)
            if len(sv.shape) == 3:
                sv = sv[0, :, 1]      # class 1 = Liver Disease
            else:
                sv = sv[0]                # first sample

            print("Final SHAP shape:", sv.shape)

            # Dict of {feature: shap_value}
            shap_dict = {
                f: round(float(v), 5)
                for f, v in zip(self.feature_names, sv)
            }

            # Top 8 features sorted by |SHAP|
            top_features = sorted(
                [{'feature': f, 'shap_value': v,
                'direction': 'increases' if v > 0 else 'decreases',
                'raw_value': round(feature_map.get(f, 0), 4)}
                for f, v in shap_dict.items()],
                key=lambda x: abs(x['shap_value']),
                reverse=True
            )[:8]

            # ── Generate SHAP waterfall bar chart ─────────────
            plot_path = self._plot_shap(top_features)

            return shap_dict, top_features, plot_path

        except Exception as e:
            print(f"SHAP explanation error: {e}")
            return {}, [], None

    def _plot_shap(self, top_features):
        """Creates a horizontal SHAP bar chart and uploads it to Cloudinary."""
        try:
            timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
            public_id = f'shap_plots/shap_{timestamp}'

            features = [f['feature'].replace('_', ' ')  for f in top_features]
            values   = [f['shap_value']                  for f in top_features]
            colors   = ['#e74c3c' if v > 0 else '#2ecc71' for v in values]

            fig, ax = plt.subplots(figsize=(9, 5))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#0f172a')

            y_pos = range(len(features))
            bars  = ax.barh(list(y_pos), values, color=colors,
                            edgecolor='none', height=0.6, alpha=0.92)

            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(features, fontsize=10, color='#e2e8f0')
            ax.axvline(0, color='#475569', linewidth=1.2)
            ax.set_xlabel('SHAP value  (impact on prediction)', fontsize=9,
                color='#94a3b8')
            ax.set_title('Feature Contributions to This Prediction',
                fontsize=11, fontweight='bold', color='#f1f5f9', pad=12)

            for bar, val in zip(bars, values):
                ax.text(
                    val + (0.002 if val >= 0 else -0.002),
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:+.3f}',
                    va='center',
                    ha='left' if val >= 0 else 'right',
                    fontsize=8.5, color='#e2e8f0'
                )

            red_patch   = mpatches.Patch(color='#e74c3c', label='Increases disease risk')
            green_patch = mpatches.Patch(color='#2ecc71', label='Decreases disease risk')
            ax.legend(handles=[red_patch, green_patch], loc='lower right',
                    fontsize=8, facecolor='#1e293b', labelcolor='#e2e8f0',
                    edgecolor='#334155')

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#334155')
            ax.spines['bottom'].set_color('#334155')
            ax.tick_params(colors='#94a3b8')

            plt.tight_layout()

            # ── Save to an in-memory buffer instead of disk ──
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                        facecolor='#0f172a')
            plt.close()
            buf.seek(0)

            # ── Upload buffer directly to Cloudinary ──
            upload_result = cloudinary.uploader.upload(
                buf,
                public_id=public_id,
                folder='liver_app/shap_plots',
                overwrite=True,
                resource_type='image'
            )

            return upload_result['secure_url']

        except Exception as e:
            print(f"SHAP plot error: {e}")
            return None


# ── Singleton instance ─────────────────────────────────────────────────────
predictor = LiverDiseasePredictor()
