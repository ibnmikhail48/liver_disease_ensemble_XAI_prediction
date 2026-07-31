from flask import Blueprint, render_template

about_bp = Blueprint('about', __name__)

@about_bp.route('/')
def index():
    model_info = {
        'models': [
            {'name': 'Random Forest',       'type': 'Ensemble (Bagging)',   'tuning': 'RandomizedSearchCV'},
            {'name': 'Logistic Regression', 'type': 'Linear Classifier',    'tuning': 'GridSearchCV'},
            {'name': 'Decision Tree',       'type': 'Tree-Based',           'tuning': 'GridSearchCV'},
            {'name': 'K-Nearest Neighbors', 'type': 'Instance-Based',       'tuning': 'GridSearchCV'},
            {'name': 'Support Vector Machine','type': 'Kernel-Based',       'tuning': 'GridSearchCV'},
            {'name': 'XGBoost',             'type': 'Ensemble (Boosting)',   'tuning': 'RandomizedSearchCV'},
        ],
        'ensemble'   : 'Soft Voting (probability averaging across all 6 models)',
        'balancing'  : 'SMOTEENN (inside ImbPipeline — no CV leakage)',
        'threshold'  : 'Optimal (Youden\'s J statistic)',
        'xai_methods': [
            'SHAP TreeExplainer (RF, XGBoost)',
            'SHAP KernelExplainer (LR, SVM)',
            'Permutation Feature Importance',
            'Partial Dependence Plots (PDP)',
            'Individual Conditional Expectation (ICE)',
            'LIME Global Surrogate',
            'Global Surrogate Decision Tree',
            'Cross-Method Consensus Ranking',
        ],
        'dataset': 'Liver Disease Patient Dataset hosted on Kaggle Repository containing 30,691 records with 11 attributes. '
    }
    return render_template('about.html', model_info=model_info)
