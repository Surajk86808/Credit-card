from django.conf import settings
import os

def app_context(request):
    """Add common context variables to all templates"""
    context = {
        'app_name': 'Credit Card Fraud Detection',
        'app_version': '1.0.0',
        'model_status': get_model_status(),
        'powerbi_configured': False,
        'nav_items': [
            {'name': 'Home', 'url': '/', 'icon': 'fas fa-home'},
            {'name': 'Predict', 'url': '/predict/', 'icon': 'fas fa-brain'},
            {'name': 'Dashboard', 'url': '/dashboard/', 'icon': 'fas fa-chart-bar'},
        ],
    }
    return context

def get_model_status():
    """Check model availability"""
    try:
        model_files = [
            'artifacts/preprocessed/logistic_regression_model.pkl',
            'artifacts/preprocessed/scaler.pkl',
            'artifacts/preprocessed/feature_names.pkl'
        ]
        
        if all(os.path.exists(f) for f in model_files):
            return {'status': 'ready', 'message': 'Model ready', 'class': 'success'}
        else:
            return {'status': 'not_ready', 'message': 'Model needs training', 'class': 'warning'}
    except:
        return {'status': 'error', 'message': 'Error checking model', 'class': 'danger'}
