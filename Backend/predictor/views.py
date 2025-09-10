from django.shortcuts import render
from django import forms
import os
import joblib
import numpy as np
import pandas as pd

# Paths to artifacts
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ART_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'preprocessed')
RAW_CSV = os.path.join(PROJECT_ROOT, 'artifacts', 'raw', 'enhanced_credit_card_fraud_dataset_2M.csv')

# Globals loaded once
MODEL = None
SCALER = None
FEATURE_NAMES = None
TARGET_NAME = None


def _infer_schema_from_csv(sample_rows=2000):
    global FEATURE_NAMES, TARGET_NAME
    if not os.path.exists(RAW_CSV):
        return None, None
    try:
        # Read a small sample to infer schema
        df = pd.read_csv(RAW_CSV, nrows=sample_rows)
        df = df.dropna()
        # Assume last column is target
        cols = df.columns.tolist()
        if len(cols) < 2:
            return None, None
        TARGET_NAME = cols[-1]
        FEATURE_NAMES = cols[:-1]
        return FEATURE_NAMES, TARGET_NAME
    except Exception:
        return None, None


def _load_artifacts():
    global MODEL, SCALER, FEATURE_NAMES, TARGET_NAME

    # Try to load known artifacts
    model_path = os.path.join(ART_DIR, 'logistic_regression_model.pkl')
    scaler_path = os.path.join(ART_DIR, 'scaler.pkl')
    feat_path = os.path.join(ART_DIR, 'feature_names.pkl')

    if os.path.exists(scaler_path):
        SCALER = joblib.load(scaler_path)
    if os.path.exists(model_path):
        MODEL = joblib.load(model_path)
    if os.path.exists(feat_path):
        FEATURE_NAMES = joblib.load(feat_path)

    if FEATURE_NAMES is None:
        _infer_schema_from_csv()


class DynamicPredictForm(forms.Form):
    def __init__(self, *args, **kwargs):
        feature_names = kwargs.pop('feature_names', [])
        super().__init__(*args, **kwargs)
        # Create float fields by default; can be customized later if needed
        for name in feature_names:
            self.fields[name] = forms.FloatField(
                required=True,
                label=name.replace('_', ' ').title(),
                widget=forms.NumberInput(attrs={
                    'class': 'form-control',
                    'placeholder': f'Enter {name}'
                })
            )


def predict_view(request):
    _load_artifacts()
    context = {
        'has_model': MODEL is not None and SCALER is not None,
    }

    feature_names = FEATURE_NAMES or []

    if request.method == 'POST' and feature_names:
        form = DynamicPredictForm(request.POST, feature_names=feature_names)
        if form.is_valid() and MODEL is not None and SCALER is not None:
            # Arrange features in training order
            values = [form.cleaned_data[name] for name in feature_names]
            X = np.array(values, dtype=float).reshape(1, -1)
            try:
                X_scaled = SCALER.transform(X)
            except Exception:
                # Fallback if scaler missing or incompatible
                X_scaled = X
            pred = MODEL.predict(X_scaled)[0]
            proba = None
            try:
                proba = float(MODEL.predict_proba(X_scaled)[0, 1])
            except Exception:
                pass
            context.update({
                'form': form,
                'prediction': int(pred),
                'probability': proba,
                'feature_names': feature_names,
            })
            return render(request, 'predict.html', context)
    else:
        form = DynamicPredictForm(feature_names=feature_names)

    context.update({
        'form': form,
        'feature_names': feature_names,
    })
    return render(request, 'predict.html', context)