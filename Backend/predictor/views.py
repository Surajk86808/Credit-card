from django.shortcuts import render
from django import forms
import os
import json
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
    """Infer feature and target names from the raw CSV, assuming last column is target."""
    global FEATURE_NAMES, TARGET_NAME
    if not os.path.exists(RAW_CSV):
        return None, None
    try:
        df = pd.read_csv(RAW_CSV, nrows=sample_rows)
        df = df.dropna()
        cols = df.columns.tolist()
        if len(cols) < 2:
            return None, None
        TARGET_NAME = cols[-1]
        FEATURE_NAMES = cols[:-1]
        return FEATURE_NAMES, TARGET_NAME
    except Exception:
        return None, None


def _load_artifacts():
    """Lazy-load model artifacts if available."""
    global MODEL, SCALER, FEATURE_NAMES, TARGET_NAME

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
    """Dynamically generate numeric inputs for all detected features."""
    def __init__(self, *args, **kwargs):
        feature_names = kwargs.pop('feature_names', [])
        super().__init__(*args, **kwargs)
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
    """Interactive prediction form with optional probability display."""
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


def home_view(request):
    """Landing page for CredSafe app."""
    return render(request, 'home_template.html', {
        'app_name': 'CredSafe'
    })


def _load_sample_df(sample_rows=50000):
    """Load a sample of the dataset for dashboard analytics. Fallback to empty DataFrame if missing."""
    if not os.path.exists(RAW_CSV):
        return pd.DataFrame()
    try:
        return pd.read_csv(RAW_CSV, nrows=sample_rows)
    except Exception:
        return pd.DataFrame()


def dashboard_view(request):
    """Simple analytics dashboard powered by Chart.js.
    Shows class distribution and amount histogram if available.
    """
    # Ensure we have schema info
    _infer_schema_from_csv()

    df = _load_sample_df()
    kpis = {
        'total_transactions': int(df.shape[0]) if not df.empty else 0,
        'fraud_rate': None,
    }

    # Determine target column (last column by convention)
    target_col = TARGET_NAME if TARGET_NAME in (df.columns.tolist() if not df.empty else []) else None

    class_dist_labels = []
    class_dist_values = []

    if target_col is not None and not df.empty:
        # Compute fraud distribution
        vc = df[target_col].value_counts(dropna=False)
        total = vc.sum()
        class_dist_labels = [str(k) for k in vc.index.tolist()]
        class_dist_values = [int(v) for v in vc.values.tolist()]
        if set(vc.index.tolist()) <= {0, 1} or set(vc.index.tolist()) <= {'0', '1'}:
            fraud = int(vc.get(1, 0)) if 1 in vc.index else int(vc.get('1', 0))
            kpis['fraud_rate'] = round((fraud / total) * 100, 2) if total else None

    # Pick an amount-like column if present
    amount_candidates = ['Amount', 'amount', 'amt', 'TransactionAmt', 'transaction_amount']
    amount_col = None
    if not df.empty:
        for c in amount_candidates:
            if c in df.columns:
                amount_col = c
                break
        if amount_col is None:
            # fallback to first numeric non-target column
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            num_cols = [c for c in num_cols if c != target_col]
            if num_cols:
                amount_col = num_cols[0]

    amount_bins_labels = []
    amount_bins_values = []
    if not df.empty and amount_col is not None:
        try:
            series = pd.to_numeric(df[amount_col], errors='coerce').dropna()
            # Use quantile-based binning to avoid skew
            bins = np.unique(np.quantile(series, np.linspace(0, 1, 11)))
            if len(bins) >= 2:
                counts, edges = np.histogram(series, bins=bins)
                # Label bins like [a-b)
                labels = [f"{edges[i]:.0f}-{edges[i+1]:.0f}" for i in range(len(edges)-1)]
                amount_bins_labels = labels
                amount_bins_values = [int(x) for x in counts.tolist()]
        except Exception:
            pass

    charts = {
        'classDist': {
            'labels': class_dist_labels,
            'values': class_dist_values,
        },
        'amountHist': {
            'labels': amount_bins_labels,
            'values': amount_bins_values,
            'label': amount_col or 'Amount',
        },
    }

    context = {
        'app_name': 'CredSafe',
        'kpis': kpis,
        'charts_json': json.dumps(charts),
        'has_data': not df.empty,
        'target_col': target_col,
    }
    return render(request, 'dashboard.html', context)


def dashboard_powerbi_view(request):
    """Embed a Power BI report via URL query: ?embed=<FULL_EMBED_URL>"""
    powerbi_embed_url = request.GET.get('embed')
    return render(request, 'dashboard_powerbi.html', { 'powerbi_embed_url': powerbi_embed_url })


def dashboard_tableau_view(request):
    """Embed a Tableau visualization via URL query: ?embed=<FULL_EMBED_URL>"""
    tableau_embed_url = request.GET.get('embed')
    return render(request, 'dashboard_tableau.html', { 'tableau_embed_url': tableau_embed_url })