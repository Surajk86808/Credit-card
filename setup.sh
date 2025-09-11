#!/bin/bash

# Credit Card Fraud Detection - Complete Deployment Script
echo "🚀 Setting up Credit Card Fraud Detection System..."

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p Backend/predictor/management/commands
mkdir -p Backend/creditrisk
mkdir -p templates
mkdir -p static/css
mkdir -p static/js
mkdir -p logs

# Create __init__.py files
echo "📝 Creating __init__.py files..."
touch Backend/predictor/management/__init__.py
touch Backend/predictor/management/commands/__init__.py

# Create management commands
echo "🛠️ Creating management commands..."

# train_model.py
cat > Backend/predictor/management/commands/train_model.py << 'EOF'
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from pipelines.pipeline import Pipeline
    from src.logger import get_logger
except ImportError as e:
    print(f"Error importing pipeline modules: {e}")

logger = get_logger(__name__)

class Command(BaseCommand):
    help = 'Train the fraud detection model and prepare artifacts'

    def add_arguments(self, parser):
        parser.add_argument('--use-mlflow', action='store_true', help='Enable MLflow logging')
        parser.add_argument('--force-retrain', action='store_true', help='Force retrain existing model')

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('🚀 Starting Credit Card Fraud Detection Model Training'))
        
        try:
            # Check if model exists
            if not options['force_retrain'] and self.model_exists():
                self.stdout.write(self.style.WARNING('⚠️  Model already exists. Use --force-retrain to retrain.'))
                return

            # Initialize pipeline
            use_mlflow = options['use_mlflow']
            self.stdout.write(f'📊 Initializing pipeline (MLflow: {use_mlflow})')
            pipeline = Pipeline(use_mlflow=use_mlflow)

            # Run training
            self.stdout.write('🤖 Running machine learning pipeline...')
            results = pipeline.run()

            # Display results
            self.stdout.write('\n' + '='*50)
            self.stdout.write(self.style.SUCCESS('📈 TRAINING RESULTS'))
            self.stdout.write('='*50)
            
            if 'accuracy' in results:
                accuracy = results['accuracy'] * 100
                self.stdout.write(f'🎯 Accuracy: {accuracy:.2f}%')
            
            if 'roc_auc' in results and results['roc_auc']:
                self.stdout.write(f'📊 ROC-AUC: {results["roc_auc"]:.4f}')
            
            self.stdout.write('='*50)
            self.stdout.write(self.style.SUCCESS('✅ Model training completed successfully!'))
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise CommandError(f'❌ Model training failed: {e}')

    def model_exists(self):
        """Check if model artifacts exist"""
        files = [
            'artifacts/preprocessed/logistic_regression_model.pkl',
            'artifacts/preprocessed/scaler.pkl', 
            'artifacts/preprocessed/feature_names.pkl'
        ]
        return all(os.path.exists(f) for f in files)
EOF

# check_setup.py
cat > Backend/predictor/management/commands/check_setup.py << 'EOF'
from django.core.management.base import BaseCommand
from django.conf import settings
import os

class Command(BaseCommand):
    help = 'Check the fraud detection system setup and configuration'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('🔍 Credit Card Fraud Detection System - Setup Check'))
        self.stdout.write('='*60)

        # Check directories
        self.stdout.write('\n📁 Checking Directory Structure:')
        required_dirs = [
            'artifacts', 'artifacts/raw', 'artifacts/preprocessed',
            'artifacts/models', 'artifacts/reports', 'logs'
        ]
        
        for dir_path in required_dirs:
            if os.path.exists(dir_path):
                self.stdout.write(f'  ✅ {dir_path}')
            else:
                self.stdout.write(f'  ❌ {dir_path} (will be created)')
                os.makedirs(dir_path, exist_ok=True)
                self.stdout.write(f'     📝 Created: {dir_path}')

        # Check model artifacts
        self.stdout.write('\n🤖 Checking Model Artifacts:')
        model_files = [
            'logistic_regression_model.pkl',
            'scaler.pkl',
            'feature_names.pkl'
        ]
        
        missing = []
        for filename in model_files:
            filepath = f'artifacts/preprocessed/{filename}'
            if os.path.exists(filepath):
                size = os.path.getsize(filepath) / 1024
                self.stdout.write(f'  ✅ {filename} ({size:.1f} KB)')
            else:
                self.stdout.write(f'  ❌ {filename} (missing)')
                missing.append(filename)
        
        if missing:
            self.stdout.write('\n💡 To create missing artifacts, run:')
            self.stdout.write('   python manage.py train_model')

        self.stdout.write('\n' + '='*60)
        self.stdout.write(self.style.SUCCESS('✅ Setup check completed!'))
EOF

# create_sample_data.py
cat > Backend/predictor/management/commands/create_sample_data.py << 'EOF'
from django.core.management.base import BaseCommand
import pandas as pd
import numpy as np
import os

class Command(BaseCommand):
    help = 'Create sample fraud detection data for testing'

    def add_arguments(self, parser):
        parser.add_argument('--size', type=int, default=10000, help='Number of sample transactions')

    def handle(self, *args, **options):
        size = options['size']
        self.stdout.write(f'🔄 Generating {size:,} sample transactions...')
        
        # Generate realistic sample data
        np.random.seed(42)
        
        data = {
            'TransactionAmount': np.random.lognormal(3, 1, size),
            'TransactionTime': np.random.uniform(0, 86400, size),
            'CustomerAge': np.random.normal(45, 15, size).clip(18, 80),
            'CreditLimit': np.random.lognormal(8, 0.5, size),
            'AvailableBalance': np.random.uniform(0, 10000, size),
            'TransactionLocation': np.random.choice(['Online', 'Store', 'ATM'], size),
            'CardNetwork': np.random.choice(['Visa', 'Mastercard', 'Amex'], size),
            'CardType': np.random.choice(['Debit', 'Credit'], size),
        }
        
        # Create fraud labels (0.1% fraud rate)
        fraud = np.zeros(size)
        fraud_indices = np.random.choice(size, int(size * 0.001), replace=False)
        fraud[fraud_indices] = 1
        data['Fraud'] = fraud
        
        # Make fraudulent transactions more suspicious
        data['TransactionAmount'][fraud_indices] *= np.random.uniform(2, 5, len(fraud_indices))
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        os.makedirs('artifacts/raw', exist_ok=True)
        filepath = f'artifacts/raw/sample_fraud_data_{size}.csv'
        df.to_csv(filepath, index=False)
        
        self.stdout.write(self.style.SUCCESS(f'✅ Created sample data: {filepath}'))
        self.stdout.write(f'  Total: {len(df):,}')
        self.stdout.write(f'  Fraud: {fraud.sum():,} ({fraud.mean()*100:.3f}%)')
        self.stdout.write('\n💡 Now run: python manage.py train_model')
EOF

# Create context_processors.py
echo "🔧 Creating context processors..."
cat > Backend/predictor/context_processors.py << 'EOF'
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
EOF

echo "📋 Creating quick start script..."
cat > quick_start.sh << 'EOF'
#!/bin/bash
echo "🚀 Quick Start - Credit Card Fraud Detection"

cd Backend

echo "1️⃣ Checking setup..."
python manage.py check_setup

echo "2️⃣ Creating database..."
python manage.py migrate

echo "3️⃣ Creating sample data..."
python manage.py create_sample_data --size 50000

echo "4️⃣ Training model..."
python manage.py train_model --use-mlflow

echo "5️⃣ Starting server..."
echo "🌐 Visit: http://localhost:8000"
python manage.py runserver
EOF

chmod +x quick_start.sh

echo "✅ Setup complete! Files created:"
echo "📁 Backend/predictor/management/commands/"
echo "📁 Backend/predictor/context_processors.py" 
echo "📋 quick_start.sh"
echo ""
echo "🚀 Next steps:"
echo "1. Copy the remaining template files (home.html, dashboard.html, enhanced predict.html)"
echo "2. Update Backend/predictor/views.py and urls.py" 
echo "3. Update Backend/creditrisk/settings.py"
echo "4. Run: ./quick_start.sh"