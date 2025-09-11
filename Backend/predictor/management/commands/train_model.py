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
