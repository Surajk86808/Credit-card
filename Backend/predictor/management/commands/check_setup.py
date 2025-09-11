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
