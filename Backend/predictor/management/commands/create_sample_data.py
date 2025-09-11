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
