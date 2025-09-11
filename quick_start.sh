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
