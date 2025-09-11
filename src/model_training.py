import os
import joblib
from sklearn.linear_model import LogisticRegression
from src.logger import get_logger
from google.cloud import storage

logger = get_logger(__name__)

PREPROCESSED_DIR = "artifacts/preprocessed"
os.makedirs(PREPROCESSED_DIR, exist_ok=True)


class ModelTraining:
    def __init__(self):
        self.model = None
        self.client = storage.Client()
        self.bucket_name = "project_878787"  # GCP bucket name
        self.gcp_dir = "models"  # Folder in bucket to save model

    def train(self, X_train, y_train):
        logger.info("Initializing LogisticRegression model...")
        self.model = LogisticRegression(random_state=42, max_iter=1000)

        logger.info("Training LogisticRegression model...")
        self.model.fit(X_train, y_train)
        logger.info("Model training completed successfully")

        # Save locally
        model_path = os.path.join(PREPROCESSED_DIR, "logistic_regression_model.pkl")
        joblib.dump(self.model, model_path)
        logger.info(f"Trained model saved locally at {model_path}")

        # Upload to GCP
        try:
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(os.path.join(self.gcp_dir, "logistic_regression_model.pkl"))
            blob.upload_from_filename(model_path)
            logger.info(f"☁️ Trained model uploaded to GCP: gs://{self.bucket_name}/{self.gcp_dir}/logistic_regression_model.pkl")
        except Exception as e:
            logger.error(f"❌ Failed to upload model to GCP: {e}")

        return self.model
