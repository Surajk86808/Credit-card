import os
import joblib
from sklearn.linear_model import LogisticRegression
from src.logger import get_logger

logger = get_logger(__name__)

PREPROCESSED_DIR = "artifacts/preprocessed"
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

class ModelTraining:
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train):
        logger.info("Initializing LogisticRegression model...")
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        logger.info("Training LogisticRegression model...")
        self.model.fit(X_train, y_train)
        logger.info("Model training completed successfully")

        # Save trained model
        model_path = os.path.join(PREPROCESSED_DIR, "logistic_regression_model.pkl")
        joblib.dump(self.model, model_path)
        logger.info(f"Trained model saved at {model_path}")

        return self.model
