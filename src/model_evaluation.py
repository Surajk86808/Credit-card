import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from src.logger import get_logger
from google.cloud import storage

logger = get_logger(__name__)

PREPROCESSED_DIR = "artifacts/preprocessed"
REPORTS_DIR = "artifacts/reports"
os.makedirs(PREPROCESSED_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


class ModelEvaluation:
    def __init__(self):
        self.client = storage.Client()
        self.bucket_name = "project_878787"
        self.gcp_dir = "reports"

    def evaluate(self, model, X_test, y_test):
        logger.info("Evaluating model performance...")

        # Predictions
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Try ROC AUC if available
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = None

        logger.info(f"Accuracy: {acc:.4f}")
        if auc is not None:
            logger.info(f"ROC-AUC: {auc:.4f}")
        logger.info(f"Classification Report:\n{report}")
        logger.info(f"Confusion Matrix:\n{cm}")

        # Save results locally
        results = {
            "accuracy": acc,
            "roc_auc": auc,
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        }
        results_path = os.path.join(PREPROCESSED_DIR, "evaluation_results.pkl")
        joblib.dump(results, results_path)
        logger.info(f"✅ Evaluation results saved locally at {results_path}")

        # Save human-readable reports
        report_path = os.path.join(REPORTS_DIR, "classification_report.txt")
        cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.csv")
        with open(report_path, "w") as f:
            f.write(report)
        pd.DataFrame(cm).to_csv(cm_path, index=False)
        logger.info(f"📊 Reports saved locally in {REPORTS_DIR}")

        # Upload reports to GCP
        try:
            bucket = self.client.bucket(self.bucket_name)
            for file in [report_path, cm_path]:
                blob = bucket.blob(os.path.join(self.gcp_dir, os.path.basename(file)))
                blob.upload_from_filename(file)
            logger.info(f"☁️ Evaluation reports uploaded to GCP: gs://{self.bucket_name}/{self.gcp_dir}")
        except Exception as e:
            logger.error(f"❌ Failed to upload evaluation reports to GCP: {e}")

        return results
