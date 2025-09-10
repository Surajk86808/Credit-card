import os
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from src.logger import get_logger

logger = get_logger(__name__)

PREPROCESSED_DIR = "artifacts/preprocessed"
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

class ModelEvaluation:
    @staticmethod
    def evaluate(model, X_test, y_test):
        logger.info("Evaluating model performance...")

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Try ROC AUC (only if predict_proba exists)
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

        # Save evaluation results
        results = {
            "accuracy": acc,
            "roc_auc": auc,
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        }
        results_path = os.path.join(PREPROCESSED_DIR, "evaluation_results.pkl")
        joblib.dump(results, results_path)
        logger.info(f"Evaluation results saved at {results_path}")

        return results
