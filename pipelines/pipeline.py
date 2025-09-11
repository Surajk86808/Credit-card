import os
import subprocess
import pandas as pd
from src.logger import get_logger
from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessing
from src.model_training import ModelTraining
from src.hyperparameter_tuning import HyperparameterTuning
from src.model_evaluation import ModelEvaluation
from utils.common_functions import read_yaml
from config.paths_config import CONFIG_PATH, RAW_DATA_FILE

# MLflow
import mlflow
import mlflow.sklearn

# Optional: dvclive for Dagshub experiment logging
try:
    import dvclive
    DVCLIVE_AVAILABLE = True
except ImportError:
    DVCLIVE_AVAILABLE = False

logger = get_logger(__name__)


class Pipeline:
    def __init__(self, use_mlflow=False):
        # MLflow flag
        self.use_mlflow = use_mlflow

        # Load config
        config = read_yaml(CONFIG_PATH)

        # Initialize modules
        self.data_ingestion = DataIngestion(config)
        self.data_processing = DataProcessing(data_path=RAW_DATA_FILE)
        self.model_training = ModelTraining()
        self.hyperparameter_tuning = HyperparameterTuning()
        self.model_evaluation = ModelEvaluation()

    def run(self):
        logger.info("🚀 Pipeline started")

        # Step 1: Data ingestion
        df = self.data_ingestion.run()
        logger.info(f"✅ Data ingestion completed: {df.shape}")

        # Step 2: Preprocessing
        X_train, X_test, y_train, y_test = self.data_processing.preprocess(df)
        logger.info("✅ Data preprocessing completed")

        # Step 3: Train model
        model = self.model_training.train(X_train, y_train)
        logger.info("✅ Model training completed")

        # Step 4: Hyperparameter tuning
        best_model = self.hyperparameter_tuning.tune(X_train, y_train)
        logger.info("✅ Hyperparameter tuning completed")

        # Step 5: Evaluation
        results = self.model_evaluation.evaluate(best_model, X_test, y_test)
        logger.info(f"✅ Model evaluation completed: {results}")

        # Step 6: Log metrics to MLflow
        if self.use_mlflow:
            self.log_to_mlflow(results, best_model)
            logger.info("📊 Metrics and model logged to MLflow successfully")

        # Step 7: Log metrics to Dagshub (via dvclive)
        if DVCLIVE_AVAILABLE:
            os.makedirs("artifacts/metrics", exist_ok=True)
            with dvclive.Live("artifacts/metrics") as live:
                live.log_metric("accuracy", results["accuracy"])
                if results["roc_auc"] is not None:
                    live.log_metric("roc_auc", results["roc_auc"])
                live.log_confusion_matrix(
                    labels=["Not Fraud", "Fraud"],
                    matrix=results["confusion_matrix"]
                )
            logger.info("📈 Metrics logged with dvclive for Dagshub")

        # Step 8: Sync artifacts with DVC + Git
        try:
            self.dvc_track_and_push()
        except Exception as e:
            logger.warning(f"⚠️ DVC/Git sync failed but pipeline finished: {e}")

        logger.info("🎯 Pipeline finished successfully")
        return results

    def log_to_mlflow(self, results, best_model):
        """Log model, metrics, and artifacts to MLflow"""
        mlflow.set_experiment("credit_card_fraud_detection")

        with mlflow.start_run():
            # Params
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("max_iter", 1000)

            # Metrics
            mlflow.log_metric("accuracy", results["accuracy"])
            if results["roc_auc"]:
                mlflow.log_metric("roc_auc", results["roc_auc"])

            # Save trained model
            mlflow.sklearn.log_model(best_model, artifact_path="model")

            # Log preprocessing artifacts
            preprocessed_dir = "artifacts/preprocessed"
            for file_name in ["scaler.pkl", "feature_names.pkl"]:
                file_path = os.path.join(preprocessed_dir, file_name)
                if os.path.exists(file_path):
                    mlflow.log_artifact(file_path)

    def dvc_track_and_push(self):
        """Automatically track artifacts and push to DVC remotes (GCS + Dagshub)."""
        dirs_to_track = ["artifacts/raw", "artifacts/preprocessed", "artifacts/metrics"]

        for dir_path in dirs_to_track:
            subprocess.run(["dvc", "add", dir_path], check=True)

        # Add DVC files to Git
        subprocess.run(["git", "add", ".gitignore"], check=True)
        subprocess.run(["git", "add"] + [f"{d}.dvc" for d in dirs_to_track], check=True)

        # Commit changes
        subprocess.run(["git", "commit", "-m", "Auto-update artifacts from pipeline"], check=True)

        # Push artifacts to DVC remotes (GCS + Dagshub)
        subprocess.run(["dvc", "push"], check=True)

        # Push Git commit
        subprocess.run(["git", "push"], check=True)

        logger.info("✅ DVC + Git sync completed successfully")


if __name__ == "__main__":
    pipeline = Pipeline(use_mlflow=True)  # MLflow logging enabled
    pipeline.run()
