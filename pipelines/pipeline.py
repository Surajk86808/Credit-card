import os
import pandas as pd
from src.logger import get_logger
from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessing
from src.model_training import ModelTraining
from src.hyperparameter_tuning import HyperparameterTuning
from src.model_evaluation import ModelEvaluation
from utils.common_functions import read_yaml
from config.paths_config import CONFIG_PATH, RAW_DATA_FILE

logger = get_logger(__name__)

class Pipeline:
    def __init__(self):
        config = read_yaml(CONFIG_PATH)  # ✅ Pass config to DataIngestion
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

        logger.info("🎯 Pipeline finished successfully")
        return results

if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run()
