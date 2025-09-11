import os
import mlflow
import mlflow.sklearn
from pipelines.pipeline import Pipeline

# Set MLflow experiment name
mlflow.set_experiment("credit_card_fraud_detection")

def run_pipeline_with_mlflow():
    pipeline = Pipeline(use_mlflow=True)

    # Start MLflow run
    with mlflow.start_run():
        # Run the pipeline
        results = pipeline.run()

        # Log model type and hyperparameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)

        # Log metrics
        mlflow.log_metric("accuracy", results["accuracy"])
        if results.get("roc_auc"):
            mlflow.log_metric("roc_auc", results["roc_auc"])

        # Log trained model (from hyperparameter tuning)
        X_train, _, y_train, _ = pipeline.data_processing.preprocess()
        best_model = pipeline.hyperparameter_tuning.tune(X_train, y_train)
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        # Log preprocessing artifacts
        preprocessed_dir = "artifacts/preprocessed"
        for file_name in ["scaler.pkl", "feature_names.pkl", "evaluation_results.pkl"]:
            file_path = os.path.join(preprocessed_dir, file_name)
            if os.path.exists(file_path):
                mlflow.log_artifact(file_path)

    print("✅ Pipeline run logged to MLflow successfully!")


if __name__ == "__main__":
    run_pipeline_with_mlflow()
