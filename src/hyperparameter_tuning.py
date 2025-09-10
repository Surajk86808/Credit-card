import os
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from src.logger import get_logger

logger = get_logger(__name__)

PREPROCESSED_DIR = "artifacts/preprocessed"
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

class HyperparameterTuning:
    def __init__(self, model=None):
        self.model = model or LogisticRegression(max_iter=1000, random_state=42)

    def tune(self, X_train, y_train):
        logger.info("Starting hyperparameter tuning for LogisticRegression...")

        param_grid = {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],   # 'l1' requires solver='liblinear'
            "solver": ["lbfgs", "liblinear"]
        }

        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1
        )

        logger.info("Running GridSearchCV...")
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        logger.info(f"Best Parameters: {grid_search.best_params_}")
        logger.info(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

        # Save tuned model
        model_path = os.path.join(PREPROCESSED_DIR, "logistic_regression_best.pkl")
        joblib.dump(best_model, model_path)
        logger.info(f"Best tuned model saved at {model_path}")

        return best_model
