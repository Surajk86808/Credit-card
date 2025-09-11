import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.logger import get_logger
from google.cloud import storage

logger = get_logger(__name__)

PREPROCESSED_DIR = "artifacts/preprocessed"
os.makedirs(PREPROCESSED_DIR, exist_ok=True)


class DataProcessing:
    def __init__(self, data_path: str, test_size: float = 0.2, random_state: int = 42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.df = None

        # Initialize GCP client to upload preprocessed data
        self.client = storage.Client()
        self.bucket_name = "project_878787"  # Replace if dynamic from config
        self.gcp_dir = "preprocessed"

    def load_data(self):
        logger.info(f"Loading dataset from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Dataset loaded successfully with shape {self.df.shape}")
        return self.df

    def preprocess(self, df=None):
        logger.info("Starting preprocessing pipeline...")
        if df is not None:
            self.df = df
        elif self.df is None:
            self.df = self.load_data()

        # Drop NA values
        before = self.df.shape
        self.df = self.df.dropna()
        after = self.df.shape
        logger.info(f"Dropped NA values: {before[0] - after[0]} rows removed")

        # Encode categorical columns
        for col in self.df.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            logger.info(f"Encoded categorical column: {col}")

        # Split features and target
        X = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1]
        feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        logger.info(f"Split data: Train={X_train.shape}, Test={X_test.shape}")

        # Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        logger.info("Applied StandardScaler to features")

        # Save locally
        dataset_path = os.path.join(PREPROCESSED_DIR, "dataset.pkl")
        joblib.dump((X_train, X_test, y_train, y_test), dataset_path)
        joblib.dump(scaler, os.path.join(PREPROCESSED_DIR, "scaler.pkl"))
        joblib.dump(feature_names, os.path.join(PREPROCESSED_DIR, "feature_names.pkl"))
        logger.info(f"Saved preprocessed dataset, scaler & feature names to {PREPROCESSED_DIR}")

        # Upload to GCP
        self.upload_to_gcp(dataset_path)
        return X_train, X_test, y_train, y_test

    def upload_to_gcp(self, local_path):
        try:
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(os.path.join(self.gcp_dir, os.path.basename(local_path)))
            blob.upload_from_filename(local_path)
            logger.info(f"☁️ Preprocessed dataset uploaded to GCP: gs://{self.bucket_name}/{self.gcp_dir}")
        except Exception as e:
            logger.error(f"❌ Failed to upload preprocessed data to GCP: {e}")
