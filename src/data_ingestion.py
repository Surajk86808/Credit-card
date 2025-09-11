import os
import pandas as pd
from google.cloud import storage
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import RAW_DIRS
from utils.common_functions import read_yaml

logger = get_logger(__name__)


class DataIngestion:
    def __init__(self, config: dict):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.bucket_file_name = self.config["bucket_file_name"]
        os.makedirs(RAW_DIRS, exist_ok=True)

        # Initialize GCP client
        self.client = storage.Client()
        logger.info(f"📂 Initialized GCP client for bucket: {self.bucket_name}")

    def download_csv_from_gcp(self):
        try:
            bucket = self.client.bucket(self.bucket_name)
            for file_name in self.bucket_file_name:
                file_path = os.path.join(RAW_DIRS, file_name)
                logger.info(f"Downloading {file_name} → {file_path}")
                blob = bucket.blob(file_name)
                blob.download_to_filename(file_path)
            logger.info("✅ All files downloaded successfully.")
        except Exception as e:
            logger.error(f"❌ Error downloading from GCP: {e}")
            raise CustomException("Error downloading files from GCP", e)

    def run(self) -> pd.DataFrame:
        try:
            self.download_csv_from_gcp()
            dfs = [pd.read_csv(os.path.join(RAW_DIRS, f)) for f in self.bucket_file_name]
            df = pd.concat(dfs, ignore_index=True)
            logger.info(f"📊 Data ingestion completed: {df.shape}")
            return df
        except Exception as e:
            raise CustomException("Data ingestion failed", e)
