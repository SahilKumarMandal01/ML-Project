import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    """
    Configuration for data ingestion paths.
    """
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    """
    Handles reading raw data and preparing train/test splits.
    """
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self) -> tuple[str, str]:
        """
        Reads raw dataset, saves raw, train, and test CSV files.

        Returns:
            tuple[str, str]: Paths to train and test CSV files.
        """
        logging.info("Entered the data ingestion method/component")

        try:
            # Load raw dataset
            raw_data_path = os.path.join("notebook", "data", 'stud.csv')
            df = pd.read_csv(raw_data_path)
            logging.info(f"Dataset loaded successfully from {raw_data_path}")

            # Ensure artifacts directory exists
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            # Save raw dataset
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw dataset saved at {self.ingestion_config.raw_data_path}")

            # Train-test split
            logging.info(f"Performing train-test split...")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info(
                f"Data ingestion completed successfully.\n"
                f"Train data: {self.ingestion_config.train_data_path}\n"
                f"Test data: {self.ingestion_config.test_data_path}"
            )

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        
        except Exception as e:
            logging.error("Error occured during data ingestion", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        # Data Ingestion
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()

        # Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

        # Model trainer
        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        
    except Exception as e:
        logging.error("Pipeline execution failed", exc_info=True)
        raise CustomException(e, sys)

