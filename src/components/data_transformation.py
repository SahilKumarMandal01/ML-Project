import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    """Configuration for data transformation."""
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    """Handles preprocessing and transformation of dataset for ML models."""
    def __init__(self):
        self.config = DataTransformationConfig()
    
    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Creates and returns a preprocessing object for numerical and categorical features.

        Returns
            - ColumnTransformer: A transformed object with numerical and categorical pipelines.
        """
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                "gender",
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            # Pipeline for numerical features
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )

            # Pipeline for categorical features
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                    ("Scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Created numerical and categorical pipelines.")
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            # Combine pipelines into a single preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Reads training and testing data, applies preprocessing,
        and saves the preprocessing object.

        Parameters:
            - train_path: str = Path to the training dataset.
            - test_path: str = Patht to the testing dataset.
        
        Returns:
            - Preprocessed training array
            - Preprocessed testing array
            - File path of the saved preprocessing object
        """
        try:
            # Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Successfully read training and testing data.")

            # Get preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'

            # Split input and target features
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing object on train and test dataframes.")

            # Transform features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine transformed features with target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessing object
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info(f"Preprocessing object saved at {self.config.preprocessor_obj_file_path}")

            return train_arr, test_arr, self.config.preprocessor_obj_file_path
        
        except Exception as e:
            raise CustomException(e, sys)