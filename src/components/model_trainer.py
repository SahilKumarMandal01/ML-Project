import os
import sys
from dataclasses import dataclass
from typing import Dict, Any

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    """Configuration for saving the trained model."""
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    """Handles training and selection of the best regressor model."""

    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array) -> float:
        """
        Trains multiple regression models, tunes hyperparameters, and saves the best model.

        Args:
            train_array (np.ndarray): Training dataset (features + target as last column).
            test_array (np.ndarray): Testing dataset (features + target as last column).
        
        Returns:
            float: R2 score of the best model on the test set.
        """
        try:
            logging.info("Splitting training and testing datasets into features and targets")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define models
            models: Dict[str, Any] = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            # Define hyperparameters
            params: Dict[str, Dict[str, Any]] = {
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"]
                },
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "CatBoost Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                }
            }

            # Evaluate models
            logging.info("Evaluating models with hyperparameter tuning")
            model_report: Dict[str, float] = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            # Select best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best modl found: {best_model_name} with R2 score = {best_model_score:.4f}")

            if best_model_score < 0.6:
                raise CustomException("No model found with acceptable performance")
            
            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Best model saved at {self.model_trainer_config.trained_model_file_path}")

            # Evaluating final test performance
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            logging.info(f"Final R2 score on test set: {r2_square:.4f}")
            return r2_square
        
        except Exception as e:
            logging.error("Error occured during model training", exc_info=True)
            raise CustomException(e, sys)