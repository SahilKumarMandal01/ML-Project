import os
import sys
import pickle
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path: str, obj: Any) -> None:
    """
    Save a Python object to disk using pickle.

    Args:
        file_path (str): Path where the object will be saved.
        obj(Any): The Python object to save.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        
        logging.info(f"Object saved successfully at {file_path}")
    
    except Exception as e:
        logging.error("Error occured while saving object", exc_info=True)
        raise CustomException(e, sys)
    

def load_object(file_path: str) -> Any:
    """
    Load a Python object from disk using pickle.

    Args:
        file_path (str): Path to the pickle file.
    
    Returns: 
        Any: Loaded Python object.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.error("Error occured while loading object", exc_info=True)
        raise CustomException(e, sys)
    

def evaluate_models(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        models: Dict[str, Any],
        param: Dict[str, Dict[str, Any]]
) -> Dict[str, float]:
    """
    Evaluate multiple models with hyperparameter tuning using GridSearchCV.

    Args:
        X_train (np.ndarray): Training fetaure set.
        y_train (np.ndarray): Training target set.
        X_test (np.ndarray): Testing fetaure set.
        y_test (np.ndarray): Testing target set.
        models (Dict[str, Any]): Dictionary of models.
        param (Dict[str, Dict[str, Any]]): Hyperparameter grid for each model.
    
    Returns:
        Dict[str, float]: Dictionary containing test R2 scores for each model.
    """
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")

            gs = GridSearchCV(model, param[model_name], cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            logging.info(f"{model_name} - Train R2: {train_score:.4f}, Test R2: {test_score:.4f}")

            report[model_name] = test_score
        
        return report
    
    except Exception as e:
        logging.error("Error occured during model evaluation", exc_info=True)
        raise CustomException(e, sys)
