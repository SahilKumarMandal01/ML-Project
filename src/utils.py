import os
import sys
import pickle
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path: str, obj: Any) -> None:
    """
    Save a Python object to a file using pickle.

    Parameters:
        file_path (str): Path to save the object.
        obj (Any): Python object to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str) -> Any:
    """
    Load a Python object from a pickle file.

    Parameters:
        - file_path (str): Path to the pickle file.
    
    Returns:
        - Any: The loaded Pyhon object.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

