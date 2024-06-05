import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.Hearthealthpredictor.logger import logging
from src.Hearthealthpredictor.exception import customexception

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e, sys)
    
def evaluate_model(true, pred, metric):
    try:
        if metric == 'r2':
            score = r2_score(true, pred)
            mse = mean_squared_error(true, pred)
            mae = mean_absolute_error(true, pred)
            precision = None
            recall = None
            f1 = None
        else:  # classification metrics
            score = accuracy_score(true, pred)
            mse = None
            mae = None
            precision = precision_score(true, pred)
            recall = recall_score(true, pred)
            f1 = f1_score(true, pred)

        return score, mse, mae, precision, recall, f1

    except Exception as e:
        logging.info('Exception occurred during model evaluation')
        raise customexception(e, sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occurred in load_object function utils')
        raise customexception(e,sys)
