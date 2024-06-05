import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
import pickle
from src.utils.utils import load_object
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from src.logger.logging import logging
from src.exception import customexception

class ModelEvaluation:
    def __init__(self):
        logging.info("Evaluation started")

    def eval_regression_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        logging.info("Regression evaluation metrics captured")
        return rmse, mae, r2

    def eval_classification_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)
        logging.info("Classification evaluation metrics captured")
        return accuracy, precision, recall, f1

    def initiate_model_evaluation(self, train_array, test_array, is_classification=False):
        try:
             X_test, y_test = (test_array[:, :-1], test_array[:, -1])

             model_path = os.path.join("artifacts", "model.pkl")
             model = load_object(model_path)

             logging.info("Model has been loaded")

             tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

             print(tracking_url_type_store)

             with mlflow.start_run():

                prediction = model.predict(X_test)

                if is_classification:
                    accuracy, precision, recall, f1 = self.eval_classification_metrics(y_test, prediction)
                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_metric("precision", precision)
                    mlflow.log_metric("recall", recall)
                    mlflow.log_metric("f1", f1)
                    logging.info("Classification evaluation metrics logged")
                else:
                    rmse, mae, r2 = self.eval_regression_metrics(y_test, prediction)
                    mlflow.log_metric("rmse", rmse)
                    mlflow.log_metric("r2", r2)
                    mlflow.log_metric("mae", mae)
                    logging.info("Regression evaluation metrics logged")

                 # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model, "model")

        except Exception as e:
            raise customexception(e, sys)
