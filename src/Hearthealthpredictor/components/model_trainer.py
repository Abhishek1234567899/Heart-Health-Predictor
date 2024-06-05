import os
import sys
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor
from src.Hearthealthpredictor.logger import logging
from src.Hearthealthpredictor.exception import customexception
from dataclasses import dataclass
from src.Hearthealthpredictor.utils.utils import save_object, evaluate_model


@dataclass 
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            regression_models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'RandomForestRegressor': RandomForestRegressor(),
                'XGBoost': XGBRegressor()
            }
            
            classification_models = {
                'LogisticRegression': LogisticRegression(),
                'KNN': KNeighborsClassifier(),
                'NaiveBayes': GaussianNB()
            }

            # Evaluate regression models
            regression_report = {model_name: evaluate_model(X_train, y_train, X_test, y_test, model) 
                                 for model_name, model in regression_models.items()}
            
            # Evaluate classification models
            classification_report = {model_name: evaluate_model(X_train, y_train, X_test, y_test, model) 
                                      for model_name, model in classification_models.items()}

            print("Regression Model Report:")
            print(regression_report)
            print("\nClassification Model Report:")
            print(classification_report)

            # Find the best regression model
            best_regression_model = max(regression_report, key=regression_report.get)
            best_regression_score = regression_report[best_regression_model]

            # Find the best classification model
            best_classification_model = max(classification_report, key=classification_report.get)
            best_classification_score = classification_report[best_classification_model]

            print(f"Best Regression Model: {best_regression_model}, R-squared: {best_regression_score}")
            print(f"Best Classification Model: {best_classification_model}, Accuracy: {best_classification_score}")

            # Save the best model
            if best_regression_score > best_classification_score:
                save_object(self.model_trainer_config.trained_model_file_path, regression_models[best_regression_model])
            else:
                save_object(self.model_trainer_config.trained_model_file_path, classification_models[best_classification_model])
          
        except Exception as e:
            logging.error('Exception occurred during Model Training')
            raise customexception(e, sys)
