# load the train and test
# train algo
# save the metrices, params
import os
import datetime
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from get_data import read_params
import argparse
import pickle
import mlflow
from urllib.parse import urlparse
from application_logging.logger import App_Logger

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def run_xgb(x1, y_train):
    model_xgb = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators = 1000,learning_rate=0.5,max_depth=8)
    model_xgb.fit(x1, y_train)
    return  model_xgb


def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    model_dir = config["model_dir"]
    file_object = open('Training_log.txt', 'a+')
    logger = App_Logger()

    df=pd.read_csv(train_data_path) #Reading the processed dataset

    df["date"] = pd.to_datetime(df["date"]).dt.date
    X_train = df[df['date'] <= datetime.date(2017, 5, 31)] #splitting the dataset based on date for trainging data
    val_X = df[df['date'] > datetime.date(2017, 5, 31)] #spliting the dataset based on date for validation data
    logger.log(file_object,"Splitting dataset completed")

    X_train = X_train.drop(['date'], axis=1)
    val_X = val_X.drop(['date'], axis=1)

    y_train = np.log1p((X_train["transactionRevenue"]).values)
    val_y = np.log1p((val_X["transactionRevenue"]).values)
    logger.log(file_object, "Log transformation of transaction Revenue values completed")
    x1 = X_train.drop(['transactionRevenue'], axis=1)
    val_x1 = val_X.drop(['transactionRevenue'], axis=1)
    y_train = pd.DataFrame(y_train)
    val_y = pd.DataFrame(val_y)

    ################## MLFLOW ######################
    mlflow_config=config["mlflow_config"]
    remote_server_uri= mlflow_config['remote_server_uri']
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run :
        model_xgb= run_xgb(x1, y_train)
        y_train_predict = model_xgb.predict(x1)
        rmse,mae,r2= eval_metrics(y_train, y_train_predict)

        mlflow.log_param("n_estimators",1200)
        mlflow.log_param("learning_rate",0.5)
        mlflow.log_param("max_depth",8)

        mlflow.log_metric('rmse',rmse)
        mlflow.log_metric("mae",mae)
        mlflow.log_metric("r2",r2)

        tracking_url_type_store=urlparse(mlflow.get_artifact_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                model_xgb,
                "model",
                registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.sklearn.load_model(model_xgb, "model")

    ##################### Saving the model as pickle file ################################
    logger.log(file_object, "Model file created successfully")
    file_object.close()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)