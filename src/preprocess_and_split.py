# split the raw data
# save it in data/processed folder
#import argparse
import pandas as pd
import re
import json
from application_logging.logger import App_Logger

import numpy as np
from pandas import json_normalize
from datetime import datetime
from get_data import read_params

# Stages of Data Preprocessing and Data transformation ##

#1.Function for extracting features from date column
def date_process(df):
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d") # seting the column as pandas datetime
    df["weekday"] = df['date'].dt.weekday #extracting week day
    df["day"] = df['date'].dt.day # extracting day
    df["month"] = df['date'].dt.month # extracting month
    df["year"] = df['date'].dt.year # extracting year
    df['visitHour'] = df['visitStartTime'].apply(lambda x: str(datetime.fromtimestamp(x).hour)).astype(int)
    return df

# 2. Function to validate the columns in the dataset for json datatype
def column_validator(df):
    json_columns = []
    for col in df.columns:
        if type(df[col][0]) == str:
            txt = df[col][0]
            if re.search("^{.*}$", txt):
                json_columns.append(col)
    return json_columns    # returning  json columns of dataset
# 2.1 Function for flattening the json columns and merge them with original dataset
def json_to_df(df, json_columns):
    for column in json_columns:
        column_to_df = json_normalize([json.loads(x) for x in df[column]])
        df = df.drop(column, axis=1).merge(column_to_df, right_index=True,left_index=True)  # drop the flattened column from the original dataset
    return df  # returns new dataframe with flattened json columns

# 3.Dropping columns which have more than 50% of null values and not contributing to the target variable
def remove_nan_cols(df):
    for col in df.columns:
        if (df[col].isnull().sum() > (0.5 * len(df))):
            df.drop(col, axis=1, inplace=True)
    return df
#4.Imputation of null values
def impute_na(df):
    for col in df.columns:
        df[col].fillna(0,inplace=True)
    return df

# 5.Changing datatypes from object to desired ones
def data_type_convert(df):
    for col in df.columns:
        if (type(df[col][0]) == str and df[col][0].isdigit()):
            df[col] = df[col].astype(int)
    return df

# 6. Removing columns with constant values or with zero standard deviation
def remove_zero_std_cols(df):
    for column in df.columns:
        if (df[column].nunique() == 1):
            df.drop(column, axis=1, inplace=True)
    return df

# 8 Function to gather categorical columns in the dataset
def categorical_cols(df):
    cat_cols = []
    for col in df.columns:
        if type(df[col][0]) == str or type(df[col][0]) == np.bool_:
            cat_cols.append(col)
    return cat_cols  # returns categorical columns in the dataset

def label_encoding(df,label_cols):
    from sklearn.preprocessing import LabelEncoder
    # creating instance of labelencoder
    labelencoder = LabelEncoder()
    # Assigning numerical values and storing in same column
    for column in label_cols:
        df[column] = labelencoder.fit_transform(df[column].astype(str))
    return df


######### Calling function ###################

def preprocess_and_split(config_path):
    file_object = open('Training_log.txt','a+')
    logger = App_Logger()
    config = read_params(config_path)

    train_data_path = config["split_data"]["train_path"]
    raw_train_data_path = config["load_data"]["raw_train_data_csv"]
    logger.log(file_object,"Training Data load was successful")

    train_df = pd.read_csv(raw_train_data_path,nrows=100000)
    logger.log(file_object,"Data reading successful")

# 1.Function for extracting features from date column
    train_df = date_process(train_df)  # function  for datetime cols processing in train data
    logger.log(file_object, "Datetime Processing in train data completed ")

# 2. Function to validate the columns in the dataset for json datatype
    train_json_columns = column_validator(train_df) # Validating the columns in the train dataset for json datatype
    logger.log(file_object,"Column_validator successful" )

# 2.1 Function for flattening the json columns and merge them with original dataset
    if train_json_columns is not None:
        train_df = json_to_df(train_df,train_json_columns) #Normalizing the json columns in train data
        target=train_df['transactionRevenue']
        logger.log(file_object,"Normalizing the json columns completed")

# 3.Dropping columns which have more than 50% of null values and columns not contributing to the target variable
    train_df= remove_nan_cols(train_df)
    logger.log(file_object,"50% NAN value columns are removed")
    train_df.drop('sessionId', axis=1,inplace=True)  # Removing this column as  it is the  combination of fullVisitorId and visitId
    train_df.drop('visitStartTime', axis=1, inplace=True)  # Removing this column as it is extracted into visitHour
    train_df.drop('fullVisitorId', axis=1,inplace=True)  # This column is very long and of no much contribution towards target variable
    #drop_columns = ['visitId', 'weekday', 'day', 'bounces', 'keyword']
    drop_columns = ['visitId', 'weekday', 'day']
    train_df.drop(drop_columns, axis=1, inplace=True)
    logger.log(file_object, 'Dropped columns which are not contributing to the transaction revenue')



# 4.Imputation of null values
    train_df = pd.concat([train_df, target], axis=1) # transactionRevenue col is attached to the dataframe for imputing nan with 0
    train_df = impute_na(train_df)
    logger.log(file_object, "Imputing NAN values with 0 is completed")


# 5.Changing datatypes from object to desired ones
    train_df = data_type_convert(train_df)
    logger.log(file_object, "Conversion of Datatype to int completed")


# 6. Removing columns with constant values or with zero standard deviation
    train_df = remove_zero_std_cols(train_df)
    logger.log(file_object, "Zero standard deviation columns are removed")


#

# 7 Function to gather categorical columns in the dataset and performing label encoding
    label_cols = categorical_cols(train_df)
    logger.log(file_object, "Gathering of label _cols in train data completed ")

    train_df=label_encoding(train_df,label_cols)
    logger.log(file_object, "Label_encoding in train data completed ")

# 8. Imputing pageviews column with KNNImputer in train data

    from sklearn.impute import KNNImputer
    imputer = KNNImputer()

    imputer_train_df = imputer.fit_transform(train_df[['pageviews']])  ## Imputing pageviews with KNNimputer in training data
    train_df['pageviews'] = imputer_train_df


    logger.log(file_object, "Pageviews column imputed with KNNimputer")
    train_df.to_csv(train_data_path, sep=",", index=False, encoding="utf-8")## Storing Processed train data
    logger.log(file_object,"Traning data is processed and stored as data/processed/train_processed.csv")
    file_object.close()


# Program Entry point#

if __name__ == "__main__":
    pass
    # args = argparse.ArgumentParser()
    # args.add_argument("--config", default="params.yaml")
    # parsed_args = args.parse_args()
    # preprocess_and_split(config_path=parsed_args.config)