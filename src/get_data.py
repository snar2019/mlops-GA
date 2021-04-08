import yaml
import pandas as pd
#import argparse

def read_params(config_path):
    with open(config_path) as yaml_file:
        config=yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    config=read_params(config_path)
    train_data_path=config['data_source']['train_csv']
    test_data_path=config['data_source']['test_csv']
    train_df = pd.read_csv(train_data_path)
    test_df=pd.read_csv(test_data_path)
    return (train_df,test_df)


if __name__ == "__main__":
    pass
     #args=argparse.ArgumentParser()
     #args.add_argument("--config",default="params.yaml")
     #parsed_args = args.parse_args()
     #train_df,test_df=get_data(config_path=parsed_args.config)
