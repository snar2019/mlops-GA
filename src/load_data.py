# read the data from data source
# save it in the data/raw for further process
from get_data import read_params,get_data
#import argparse

def load_and_save(config_path):
    config=read_params(config_path)
    train_df,test_df=get_data(config_path)
    raw_train_data_path=config["load_data"]["raw_train_data_csv"]
    raw_test_data_path=config["load_data"]["raw_test_data_csv"]
    train_df.to_csv(raw_train_data_path, index=False)
    test_df.to_csv(raw_test_data_path,index=False)


if __name__ == "__main__":
    pass
    #args = argparse.ArgumentParser()
    #args.add_argument("--config", default="params.yaml")
    #parsed_args = args.parse_args()
    #load_and_save(config_path=parsed_args.config)