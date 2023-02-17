import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import mlflow
import pandas as pd
import pickle
import time

STAGE = "Transforming" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def transform(config):
    config = read_yaml(config)
    data_path = config["artifacts"]["base_data"]
    
    with open(data_path, "rb") as input_file:
        df = pickle.load(input_file)
    
    x = df.drop(['Geography','Gender' , 'Surname', 'Exited'],axis = 1)
    y = df[ 'Exited']
    logging.info(f"Output are : {set(y)}")
    scalar = StandardScaler()
    scale_value = scalar.fit_transform(x)
    df1 = pd.DataFrame(scale_value)
    logging.info("Scale the Value Successfully")
    df1.columns = x.columns
    size = config["params"]["test_size"]
    seed = config["params"]["random_state"]
    X_train, X_test, y_train, y_test = train_test_split(df1, y, test_size=size, random_state=seed)
    return X_train, X_test, y_train, y_test
    
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        transform(config=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e