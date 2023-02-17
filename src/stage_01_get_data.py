import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
import os 
import mlflow
import mlflow.keras
import pandas as pd
import pickle


STAGE = "Get Data" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    data_path = config["source_data_dirs"]["data"]
    bool_data = os.path.exists(data_path)
    if  bool_data:
        logging.info("File Present in the given path : " + data_path)
    else:
        logging.error("File is Missing in the path : " + data_path)

    if bool_data:
        df = pd.read_csv(data_path)
        row , column = df.shape
        logging.info(f"Rows : {row} Column : {column}")
        pck_data = config["artifacts"]["base_data"]
        with open(pck_data, "wb") as output_file:
            pickle.dump(df, output_file)
            logging.info("Added Data in the Folder : " + pck_data)



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e