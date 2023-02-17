import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
import torch
import os 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch


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

    device_config = {"DEVICE" : 'cuda' if torch.cuda.is_available() else 'cpu'}
    config.update(device_config)
    logging.info("Device is : " + config["DEVICE"])


    if config["DEVICE"] == "cuda":
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_kwargs = {"batch_size": config["params"]["BATCH_SIZE"]}
    test_kwargs = {"batch_size": config["params"]["TEST_BATCH_SIZE"]}
    transform = transforms.Compose([transforms.ToTensor()])
    
    train = datasets.MNIST(config["source_data_dirs"]["data"], train=True, download=True, transform=transform)
    test = datasets.MNIST(config["source_data_dirs"]["data"], train=False, download=True, transform=transform)
    


    train_loader = torch.utils.data.DataLoader(train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test, **test_kwargs)
    logging.info("Added Data in the Folder : " + config["source_data_dirs"]["data"])



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