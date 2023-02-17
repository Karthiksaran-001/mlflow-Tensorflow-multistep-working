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


def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    train = datasets.MNIST("../data", train=True, download=True, transform=transform)
    test = datasets.MNIST("../data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test, **test_kwargs)
    pass


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e