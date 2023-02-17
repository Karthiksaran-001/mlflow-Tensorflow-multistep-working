import argparse
import os
import logging
from src.utils.common import read_yaml, get_log_path
from src.stage_02_transform import transform
from src.utils.model_utils import get_model
import tensorflow as tf
import mlflow

STAGE = "Training" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def evaluate_(config):
    config = read_yaml(config)
    config_path = config["artifacts"]["config_path"]
    X_train, X_test, y_train, y_test = transform(config_path)
    ckpt_path = config["artifacts"]["model_config_dir"]
    model = tf.keras.models.load_model(ckpt_path)



    with mlflow.start_run(run_name="training") as run:
        logging.info("Start Run MLFLOW")
        mlflow.log_params(config["params"])
        loss , acc = model.evaluate(X_test , y_test)
        mlflow.log_metric(f"Binary Cross Entropy_" , loss)
        mlflow.log_metric(f"Predicted Accuracy is_" , acc)
        mlflow.keras.log_model(model , "model")
        model_path = mlflow.get_artifact_uri("model")
        loaded_tf_model = mlflow.keras.load_model(model_path)
        loaded_tf_model.predict(X_test)



    


    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        evaluate_(config=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e