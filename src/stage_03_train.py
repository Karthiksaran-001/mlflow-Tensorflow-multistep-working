import argparse
import os
import logging
from src.utils.common import read_yaml, get_log_path
from src.stage_02_transform import transform
from src.utils.model_utils import get_model
import tensorflow as tf

STAGE = "Training" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def train_(config):
    config = read_yaml(config)
    config_path = config["artifacts"]["config_path"]
    X_train, X_test, y_train, y_test = transform(config=config_path)
    log_path = config["artifacts"]["callbacks_log"]
    log_dir = get_log_path(log_path)
    file_writer = tf.summary.create_file_writer(logdir = log_dir)
    activation_1 = config["params"]["activation_1"]
    activation_2 = config["params"]["activation_2"]
    Losses = config["params"]["Losses"]
    Optimizer = config["params"]["Optimizer"]
    metrics= config["params"]["Metrics"]
    model = get_model(activation_1 , activation_2)
    weights , biases = model.layers[1].get_weights()  ## becozz first hidden layer get 784 input layer weight 
    logging.info(f"The Weight of the Hidden Layer 1 : {weights.shape}\nThe Biases Of the Hidden Layer 1 : {biases.shape}")
    model.compile(loss = Losses , optimizer = Optimizer , metrics = metrics)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    patience = config["params"]["patience"]
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience = patience , restore_best_weights = True)
    ckpt_path = config["artifacts"]["model_config_dir"]
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(ckpt_path , save_best_only=True)
    CALLBACKS_LIST = [tensorboard_cb, early_stopping_cb, ckpt_cb]
    X_valid , y_valid = X_test[:20] , y_test[:20]
    Validation = (X_valid , y_valid)
    epoch = config["params"]["EPOCHS"]
    history = model.fit(X_train , 
                        y_train , epochs =  epoch,
                        validation_data= Validation ,
                        callbacks = CALLBACKS_LIST)
    test_loss , test_accuracy = model.evaluate(X_test , y_test)
    logging.info(f"Testing Loss : {test_loss} \nTest Accuracy : {test_accuracy}")
    


    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        train_(config=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e