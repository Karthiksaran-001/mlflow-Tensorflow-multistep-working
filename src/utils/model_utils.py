import tensorflow as tf
import logging

def get_model(act_1 , act_2):
    model = tf.keras.Sequential([
                             tf.keras.layers.Dense(20 , input_shape=(10 ,) , activation=act_1 ),
                            tf.keras.layers.Dense(215 , activation=act_1 ),
                              tf.keras.layers.Dense(1 , activation=act_2 )])
    logging.info("Model is Defined")
    return model
