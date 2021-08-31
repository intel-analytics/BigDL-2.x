import os
import json

from pyspark.context import SparkContext
from pyspark.conf import SparkConf
import itertools
import tensorflow as tf

#from zoo.orca.learn.tf2.pyspark_estimator import Estimator

sc = SparkContext()

from numpy import array
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:12345"]
    },
    'task': {'type': 'worker', 'index': 0}
})

def transform_func(*args):
    x = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
    y = array([40, 50, 60, 70])

    x = x.reshape((4,3,1))

    strategy =  tf.distribute.experimental.MultiWorkerMirroredStrategy()
    def build_and_compile_model():
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(3, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model
    with strategy.scope():
        mutil_model = build_and_compile_model()
    # checkpoint_dir = './training_checkpoints'
    # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    # callbacks = [
    #     tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    #     tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
    #                                         save_weights_only=True)]
    mutil_model.fit(x, y, epochs=1, verbose=0)

    stats = mutil_model.get_weights()
    return stats

rdd = sc.parallelize([1, 2], 4)

res = rdd.mapPartitions(transform_func).collect()
res = list(itertools.chain.from_iterable(res))
rel = res[0].copy()
print(res)
sc.stop()
