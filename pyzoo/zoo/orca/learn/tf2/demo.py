import os
import json

from pyspark.context import SparkContext
from pyspark.conf import SparkConf
import tensorflow as tf

#from zoo.orca.learn.tf2.pyspark_estimator import Estimator

sc = SparkContext()

from numpy import array
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense 

workers = 2

def build_and_compile_model():
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(3, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["172.168.0.204:20000", "172.168.0.202:20001"]
    },
    'task': {'type': 'worker', 'index': 0}
})

urls =["172.168.0.204:20000", "172.168.0.202:20001"]
no_proxy = os.environ.get("no_proxy", "")
ips = [url.split(":")[0] for url in urls]
os.environ["no_proxy"] = ",".join(ips) + "," + no_proxy

strategy =  tf.distribute.experimental.MultiWorkerMirroredStrategy()
print('Number of devices: %d' % strategy.num_replicas_in_sync)

def transform_func(*args):
    train_data = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
    val_data = array([40, 50, 60, 70])
    train_data = train_data.reshape((4,3,1))

    with strategy.scope():
        mutil_model = build_and_compile_model()

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                            save_weights_only=True)] 

    mutil_model.fit(train_data, val_data, epochs=10, verbose=0, callbacks=callbacks)

    weights = mutil_model.get_weights()
    return [weights]

workerRDD = sc.parallelize(list(range(workers)), workers)
res = workerRDD.mapPartitions(transform_func).collect()

print(res)

sc.stop()
