import json

from pyspark import BarrierTaskContext
from pyspark.context import SparkContext
import tensorflow as tf
from numpy import array
from contextlib import closing
import socket

def model_creator():
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import Dense
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(3, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def find_free_port(tc):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        tc.barrier()
        return f"{s.getsockname()[0]}:{s.getsockname()[1]}"

def train_data_creator(*args):
    train_data = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
    train_data = train_data.reshape((4,3,1))
    return train_data

def val_data_creator(*args):
    val_data = array([40, 50, 60, 70])
    return val_data

def transform_func(parent_iter):
    tc = BarrierTaskContext().get()
    rank = tc.partitionId()
    free_port = find_free_port(tc)
    cluster = tc.allGather(str(free_port))
    print(cluster)

    import os
    os.environ["TF_CONFIG"] = json.dumps({
      'cluster': {
          'worker': cluster
      },
      'task': {'type': 'worker', 'index': rank}
    })
    ips = set([node.split(":")[0] for node in cluster])
    os.environ["no_proxy"] = ",".join(ips)

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    with strategy.scope():
        model = model_creator()

    train_data = train_data_creator()
    val_data = val_data_creator()
    model.fit(train_data, val_data, epochs=10, verbose=1)

    return [model.get_weights()]

if __name__ == "__main__":

    import numpy as np

    sc = SparkContext()
    sparkRDD = sc.parallelize(list(range(40)), 4)
    workerRDD = sparkRDD.repartition(2)
    res = workerRDD.barrier().mapPartitions(transform_func).collect()
    assert np.all(res[0][0] == res[1][0])
    sc.stop()