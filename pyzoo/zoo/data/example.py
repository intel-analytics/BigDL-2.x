import tensorflow as tf
import numpy as np
import os
os.environ["JAVA_HOME"] = "/home/yang/applications/jdk1.8.0_121"
from zoo import init_nncontext
import zoo.data

sc = init_nncontext("TFNet Object Detection Example")

dataset = zoo.data.Dataset.from_tensor_slices(sc, (np.arange(0, 10), np.arange(0, 10)))
dataset = dataset.map(lambda x, y: tf.multiply(x, y))
dataset = dataset.filter(lambda x: x < 25)

rdd = dataset.as_rdd()

print rdd.collect()
# >> [0, 1, 4, 9, 16]



