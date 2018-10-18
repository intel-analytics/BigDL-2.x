import tensorflow as tf
import numpy as np
import os
from zoo.pipeline.api.net import TFOptimizer
from bigdl.dataset import mnist
from bigdl.optim.optimizer import Adam, Top1Accuracy
from nets import lenet


slim = tf.contrib.slim

from zoo import init_nncontext
import zoo.data


# os.environ['PYSPARK_SUBMIT_ARGS'] = "--driver-java-options \" -Xdebug -Xrunjdwp:transport=dt_socket,server=y,suspend=y,address=5050\" /home/yang/sources/zoo/pyzoo/zoo/data/example.py"

sc = init_nncontext("TFNet Object Detection Example")


def create_input_fn():
    (images_data, labels_data) = mnist.read_data_sets("/tmp/mnist", "train")
    dataset = zoo.data.Dataset.from_tensor_slices(sc, (images_data, labels_data))
    dataset = dataset.map(lambda x, y: (tf.to_float(x), tf.to_int32(y)))
    dataset = dataset.map(lambda x, y: ((x - mnist.TRAIN_MEAN) / mnist.TRAIN_STD, y))
    dataset = dataset.batch(100)
    return dataset


def create_loss_fn(tensors):
    images, labels = tensors
    with slim.arg_scope(lenet.lenet_arg_scope()):
        logits, end_points = lenet.lenet(images, num_classes=10, is_training=True)
    loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))
    return loss

# create a optimizer
optimizer = TFOptimizer.create(create_loss_fn, create_input_fn, Adam(1e-3))
optimizer.optimize()
# # >> [0, 1, 4, 9, 16]



