from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import math
import sys
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
from zoo import init_nncontext
from zoo.pipeline.api.net import TFOptimizer, TFDataset
from bigdl.optim.optimizer import *


def get_data_rdd(sc, data, label):
    """Get spark rdd from numpy data

    :param sc: the nncontext of zoo
    :param data: the array of data
    :param label: the array of label
    :return zipped_rdd: zipped rdd with "features" and "labels"
    """
    image_rdd = sc.parallelize(data)
    label_rdd = sc.parallelize(label)
    zipped_rdd = image_rdd.zip(label_rdd)\
        .map(lambda zipped: [np.asarray(zipped[0]), np.asarray(zipped[1])])
    return zipped_rdd


def print_activations(t):
    """Print the operation info"""
    print(t.op.name, ' ', t.get_shape().as_list())


def inference(images):
    """Build the AlexNet model. Directly borrow from Tensorflow tutorial.

    :param images: Images Tensor
    :return pool5: the last Tensor in the convolutional component of AlexNet.
    :return parameters: a list of Tensors corresponding to the weights and
                        biases of the AlexNet model.
    """
    parameters = []
    # conv1
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]

    # lrn1
    with tf.name_scope('lrn1') as scope:
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                  alpha=1e-4,
                                                  beta=0.75,
                                                  depth_radius=2,
                                                  bias=2.0)

    # pool1
    pool1 = tf.nn.max_pool(lrn1,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool1')
    print_activations(pool1)

    # conv2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv2)

    # lrn2
    with tf.name_scope('lrn2') as scope:
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                  alpha=1e-4,
                                                  beta=0.75,
                                                  depth_radius=2,
                                                  bias=2.0)

    # pool2
    pool2 = tf.nn.max_pool(lrn2,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool2')
    print_activations(pool2)

    # conv3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv3)

    # conv4
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv4)

    # conv5
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv5)

    # pool5
    pool5 = tf.nn.max_pool(conv5,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool5')
    print_activations(pool5)

    return pool5, parameters


def main():
    image_size = 120
    image_channel = 3
    n_classes = 10
    n_samples = 28
    n_batch = 28
    n_epoch = 100

    # For sample, use random fake training data to test the model (overfitting)
    data = np.random.randn(n_samples, image_size, image_size, image_channel)
    label = np.random.randint(0, n_classes, size=(n_samples,))

    # Get TFDataset from numpy array
    sc = init_nncontext()
    data_rdd = get_data_rdd(sc, data, label)
    train_dataset = TFDataset.from_rdd(
        data_rdd,
        names=["features", "labels"],
        shapes=[[image_size, image_size, image_channel], []],
        types=[tf.float32, tf.int32],
        batch_size=n_batch)

    # Get holder tensor from TFDataset
    image_holder, label_holder = train_dataset.tensors
    with tf.variable_scope("az_alexnet"):
        pool5, endpoints = inference(image_holder)
        flatten = tf.contrib.layers.flatten(pool5)
        logits = tf.layers.dense(flatten, n_classes)
        loss = tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(
                logits=logits,
                labels=label_holder))

    # Get TFOptimizer and Optimize
    optimizer = TFOptimizer(loss, Adam(1e-3))
    optimizer.optimize(end_trigger=MaxEpoch(n_epoch))
    optimizer.set_train_summary(TrainSummary("./az_alexnet", "summary"))

    # Save the model
    saver = tf.train.Saver()
    saver.save(optimizer.sess, "./az_alexnet/model/")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--batch_size',
    #     type=int,
    #     default=128,
    #     help='Batch size.'
    # )
    # parser.add_argument(
    #     '--num_batches',
    #     type=int,
    #     default=100,
    #     help='Number of batches to run.'
    # )
    # FLAGS, unparsed = parser.parse_known_args()
    # tf.app.run()
    main()
