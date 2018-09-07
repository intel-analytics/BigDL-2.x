#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import heapq

import tensorflow as tf
from zoo import init_nncontext
from zoo.pipeline.api.net import TFOptimizer, TFDataset
from bigdl.optim.optimizer import MaxIteration, Adam, MaxEpoch, TrainSummary
import numpy as np
import sys

from bigdl.dataset import mnist
from bigdl.dataset.transformer import *

sys.path.append("/tmp/models/slim")  # add the slim library
from nets import lenet

slim = tf.contrib.slim


def main():
    sc = init_nncontext()

    # get data, pre-process and create TFDataset
    (images_data, labels_data) = mnist.read_data_sets("/tmp/mnist", "train")
    image_rdd = sc.parallelize(images_data)
    labels_rdd = sc.parallelize(labels_data)
    rdd = image_rdd.zip(labels_rdd) \
        .map(lambda rec_tuple: [normalizer(rec_tuple[0], mnist.TRAIN_MEAN, mnist.TRAIN_STD),
                                np.array(rec_tuple[1])])

    dataset = TFDataset.from_rdd(rdd,
                                 names=["features", "labels"],
                                 shapes=[[28, 28, 1], [1]],
                                 types=[tf.float32, tf.int32],
                                 batch_size=280
                                 )

    # construct the model from TFDataset
    images, labels = dataset.tensors

    labels = tf.squeeze(labels)

    with slim.arg_scope(lenet.lenet_arg_scope()):
        logits, end_points = lenet.lenet(images, num_classes=10, is_training=True)

    loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))

    # create a optimizer
    optimizer = TFOptimizer(loss, Adam(1e-3))
    optimizer.set_train_summary(TrainSummary("/tmp/az_lenet", "lenet"))
    # kick off training
    for i in range(5):
        optimizer.optimize(end_trigger=MaxEpoch(i + 1))

    saver = tf.train.Saver()
    saver.save(optimizer.sess, "/tmp/lenet/")

if __name__ == '__main__':
    main()
