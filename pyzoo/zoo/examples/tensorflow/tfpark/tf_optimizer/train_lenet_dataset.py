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
import tensorflow as tf
from zoo import init_nncontext
from zoo.tfpark import TFOptimizer, TFDataset
from bigdl.optim.optimizer import *
import numpy as np
import sys

from bigdl.dataset import mnist
from bigdl.dataset.transformer import *

sys.path.append("/tmp/models/slim")  # add the slim library
from nets import lenet

slim = tf.contrib.slim

import tensorflow_datasets as tfds

BATCH_SIZE=1024
def main(max_epoch, data_num):
    sc = init_nncontext()

    # (images_data, labels_data) = mnist.read_data_sets("/tmp/mnist", "train")
    #
    # images_data = (images_data - mnist.TRAIN_MEAN) / mnist.TRAIN_STD
    #
    # images_data = images_data[:1000]
    # labels_data = labels_data[:1000]

    # tfds.image.MNIST.as_dataset()

    def get_dataset(split):
        dataset = tfds.load(name="mnist", split=split)
        dataset = dataset.map(lambda data: (((tf.to_float(data['image']) - mnist.TRAIN_MEAN)/mnist.TRAIN_STD), data['label']))
        return dataset

    train_dataset = get_dataset("train")
    test_dataset = get_dataset("test")

    tf_dataset = TFDataset.from_tf_data_dataset(train_dataset, data_count=int(60000/BATCH_SIZE), batch_size=BATCH_SIZE, validation_dataset=test_dataset, validation_data_count=10000)

    # construct the model from TFDataset
    images, labels = tf_dataset.tensors

    with slim.arg_scope(lenet.lenet_arg_scope()):
        logits, end_points = lenet.lenet(images, num_classes=10, is_training=True)

    loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=tf.to_int32(labels)))

    # create a optimizer
    optimizer = TFOptimizer.from_loss(loss, Adam(1e-3),
                                      val_outputs=[logits],
                                      val_labels=[labels],
                                      val_method=Top1Accuracy(), model_dir="/tmp/lenet/")
    # kick off training
    optimizer.optimize(end_trigger=MaxEpoch(max_epoch))

    saver = tf.train.Saver()
    saver.save(optimizer.sess, "/tmp/lenet/model")

if __name__ == '__main__':

    max_epoch = 5
    data_num = 60000

    if len(sys.argv) > 1:
        max_epoch = int(sys.argv[1])
        data_num = int(sys.argv[2])
    main(max_epoch, data_num)
