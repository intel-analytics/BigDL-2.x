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
import numpy as np
import sys

from bigdl.dataset import mnist
from bigdl.dataset.transformer import *
from bigdl.optim.optimizer import *
from zoo import init_nncontext
from zoo.tfpark import TFOptimizer, TFDataset

from zoo.orca.data.shard import SparkXShards
from zoo.orca.learn.tf.estimator import Estimator

sys.path.append("/tmp/models/slim")  # add the slim library
from nets import lenet

slim = tf.contrib.slim


def accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1, output_type=labels.dtype)
    is_correct = tf.cast(tf.equal(predictions, labels), dtype=tf.float32)
    return tf.reduce_mean(is_correct)


def get_data_xshards(dataset, sc):
    from bigdl.dataset import mnist
    (images_data, labels_data) = mnist.read_data_sets("/tmp/mnist", dataset)
    image_rdd = sc.parallelize(images_data).mapPartitions(lambda iter: [np.array(list(iter))])
    labels_rdd = sc.parallelize(labels_data).mapPartitions(lambda iter: [np.array(list(iter))])
    rdd = image_rdd.zip(labels_rdd) \
        .map(lambda images_labels_tuple:
                       {
                           "x":(images_labels_tuple[0] - mnist.TRAIN_MEAN) / mnist.TRAIN_STD,
                           "y": images_labels_tuple[1]
                       })
    return SparkXShards(rdd)


def main(max_epoch, data_num):
    sc = init_nncontext()

    # get data, pre-process and create TFDataset
    training_shards = get_data_xshards("train", sc)
    testing_shards = get_data_xshards("test", sc)

    images = tf.placeholder(dtype=tf.float32, shape=(None, 28, 28, 1))
    labels = tf.placeholder(dtype=tf.int32, shape=(None,))

    with slim.arg_scope(lenet.lenet_arg_scope()):
        logits, end_points = lenet.lenet(images, num_classes=10, is_training=True)

    loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))

    acc = accuracy(logits, labels)

    # create an estimator
    est = Estimator.from_graph(inputs=images,
                               outputs=logits,
                               labels=labels,
                               loss=loss,
                               optimizer=tf.train.AdamOptimizer(),
                               metrics={"acc": acc})
    est.fit(data=training_shards,
            batch_size=320,
            epochs=max_epoch,
            validation_data=testing_shards)

    result = est.evaluate(testing_shards)
    print(result)

    est.save_tf_checkpoint("/tmp/lenet/model")


if __name__ == '__main__':

    max_epoch = 5
    data_num = 60000

    if len(sys.argv) > 1:
        max_epoch = int(sys.argv[1])
        data_num = int(sys.argv[2])
    main(max_epoch, data_num)
