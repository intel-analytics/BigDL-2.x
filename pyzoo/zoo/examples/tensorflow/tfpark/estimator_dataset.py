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
import sys

import tensorflow as tf
import numpy as np

from bigdl.optim.optimizer import Top1Accuracy
from zoo import init_nncontext
from zoo.tfpark import KerasModel, TFDataset
from zoo.tfpark.estimator import Estimator, EstimatorSpec


def get_data_rdd(dataset, sc):
    from bigdl.dataset import mnist
    (images_data, labels_data) = mnist.read_data_sets("/tmp/mnist", dataset)
    image_rdd = sc.parallelize(images_data)
    labels_rdd = sc.parallelize(labels_data)
    rdd = image_rdd.zip(labels_rdd) \
        .map(lambda rec_tuple: ((rec_tuple[0] - mnist.TRAIN_MEAN) / mnist.TRAIN_STD,
                                np.array(rec_tuple[1])))
    return rdd


def main(max_epoch):
    sc = init_nncontext()

    def model_fn(features, labels, mode):
        from nets import lenet
        slim = tf.contrib.slim
        with slim.arg_scope(lenet.lenet_arg_scope()):
            logits, end_points = lenet.lenet(features, num_classes=10, is_training=True)

        loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))
        return EstimatorSpec(mode, predictions=logits, loss=loss)

    def input_fn(mode):
        if mode == tf.estimator.ModeKeys.TRAIN:
            training_rdd = get_data_rdd("train", sc)
            dataset = TFDataset.from_rdd(training_rdd,
                                         features=(tf.float32, [28, 28, 1]),
                                         labels=(tf.int32, []),
                                         batch_size=320)
        elif mode == tf.estimator.ModeKeys.EVAL:
            testing_rdd = get_data_rdd("test", sc)
            dataset = TFDataset.from_rdd(testing_rdd,
                                         features=(tf.float32, [28, 28, 1]),
                                         labels=(tf.int32, []),
                                         batch_size=320)
        else:
            testing_rdd = get_data_rdd("test", sc).map()
            dataset = TFDataset.from_rdd(testing_rdd,
                                         features=(tf.float32, [28, 28, 1]),
                                         batch_size=320)

        return dataset
    estimator = Estimator(model_fn, tf.train.AdamOptimizer(), model_dir="/tmp/estimator")

    estimator.train(input_fn, steps=60000//320*5)

    result = estimator.evaluate(input_fn, [Top1Accuracy()])
    print(result)

if __name__ == '__main__':

    max_epoch = 5

    if len(sys.argv) > 1:
        max_epoch = int(sys.argv[1])
    main(max_epoch)
