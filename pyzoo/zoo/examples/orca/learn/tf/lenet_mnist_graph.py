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
import argparse

import tensorflow as tf
import tensorflow_datasets as tfds
import sys

from zoo.orca.learn.tf.estimator import Estimator
from zoo.orca import init_orca_context, stop_orca_context


def accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1, output_type=labels.dtype)
    is_correct = tf.cast(tf.equal(predictions, labels), dtype=tf.float32)
    return tf.reduce_mean(is_correct)


def lenet(images, is_training):
    with tf.variable_scope('LeNet', [images]):
        net = tf.layers.conv2d(images, 32, (5, 5), activation=tf.nn.relu, name='conv1')
        net = tf.layers.max_pooling2d(net, (2, 2), 2, name='pool1')
        net = tf.layers.conv2d(net, 64, (5, 5), activation=tf.nn.relu, name='conv2')
        net = tf.layers.max_pooling2d(net, (2, 2), 2, name='pool2')
        net = tf.layers.flatten(net)
        net = tf.layers.dense(net, 1024, activation=tf.nn.relu, name='fc3')
        net = tf.layers.dropout(
            net, 0.5, training=is_training, name='dropout3')
        logits = tf.layers.dense(net, 10)
        return logits


def main(max_epoch):

    # get DataSet
    mnist_train = tfds.load(name="mnist", split="train")
    mnist_test = tfds.load(name="mnist", split="test")

    # Normalizes images
    def normalize_img(data):
        data['image'] = tf.cast(data["image"], tf.float32) / 255.
        return data

    mnist_train = mnist_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    mnist_test = mnist_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # tensorflow inputs
    images = tf.placeholder(dtype=tf.float32, shape=(None, 28, 28, 1))
    # tensorflow labels
    labels = tf.placeholder(dtype=tf.int32, shape=(None,))

    is_training = tf.placeholder_with_default(False, shape=())

    logits = lenet(images, is_training=is_training)

    loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))

    acc = accuracy(logits, labels)

    # create an estimator
    est = Estimator.from_graph(inputs=images,
                               outputs=logits,
                               labels=labels,
                               loss=loss,
                               optimizer=tf.train.AdamOptimizer(),
                               metrics={"acc": acc})
    est.fit(data=mnist_train,
            batch_size=320,
            epochs=max_epoch,
            validation_data=mnist_test,
            feed_dict={is_training: (True, False)})

    result = est.evaluate(mnist_test)
    print(result)

    est.save_tf_checkpoint("/tmp/lenet/model")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The mode for the Spark cluster.')
    parser.add_argument("--num_nodes", type=int, default=1,
                        help="The number of nodes to be used in the cluster. "
                             "You can change it depending on your own cluster setting.")
    parser.add_argument("--cores", type=int, default=4,
                        help="The number of cpu cores you want to use on each node. "
                             "You can change it depending on your own cluster setting.")
    parser.add_argument("--memory", type=str, default="10g",
                        help="The memory you want to use on each node. "
                             "You can change it depending on your own cluster setting.")

    parser.add_argument("--max_epoch", type=int, default=5)

    args = parser.parse_args()
    init_orca_context(cluster_mode=args.cluster_mode, cores=args.cores,
                      num_nodes=args.num_nodes, memory=args.memory)
    main(args.max_epoch)
    stop_orca_context()
