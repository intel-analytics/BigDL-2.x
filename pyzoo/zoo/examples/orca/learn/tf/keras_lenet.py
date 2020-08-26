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
from zoo import init_nncontext
from zoo.orca.data.shard import SparkXShards
from zoo.orca.learn.tf.estimator import Estimator


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


def main(max_epoch):
    sc = init_nncontext()

    training_shards = get_data_xshards("train", sc)
    testing_shards = get_data_xshards("test", sc)

    model = tf.keras.Sequential(
        [tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
         tf.keras.layers.Dense(64, activation='relu'),
         tf.keras.layers.Dense(64, activation='relu'),
         tf.keras.layers.Dense(10, activation='softmax'),
         ]
    )

    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    est = Estimator.from_keras(keras_model=model)
    est.fit(data=training_shards,
            batch_size=320,
            epochs=max_epoch,
            validation_data=testing_shards)

    result = est.evaluate(testing_shards)
    print(result)
    # >> [0.08865142822265625, 0.9722]

    # the following assert is used for internal testing
    assert result['acc Top1Accuracy'] > 0.95

    est.save_keras_model("/tmp/mnist_keras.h5")


if __name__ == '__main__':

    max_epoch = 5

    if len(sys.argv) > 1:
        max_epoch = int(sys.argv[1])
    main(max_epoch)
