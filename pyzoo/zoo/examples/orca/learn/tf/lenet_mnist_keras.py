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
import sys

import tensorflow as tf
import tensorflow_datasets as tfds
from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca.learn.tf.estimator import Estimator


def main(max_epoch):

    # get DataSet
    # as_supervised returns tuple (img, label) instead of dict {'image': img, 'label':label}
    mnist_train = tfds.load(name="mnist", split="train", as_supervised=True)
    mnist_test = tfds.load(name="mnist", split="test", as_supervised=True)

    # Normalizes images, unit8 -> float32
    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255., label

    mnist_train = mnist_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    mnist_test = mnist_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    model = tf.keras.Sequential(
        [tf.keras.layers.Conv2D(20, kernel_size=(5, 5), strides=(1, 1), activation='tanh',
                                input_shape=(28, 28, 1), padding='valid'),
         tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
         tf.keras.layers.Conv2D(50, kernel_size=(5, 5), strides=(1, 1), activation='tanh',
                                padding='valid'),
         tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(500, activation='tanh'),
         tf.keras.layers.Dense(10, activation='softmax'),
         ]
    )

    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    est = Estimator.from_keras(keras_model=model)
    est.fit(data=mnist_train,
            batch_size=320,
            epochs=max_epoch,
            validation_data=mnist_test)

    result = est.evaluate(mnist_test)
    print(result)

    est.save_keras_model("/tmp/mnist_keras.h5")

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
