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
import tensorflow_datasets as tfds
from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca.learn.tf.estimator import Estimator


def main(max_epoch):
    sc = init_orca_context(cluster_mode="local", cores=4, memory="2g")

    # get DataSet
    mnist_train = tfds.load(name="mnist", split="train", as_supervised=True)
    mnist_test = tfds.load(name="mnist", split="test", as_supervised=True)

    # Normalizes images, unit8 -> float32
    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255., label

    mnist_train = mnist_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    mnist_test = mnist_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

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
    est.fit(data=mnist_train,
            batch_size=320,
            epochs=max_epoch,
            validation_data=mnist_test)

    result = est.evaluate(mnist_test)
    print(result)

    est.save_keras_model("/tmp/mnist_keras.h5")
    stop_orca_context()


if __name__ == '__main__':

    max_epoch = 5

    if len(sys.argv) > 1:
        max_epoch = int(sys.argv[1])
    main(max_epoch)
