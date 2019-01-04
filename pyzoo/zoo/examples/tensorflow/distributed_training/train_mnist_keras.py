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
from zoo.pipeline.api.net import TFOptimizer, TFDataset
from bigdl.optim.optimizer import *
import sys
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

from bigdl.dataset import mnist
from bigdl.dataset.transformer import *


def main(max_epoch, data_num):
    sc = init_nncontext()

    # get data, pre-process and create TFDataset
    def get_data_rdd(dataset):
        (images_data, labels_data) = mnist.read_data_sets("/tmp/mnist", dataset)
        image_rdd = sc.parallelize(images_data[:data_num])
        labels_rdd = sc.parallelize(labels_data[:data_num])
        rdd = image_rdd.zip(labels_rdd) \
            .map(lambda rec_tuple: [normalizer(rec_tuple[0], mnist.TRAIN_MEAN, mnist.TRAIN_STD),
                                    np.array(rec_tuple[1])])
        return rdd

    training_rdd = get_data_rdd("train")
    testing_rdd = get_data_rdd("test")
    dataset = TFDataset.from_rdd(training_rdd,
                                 names=["features", "labels"],
                                 shapes=[[28, 28, 1], []],
                                 types=[tf.float32, tf.int32],
                                 batch_size=280,
                                 val_rdd=testing_rdd
                                 )

    data = Input(shape=[28, 28, 1])

    x = Flatten()(data)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    model = Model(inputs=data, outputs=predictions)

    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    optimizer = TFOptimizer.from_keras(model, dataset)

    optimizer.set_train_summary(TrainSummary("/tmp/mnist_log", "mnist"))
    optimizer.set_val_summary(ValidationSummary("/tmp/mnist_log", "mnist"))
    # kick off training
    optimizer.optimize(end_trigger=MaxEpoch(max_epoch))

    model.save_weights("/tmp/mnist_keras.h5")

if __name__ == '__main__':

    max_epoch = 5
    data_num = 60000

    if len(sys.argv) > 1:
        max_epoch = int(sys.argv[1])
        data_num = int(sys.argv[2])
    main(max_epoch, data_num)
