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
from bigdl.dataset import mnist
from bigdl.dataset.transformer import *


from zoo import init_nncontext
from zoo.pipeline.api.net import TFOptimizer, TFDataset
from bigdl.optim.optimizer import *
import numpy as np
import sys

sys.path.append("/tmp/models/slim")  # add the slim library
from nets import lenet

slim = tf.contrib.slim

sc = init_nncontext()
spark = SQLContext(sc)

# get data, pre-process and create TFDataset
(images_data, labels_data) = mnist.read_data_sets("/tmp/mnist", "train")
image_rdd = sc.parallelize(images_data)
labels_rdd = sc.parallelize(labels_data)
rdd = image_rdd.zip(labels_rdd)\
    .map(lambda rec_tuple: [normalizer(rec_tuple[0], mnist.TRAIN_MEAN, mnist.TRAIN_STD),
                            np.array(rec_tuple[1])])

dataset = TFDataset.from_rdd(rdd,
                             names=["features", "labels"],
                             shapes=[(None, 28, 28, 1), (None, 1)],
                             types=[tf.float32, tf.int32]
                             )

# construct the model from TFDataset
images, labels = dataset.inputs

labels = tf.squeeze(labels)

with slim.arg_scope(lenet.lenet_arg_scope()):
    logits, end_points = lenet.lenet(images, num_classes=10, is_training=True)

loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))

# create a optimizer
optimizer = TFOptimizer(loss, Adam(1e-3))

# kick off training
optimizer.optimize(end_trigger=MaxEpoch(5), batch_size=256)

# evaluate
(images_data, labels_data) = mnist.read_data_sets("/tmp/mnist", "test")
images_data = normalizer(images_data, mnist.TRAIN_MEAN, mnist.TRAIN_STD)
predictions = tf.argmax(logits, axis=1)
predictions_data, loss_value = optimizer.sess.run([predictions, loss],
                                                  feed_dict={images: images_data,
                                                             labels: labels_data})
print(np.mean(np.equal(predictions_data, labels_data)))
print(loss_value)
