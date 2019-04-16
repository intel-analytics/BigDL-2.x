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

from bigdl.dataset.dataset import DataSet
from zoo import init_nncontext
from zoo.feature.common import ChainedPreprocessing
from zoo.feature.image import ImageSet, ImageSetToSample, ImageResize, ImageMatToTensor, ImageRandomCrop, \
    ImageChannelNormalize
from zoo.tfpark import TFDataset
from zoo.tfpark.estimator import TFEstimator, TFEstimatorSpec
import os

# os.environ['PYSPARK_SUBMIT_ARGS'] = "--driver-java-options \" -Xdebug -Xrunjdwp:transport=dt_socket,server=y,suspend=y,address=5050\" /home/yang/sources/zoo/pyzoo/zoo/examples/tensorflow/tfpark/estimator_inception.py"

def main():
    sc = init_nncontext()

    def input_fn(mode):

        if mode == tf.estimator.ModeKeys.TRAIN:
            image_set = ImageSet.read("/home/yang/sources/datasets/cat_dog/demo_small", sc=sc, with_label=True)
            transformer = ChainedPreprocessing([ImageResize(256, 256),
                                                ImageRandomCrop(224, 224, True),
                                                ImageChannelNormalize(123.0, 117.0, 104.0),
                                                ImageMatToTensor(format="NHWC"),
                                                ImageSetToSample(input_keys=["imageTensor"], target_keys=["label"])])
            image_set = image_set.transform(transformer)
            dataset = DataSet.image_frame(image_set.to_image_frame())
            dataset = TFDataset.from_bigdl_dataset(dataset, features=(tf.float32, [224, 224, 3]), labels=(tf.int32, [1]), batch_size=16)
        else:
            raise NotImplementedError

        return dataset

    def model_fn(features, labels, mode):
        from nets import inception
        slim = tf.contrib.slim
        labels = tf.squeeze(labels, axis=1) - 1  # ImageSet's label is a 1-based vector
        with slim.arg_scope(inception.inception_v1_arg_scope()):
            logits, end_points = inception.inception_v1(features, num_classes=2, is_training=True)

        if mode == tf.estimator.ModeKeys.TRAIN:
            loss = tf.reduce_mean(
                tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))
            return TFEstimatorSpec(mode, predictions=logits, loss=loss)
        else:
            raise NotImplementedError

    estimator = TFEstimator(model_fn, tf.train.AdamOptimizer())

    estimator.train(input_fn, steps=100)

if __name__ == '__main__':
    main()
