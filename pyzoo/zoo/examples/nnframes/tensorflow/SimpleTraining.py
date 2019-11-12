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
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import Adam
from bigdl.util.common import *
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from zoo.common.nncontext import *
from zoo.pipeline.api.net import TFNet
from zoo.pipeline.nnframes import *

import tensorflow as tf

# This is a simple example, showing how to train and inference a TensorFlow model with NNFrames
# on Spark DataFrame. It can also be used as a part of Spark ML Pipeline.

if __name__ == '__main__':

    sparkConf = init_spark_conf().setAppName("testNNClassifer").setMaster('local[1]')
    sc = init_nncontext(sparkConf)
    spark = SparkSession \
        .builder \
        .getOrCreate()

    with tf.Graph().as_default():
        input1 = tf.placeholder(dtype=tf.float32, shape=(None, 2))
        hidden = tf.layers.dense(input1, 4)
        output = tf.sigmoid(tf.layers.dense(hidden, 1))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            net = TFNet.from_session(sess, [input1], [output], generate_backward=True)

    df = spark.createDataFrame(
        [(Vectors.dense([2.0, 1.0]), 1.0),
         (Vectors.dense([1.0, 2.0]), 0.0),
         (Vectors.dense([2.0, 1.0]), 1.0),
         (Vectors.dense([1.0, 2.0]), 0.0)],
        ["features", "label"])

    print("before training:")
    NNModel(net).transform(df).show()

    classifier = NNClassifier(net, MSECriterion()) \
        .setBatchSize(4) \
        .setOptimMethod(Adam()) \
        .setLearningRate(0.1) \
        .setMaxEpoch(10)

    nnClassifierModel = classifier.fit(df)

    print("After training: ")
    res = nnClassifierModel.transform(df)
    res.show(10, False)

# expected output:
#
# before training:
# +---------+-----+------------+
# | features|label|  prediction|
# +---------+-----+------------+
# |[2.0,1.0]|  1.0|[0.46490368]|
# |[1.0,2.0]|  0.0|[0.51738966]|
# |[2.0,1.0]|  1.0|[0.46490368]|
# |[1.0,2.0]|  0.0|[0.51738966]|
# +---------+-----+------------+
#
# After training:
# +---------+-----+----------+
# |features |label|prediction|
# +---------+-----+----------+
# |[2.0,1.0]|1.0  |1.0       |
# |[1.0,2.0]|0.0  |0.0       |
# |[2.0,1.0]|1.0  |1.0       |
# |[1.0,2.0]|0.0  |0.0       |
# +---------+-----+----------+
