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
import pytest

from bigdl.optim.optimizer import Adam, SGD, MaxEpoch
from zoo.pipeline.api.keras.metrics import Accuracy
from test.zoo.pipeline.utils.test_utils import ZooTestCase
import tensorflow as tf
import numpy as np

from zoo.tfpark import TFDataset, TFOptimizer, ZooOptimizer


class TestTFParkTFOptimizer(ZooTestCase):

    def setup_method(self, method):
        tf.keras.backend.clear_session()
        super(TestTFParkTFOptimizer, self).setup_method(method)

    def test_tf_optimizer_with_sparse_gradient(self):
        ids = np.random.randint(0, 10, size=[40])
        labels = np.random.randint(0, 5, size=[40])
        id_rdd = self.sc.parallelize(ids)
        label_rdd = self.sc.parallelize(labels)
        training_rdd = id_rdd.zip(label_rdd).map(lambda x: [x[0], x[1]])
        with tf.Graph().as_default():
            dataset = TFDataset.from_rdd(training_rdd,
                                         names=["ids", "labels"],
                                         shapes=[[], []],
                                         types=[tf.int32, tf.int32],
                                         batch_size=8)
            id_tensor, label_tensor = dataset.tensors
            embedding_table = tf.get_variable(
                name="word_embedding",
                shape=[10, 5])

            embedding = tf.nn.embedding_lookup(embedding_table, id_tensor)
            loss = tf.reduce_mean(tf.losses.
                                  sparse_softmax_cross_entropy(logits=embedding,
                                                               labels=label_tensor))
            train_op = ZooOptimizer(tf.train.AdamOptimizer(1e-3)).minimize(loss)
            optimizer = TFOptimizer.from_train_op(train_op, loss=loss)
            optimizer.optimize(end_trigger=MaxEpoch(1))
            optimizer.sess.close()

    def test_tf_optimizer_metrics(self):

        features = np.random.randn(20, 10)
        labels = np.random.randint(0, 10, size=[20])
        with tf.Graph().as_default():
            dataset = TFDataset.from_ndarrays((features, labels),
                                              batch_size=4,
                                              val_tensors=(features, labels))
            feature_tensor, label_tensor = dataset.tensors
            with tf.variable_scope("first_dense"):
                features = tf.layers.dense(feature_tensor, 8, name="first_dense")
            with tf.variable_scope("second_dense"):
                output = tf.layers.dense(features, 10, name="")

            loss = tf.reduce_mean(tf.losses.
                                  sparse_softmax_cross_entropy(logits=output,
                                                               labels=label_tensor))

            train_op_1 = ZooOptimizer(tf.train.AdamOptimizer(1e-3))\
                .minimize(loss, var_list=tf.trainable_variables("first_dense"))
            train_op_2 = ZooOptimizer(tf.train.GradientDescentOptimizer(0.0))\
                .minimize(loss, var_list=tf.trainable_variables("second_dense"))
            train_op = tf.group(train_op_1, train_op_2)
            optimizer = TFOptimizer.from_train_op(train_op, loss=loss, metrics={"loss": loss})
            # first run to get rid of bf16 conversion effect
            optimizer.optimize(end_trigger=MaxEpoch(1))
            initial_weights = optimizer.sess.run(tf.trainable_variables("first_dense") +
                                                 tf.trainable_variables("second_dense"))
            optimizer.optimize(end_trigger=MaxEpoch(2))
            updated_weights = optimizer.sess.run(tf.trainable_variables("first_dense") +
                                                 tf.trainable_variables("second_dense"))
            for i in [0, 1]:  # weights and bias combined with "first_dense" should be updated
                assert not np.allclose(initial_weights[i], updated_weights[i])
            for i in [2, 3]:  # weights and bias combined with "second_dense" should be unchanged
                assert np.allclose(initial_weights[i], updated_weights[i])
            optimizer.sess.close()


if __name__ == "__main__":
    pytest.main([__file__])
