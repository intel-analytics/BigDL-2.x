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
import zoo.orca.data.pandas
import os
import tensorflow as tf

from zoo.orca.learn.tensorflow.estimator import Estimator
from zoo.tfpark import ZooOptimizer

resource_path = os.path.join(os.path.split(__file__)[0], "../../../resources")


def test_estimator_model_fn(estimator_for_spark_fixture):
    def model_fn(features, labels, mode):
        user = features['user']
        item = features['item']

        feat = tf.stack([user, item], axis=1)
        logits = tf.layers.dense(tf.to_float(feat), 2)

        if mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.TRAIN:
            labels = labels['label']
            loss = tf.reduce_mean(
                tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))
            train_op = ZooOptimizer(tf.train.AdamOptimizer()).minimize(loss)
            return tf.estimator.EstimatorSpec(mode, train_op=train_op,
                                              predictions=logits, loss=loss)
        else:
            return tf.estimator.EstimatorSpec(mode, predictions=logits)

    sc = estimator_for_spark_fixture

    file_path = os.path.join(resource_path, "orca/learn/ncf.csv")
    data_shard = zoo.orca.data.pandas.read_csv(file_path, sc)
    est = Estimator.from_model_fn(model_fn, backend="spark")
    est.fit(data_shard,
            features=["user", "item"],
            labels=["label"],
            steps=100,
            batch_size=8)
    result = est.evaluate(data_shard,
                          features=["user", "item"],
                          labels=["label"],
                          eval_methods=["acc"])
    print(result)
    predictions = est.predict(data_shard, features=["user", "item"]).collect()
    print(predictions)
