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
import os
import tensorflow as tf


from zoo.orca.learn.tensorflow.estimator import Estimator

resource_path = os.path.join(os.path.split(__file__)[0], "../../../resources")


def test_estimator_pre_built_graph(estimator_for_spark_fixture):
    from bigdl.optim.optimizer import SGD
    import zoo.orca.data.pandas

    user = tf.placeholder(dtype=tf.int32, shape=(None,))
    item = tf.placeholder(dtype=tf.int32, shape=(None,))
    label = tf.placeholder(dtype=tf.int32, shape=(None,))

    feat = tf.stack([user, item], axis=1)
    logits = tf.layers.dense(tf.to_float(feat), 2)

    loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=label))

    sc = estimator_for_spark_fixture

    file_path = os.path.join(resource_path, "orca/learn/ncf.csv")
    data_shard = zoo.orca.data.pandas.read_csv(file_path, sc)

    def transform(df):
        result = {
            "x": [df['user'].to_numpy(), df['item'].to_numpy()],
            "y": [df['label']]
        }
        return result

    data_shard.transform_shard(transform)

    est = Estimator.from_pre_built_graph(
        inputs=[user, item],
        labels=[label],
        outputs=[logits],
        loss=loss,
        optimizer=SGD(),
        metrics={"loss": loss})
    est.fit(data_shard=data_shard,
            batch_size=8,
            steps=10,
            validation_data_shard=data_shard)

    data_shard = zoo.orca.data.pandas.read_csv(file_path, sc)

    def transform(df):
        result = {
            "x": [df['user'].to_numpy(), df['item'].to_numpy()],
        }
        return result

    data_shard.transform_shard(transform)
    predictions = est.predict(data_shard).collect()
    print(predictions)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
