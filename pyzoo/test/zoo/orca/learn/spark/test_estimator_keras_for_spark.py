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

import zoo.orca.data.pandas
from zoo.orca.data.tf.data import Dataset
from zoo.orca.learn.tf.estimator import Estimator
from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.common.nncontext import *
from zoo.util import nest

class TestEstimatorForKeras(ZooTestCase):
    def setup_method(self, method):
        self.resource_path = os.path.join(os.path.split(__file__)[0], "../../../resources")
        sparkConf = init_spark_conf().setMaster("local[4]").setAppName("testSparkXShards")
        self.sc = init_nncontext(sparkConf)

    def teardown_method(self, method):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        self.sc.stop()

    def create_model(self):

        user = tf.keras.layers.Input(shape=[1])
        item = tf.keras.layers.Input(shape=[1])

        feat = tf.keras.layers.concatenate([user, item], axis=1)
        predictions = tf.keras.layers.Dense(2, activation='softmax')(feat)


        model = tf.keras.models.Model(inputs=[user, item], outputs=predictions)
        model.compile(optimizer='rmsprop',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def test_estimator_keras_xshards(self):
        import zoo.orca.data.pandas

        model = self.create_model()
        file_path = os.path.join(self.resource_path, "orca/learn/ncf.csv")
        data_shard = zoo.orca.data.pandas.read_csv(file_path, self.sc)

        def transform(df):
            result = {
                "x": (df['user'].to_numpy().reshape([-1,1]), df['item'].to_numpy().reshape([-1,1])),
                "y": df['label'].to_numpy()
            }
            return result

        data_shard = data_shard.transform_shard(transform)

        est = Estimator.from_keras(keras_model=model)
        est.fit(train_data=data_shard,
                batch_size=8,
                epochs=10,
                validation_data=data_shard)

        result = est.evaluate(data_shard)
        print(result)

        data_shard = zoo.orca.data.pandas.read_csv(file_path, self.sc)

        def transform(df):
            result = {
                "x": (df['user'].to_numpy().reshape([-1,1]), df['item'].to_numpy().reshape([-1,1])),
            }
            return result

        data_shard = data_shard.transform_shard(transform)
        predictions = est.predict(data_shard).collect()
        print(predictions)

    def test_estimator_keras_tfdataset(self):
        import zoo.orca.data.pandas

        model = self.create_model()
        file_path = os.path.join(self.resource_path, "orca/learn/ncf.csv")
        data_shard = zoo.orca.data.pandas.read_csv(file_path, self.sc)

        def transform(df):
            result = {
                "x": (df['user'].to_numpy().reshape([-1,1]), df['item'].to_numpy().reshape([-1,1])),
                "y": df['label'].to_numpy()
            }
            return result

        data_shard = data_shard.transform_shard(transform)
        dataset = Dataset.from_tensor_slices(data_shard)

        est = Estimator.from_keras(keras_model=model)
        est.fit(train_data=dataset,
                batch_size=8,
                epochs=10,
                validation_data=dataset)

        result = est.evaluate(dataset)
        print(result)

        data_shard = zoo.orca.data.pandas.read_csv(file_path, self.sc)

        def transform(df):
            result = {
                "x": (df['user'].to_numpy().reshape([-1,1]), df['item'].to_numpy().reshape([-1,1])),
            }
            return result

        data_shard = data_shard.transform_shard(transform)
        dataset = Dataset.from_tensor_slices(data_shard)
        predictions = est.predict(dataset).collect()
        print(predictions)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
