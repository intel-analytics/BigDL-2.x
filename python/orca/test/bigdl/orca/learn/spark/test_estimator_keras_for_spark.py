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

import shutil
from unittest import TestCase

import tensorflow as tf

from bigdl.optim.optimizer import SeveralIteration
from zoo.orca.learn.tf.estimator import Estimator
from zoo.common.nncontext import *
from zoo.orca.learn.tf.utils import convert_predict_to_dataframe
import zoo.orca.data.pandas


class TestEstimatorForKeras(TestCase):
    def setup_method(self, method):
        self.resource_path = os.path.join(os.path.split(__file__)[0], "../../../resources")

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

    def create_model_with_clip(self):

        user = tf.keras.layers.Input(shape=[1])
        item = tf.keras.layers.Input(shape=[1])

        feat = tf.keras.layers.concatenate([user, item], axis=1)
        predictions = tf.keras.layers.Dense(2, activation='softmax')(feat)

        model = tf.keras.models.Model(inputs=[user, item], outputs=predictions)
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=0.001,
            rho=0.9,
            momentum=0.0,
            epsilon=1e-07,
            centered=False,
            name="RMSprop",
            clipnorm=1.2,
            clipvalue=0.2
        )
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def test_estimator_keras_xshards(self):
        import zoo.orca.data.pandas

        tf.reset_default_graph()

        model = self.create_model()
        file_path = os.path.join(self.resource_path, "orca/learn/ncf.csv")
        data_shard = zoo.orca.data.pandas.read_csv(file_path)

        def transform(df):
            result = {
                "x": (df['user'].to_numpy().reshape([-1, 1]),
                      df['item'].to_numpy().reshape([-1, 1])),
                "y": df['label'].to_numpy()
            }
            return result

        data_shard = data_shard.transform_shard(transform)

        est = Estimator.from_keras(keras_model=model)
        est.fit(data=data_shard,
                batch_size=8,
                epochs=10,
                validation_data=data_shard)

        eval_result = est.evaluate(data_shard)
        print(eval_result)

        data_shard = zoo.orca.data.pandas.read_csv(file_path)

        def transform(df):
            result = {
                "x": (df['user'].to_numpy().reshape([-1, 1]),
                      df['item'].to_numpy().reshape([-1, 1])),
            }
            return result

        data_shard = data_shard.transform_shard(transform)
        predictions = est.predict(data_shard).collect()
        assert predictions[0]['prediction'].shape[1] == 2

    def test_estimator_keras_xshards_options(self):
        import zoo.orca.data.pandas

        tf.reset_default_graph()

        model = self.create_model()
        file_path = os.path.join(self.resource_path, "orca/learn/ncf.csv")
        data_shard = zoo.orca.data.pandas.read_csv(file_path)

        def transform(df):
            result = {
                "x": (df['user'].to_numpy().reshape([-1, 1]),
                      df['item'].to_numpy().reshape([-1, 1])),
                "y": df['label'].to_numpy()
            }
            return result

        data_shard = data_shard.transform_shard(transform)

        est = Estimator.from_keras(keras_model=model)
        # train with no validation
        est.fit(data=data_shard,
                batch_size=8,
                epochs=10)
        # train with different optimizer
        est = Estimator.from_keras(keras_model=model)
        est.fit(data=data_shard,
                batch_size=8,
                epochs=10
                )
        # train with session config
        tf_session_config = tf.ConfigProto(inter_op_parallelism_threads=1,
                                           intra_op_parallelism_threads=1)
        est = Estimator.from_keras(keras_model=model)
        est.fit(data=data_shard,
                batch_size=8,
                epochs=10,
                session_config=tf_session_config
                )
        # train with model dir
        temp = tempfile.mkdtemp()
        model_dir = os.path.join(temp, "model")
        est = Estimator.from_keras(keras_model=model, model_dir=model_dir)
        est.fit(data=data_shard,
                batch_size=8,
                epochs=10,
                validation_data=data_shard)
        assert len(os.listdir(model_dir)) > 0
        shutil.rmtree(temp)

    def test_estimator_keras_xshards_clip(self):
        import zoo.orca.data.pandas

        tf.reset_default_graph()

        model = self.create_model_with_clip()
        file_path = os.path.join(self.resource_path, "orca/learn/ncf.csv")
        data_shard = zoo.orca.data.pandas.read_csv(file_path)

        def transform(df):
            result = {
                "x": (df['user'].to_numpy().reshape([-1, 1]),
                      df['item'].to_numpy().reshape([-1, 1])),
                "y": df['label'].to_numpy()
            }
            return result

        data_shard = data_shard.transform_shard(transform)

        est = Estimator.from_keras(keras_model=model)
        est.fit(data=data_shard,
                batch_size=8,
                epochs=10,
                validation_data=data_shard)

    def test_estimator_keras_xshards_checkpoint(self):
        import zoo.orca.data.pandas

        tf.reset_default_graph()

        import tensorflow.keras.backend as K
        K.clear_session()
        tf.reset_default_graph()
        model = self.create_model()
        file_path = os.path.join(self.resource_path, "orca/learn/ncf.csv")
        data_shard = zoo.orca.data.pandas.read_csv(file_path)

        def transform(df):
            result = {
                "x": (df['user'].to_numpy().reshape([-1, 1]),
                      df['item'].to_numpy().reshape([-1, 1])),
                "y": df['label'].to_numpy()
            }
            return result

        data_shard = data_shard.transform_shard(transform)

        temp = tempfile.mkdtemp()
        model_dir = os.path.join(temp, "test_model")

        est = Estimator.from_keras(keras_model=model, model_dir=model_dir)
        est.fit(data=data_shard,
                batch_size=8,
                epochs=6,
                validation_data=data_shard,
                checkpoint_trigger=SeveralIteration(4))

        eval_result = est.evaluate(data_shard)
        print(eval_result)

        K.get_session().close()
        tf.reset_default_graph()
        model = self.create_model()

        est = Estimator.from_keras(keras_model=model, model_dir=model_dir)
        est.load_latest_checkpoint(model_dir)
        est.fit(data=data_shard,
                batch_size=8,
                epochs=10,
                validation_data=data_shard,
                checkpoint_trigger=SeveralIteration(4))

        eval_result = est.evaluate(data_shard)
        print(eval_result)
        shutil.rmtree(temp)

    def test_estimator_keras_dataframe(self):

        tf.reset_default_graph()

        model = self.create_model()
        sc = init_nncontext()
        sqlcontext = SQLContext(sc)
        file_path = os.path.join(self.resource_path, "orca/learn/ncf.csv")
        df = sqlcontext.read.csv(file_path, header=True, inferSchema=True)
        from pyspark.sql.functions import array
        df = df.withColumn('user', array('user')) \
            .withColumn('item', array('item'))

        est = Estimator.from_keras(keras_model=model)
        est.fit(data=df,
                batch_size=8,
                epochs=4,
                feature_cols=['user', 'item'],
                labels_cols=['label'],
                validation_data=df)

        eval_result = est.evaluate(df, feature_cols=['user', 'item'], labels_cols=['label'])
        assert 'acc Top1Accuracy' in eval_result

        prediction_df = est.predict(df, batch_size=4, feature_cols=['user', 'item'])
        assert 'prediction' in prediction_df.columns
        predictions = prediction_df.collect()
        assert len(predictions) == 10

    def test_estimator_keras_dataframe_no_fit(self):

        tf.reset_default_graph()

        model = self.create_model()
        sc = init_nncontext()
        sqlcontext = SQLContext(sc)
        file_path = os.path.join(self.resource_path, "orca/learn/ncf.csv")
        df = sqlcontext.read.csv(file_path, header=True, inferSchema=True)
        from pyspark.sql.functions import array
        df = df.withColumn('user', array('user')) \
            .withColumn('item', array('item'))

        est = Estimator.from_keras(keras_model=model)

        eval_result = est.evaluate(df, feature_cols=['user', 'item'], labels_cols=['label'])
        assert 'acc Top1Accuracy' in eval_result

        prediction_df = est.predict(df, batch_size=4, feature_cols=['user', 'item'])
        assert 'prediction' in prediction_df.columns
        predictions = prediction_df.collect()
        assert len(predictions) == 10

    def test_convert_predict_list_of_array(self):

        tf.reset_default_graph()

        sc = init_nncontext()
        sqlcontext = SQLContext(sc)
        rdd = sc.parallelize([(1, 2, 3), (4, 5, 6), (7, 8, 9)])
        df = rdd.toDF(["feature", "label", "c"])
        predict_rdd = df.rdd.map(lambda row: [np.array([1, 2]), np.array(0)])
        resultDF = convert_predict_to_dataframe(df, predict_rdd)
        resultDF.printSchema()
        print(resultDF.collect()[0])
        predict_rdd = df.rdd.map(lambda row: np.array(1))
        resultDF = convert_predict_to_dataframe(df, predict_rdd)
        resultDF.printSchema()
        print(resultDF.collect()[0])


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
