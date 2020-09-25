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

from bigdl.nn.criterion import *
from bigdl.nn.layer import *
from pyspark.sql.types import *

from zoo.common.nncontext import *
from zoo.feature.common import *
from zoo.orca.learn.bigdl import Estimator
from bigdl.optim.optimizer import *
from zoo.pipeline.api.keras import layers as ZLayer
from zoo.pipeline.api.keras.models import Model as ZModel
from zoo.orca.data import SparkXShards
from zoo.orca.learn.metrics import Accuracy
from zoo.orca.learn.trigger import EveryEpoch
from zoo.orca.data.pandas import read_csv


class TestEstimatorForKeras(TestCase):
    def get_estimator_df(self):
        self.sc = init_nncontext()
        data = self.sc.parallelize([
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0)),
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0))])

        schema = StructType([
            StructField("features", ArrayType(DoubleType(), False), False),
            StructField("label", ArrayType(DoubleType(), False), False)])
        self.sqlContext = SQLContext(self.sc)
        df = self.sqlContext.createDataFrame(data, schema)
        return df

    def test_nnEstimator(self):
        from zoo.pipeline.nnframes import NNModel
        linear_model = Sequential().add(Linear(2, 2))
        mse_criterion = MSECriterion()
        df = self.get_estimator_df()
        est = Estimator.from_bigdl(model=linear_model, loss=mse_criterion,
                                   feature_preprocessing=SeqToTensor([2]),
                                   label_preprocessing=SeqToTensor([2]))
        est.fit(df, 1, batch_size=4, optimizer=Adam())
        nn_model = NNModel(est.get_model(), feature_preprocessing=SeqToTensor([2]))
        res1 = nn_model.transform(df)
        res2 = est.predict(df)
        res1_c = res1.collect()
        res2_c = res2.collect()
        assert type(res1).__name__ == 'DataFrame'
        assert type(res2).__name__ == 'DataFrame'
        assert len(res1_c) == len(res2_c)
        for idx in range(len(res1_c)):
            assert res1_c[idx]["prediction"] == res2_c[idx]["prediction"]
        with tempfile.TemporaryDirectory() as tempdirname:
            temp_path = os.path.join(tempdirname, "model")
            est.save(temp_path)
            est2 = Estimator.from_bigdl(model=linear_model, loss=mse_criterion)
            est2.load(temp_path, optimizer=Adam(), loss=mse_criterion,
                      feature_preprocessing=SeqToTensor([2]), label_preprocessing=SeqToTensor([2]))
            with self.assertRaises(Exception) as context:
                est2.predict(df)
            self.assertTrue('You should fit or set_input_type before calling predict'
                            in str(context.exception))
            est2.set_input_type(input_type="Spark_DataFrame")
            est2.set_constant_gradient_clipping(0.1, 1.2)
            est2.clear_gradient_clipping()
            res3 = est2.predict(df)
            res3_c = res3.collect()
            assert type(res3).__name__ == 'DataFrame'
            assert len(res1_c) == len(res3_c)
            for idx in range(len(res1_c)):
                assert res1_c[idx]["prediction"] == res3_c[idx]["prediction"]
            est2.fit(df, 1, batch_size=4, optimizer=Adam())
        resource_path = os.path.join(os.path.split(__file__)[0], "../../../resources")

        file_path = os.path.join(resource_path, "orca/learn/ncf2.csv")
        data_shard = read_csv(file_path)
        with self.assertRaises(Exception) as context:
            est.fit(data_shard, 1, batch_size=4, optimizer=Adam())
        self.assertTrue('This estimator only support spark DataFrame as training data'
                        in str(context.exception))
        with self.assertRaises(Exception) as context:
            est.predict(data_shard)
        self.assertTrue('Data should be spark DataFrame but get' in str(context.exception))

    def test_nnEstimator_multiInput(self):
        zx1 = ZLayer.Input(shape=(1,))
        zx2 = ZLayer.Input(shape=(1,))
        zz = ZLayer.merge([zx1, zx2], mode="concat")
        zy = ZLayer.Dense(2)(zz)
        zmodel = ZModel([zx1, zx2], zy)

        criterion = MSECriterion()
        df = self.get_estimator_df()
        estimator = Estimator.from_bigdl(model=zmodel, loss=criterion,
                                         feature_preprocessing=[[1], [1]])
        estimator.fit(df, epochs=5, batch_size=4)
        pred = estimator.predict(df)
        pred_data = pred.collect()
        assert type(pred).__name__ == 'DataFrame'

    def test_nnEstimator_multiInput_cols(self):
        from pyspark.ml.linalg import Vectors
        from pyspark.sql import SparkSession

        spark = SparkSession \
            .builder \
            .getOrCreate()

        df = spark.createDataFrame(
            [(1, 35, 109.0, Vectors.dense([2.0, 5.0, 0.5, 0.5]), 1.0),
             (2, 58, 2998.0, Vectors.dense([4.0, 10.0, 0.5, 0.5]), 2.0),
             (3, 18, 123.0, Vectors.dense([3.0, 15.0, 0.5, 0.5]), 1.0),
             (4, 18, 123.0, Vectors.dense([3.0, 15.0, 0.5, 0.5]), 1.0)],
            ["user", "age", "income", "history", "label"])

        x1 = ZLayer.Input(shape=(1,))
        x2 = ZLayer.Input(shape=(2,))
        x3 = ZLayer.Input(shape=(2, 2,))

        user_embedding = ZLayer.Embedding(5, 10)(x1)
        flatten = ZLayer.Flatten()(user_embedding)
        dense1 = ZLayer.Dense(2)(x2)
        gru = ZLayer.LSTM(4, input_shape=(2, 2))(x3)

        merged = ZLayer.merge([flatten, dense1, gru], mode="concat")
        zy = ZLayer.Dense(2)(merged)

        zmodel = ZModel([x1, x2, x3], zy)
        criterion = ClassNLLCriterion()
        est = Estimator.from_bigdl(model=zmodel, loss=criterion,
                                   feature_preprocessing=[[1], [2], [2, 2]])
        est.fit(df, epochs=1, batch_size=4, optimizer=Adam(learningrate=0.1),
                feature_cols=["user", "age", "income", "history"])

        res = est.predict(df, feature_cols=["user", "age", "income", "history"])
        res_c = res.collect()
        assert type(res).__name__ == 'DataFrame'

    def test_xshards_spark_estimator(self):
        resource_path = os.path.join(os.path.split(__file__)[0], "../../../resources")

        def transform(df):
            result = {
                "x": [df['user'].to_numpy(), df['item'].to_numpy()],
                "y": df['label'].to_numpy()
            }
            return result

        file_path = os.path.join(resource_path, "orca/learn/ncf2.csv")
        data_shard = read_csv(file_path)
        data_shard = data_shard.transform_shard(transform)
        model = Sequential()
        model.add(Linear(2, 2))
        model.add(LogSoftMax())
        optim_method = SGD(learningrate=0.01)

        estimator = Estimator.from_bigdl(model=model, optimizer=optim_method,
                                         loss=ClassNLLCriterion())
        with self.assertRaises(Exception) as context:
            estimator.set_constant_gradient_clipping(0.1, 1.2)
        self.assertTrue('Please call set_input_type before calling set_constant_gradient_clipping.'
                        in str(context.exception))
        estimator.set_input_type(input_type="sparkXshards")
        estimator.set_constant_gradient_clipping(0.1, 1.2)
        r1 = estimator.predict(data=data_shard)
        r_c = r1.collect()
        estimator.fit(data=data_shard, epochs=5, batch_size=8, val_data=data_shard,
                      val_methods=[Accuracy()], checkpoint_trigger=EveryEpoch())
        estimator.evaluate(data=data_shard, validation_methods=[Accuracy()], batch_size=8)
        result = estimator.predict(data=data_shard)
        assert type(result).__name__ == 'SparkXShards'
        result_c = result.collect()
        df = self.get_estimator_df()
        with self.assertRaises(Exception) as context:
            estimator.fit(df, epochs=1)
        self.assertTrue('Data and validation data should be SparkXShards, but get'
                        in str(context.exception))


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
