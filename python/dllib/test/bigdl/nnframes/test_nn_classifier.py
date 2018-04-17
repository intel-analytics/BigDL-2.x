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
from bigdl.nn.criterion import *
from bigdl.nn.layer import *
from bigdl.util.common import *
from numpy.testing import assert_allclose
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler
from pyspark.sql.types import *

from zoo.common.nncontext import *
from zoo.pipeline.nnframes.nn_classifier import *


class TestNNClassifer():
    def setup_method(self, method):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        sparkConf = create_spark_conf().setMaster("local[1]").setAppName("testNNClassifer")
        self.sc = get_nncontext(sparkConf)
        self.sqlContext = SQLContext(self.sc)

    def teardown_method(self, method):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        self.sc.stop()

    def test_all_set_get_methods(self):
        """ run tests for all the set and get methods involved in NNEstimator, NNModel,
            DLClassifier, DLClassifierModel
        """

        linear_model = Sequential().add(Linear(2, 2))
        mse_criterion = MSECriterion()

        estimator = NNEstimator(model=linear_model, criterion=mse_criterion,
                                feature_size=[2], label_size=[2])
        assert estimator.setBatchSize(30).getBatchSize() == 30
        assert estimator.setMaxEpoch(40).getMaxEpoch() == 40
        assert estimator.setLearningRate(1e-4).getLearningRate() == 1e-4
        assert estimator.setFeaturesCol("abcd").getFeaturesCol() == "abcd"
        assert estimator.setLabelCol("xyz").getLabelCol() == "xyz"

        nn_model = NNModel(model=linear_model, featureSize=[1])
        assert nn_model.setFeatureSize([2]).getFeatureSize() == [2]
        assert nn_model.setBatchSize(20).getBatchSize() == 20

        linear_model = Sequential().add(Linear(2, 2))
        classNLL_criterion = ClassNLLCriterion()

        classifier = NNClassifier(model=linear_model, criterion=classNLL_criterion,
                                  feature_size=[2])
        assert classifier.setBatchSize(20).getBatchSize() == 20
        assert classifier.setMaxEpoch(50).getMaxEpoch() == 50
        assert classifier.setLearningRate(1e-5).getLearningRate() == 1e-5

        nn_classifier_model = NNClassifierModel(model=linear_model, featureSize=[1])
        assert nn_classifier_model.setFeatureSize([2]).getFeatureSize() == [2]
        assert nn_classifier_model.setBatchSize((20)).getBatchSize() == 20

    def test_nnEstimator_fit_nnmodel_transform(self):
        model = Sequential().add(Linear(2, 2))
        criterion = MSECriterion()
        estimator = NNEstimator(model, criterion, [2], [2]).setBatchSize(4) \
            .setLearningRate(0.2).setMaxEpoch(40)

        data = self.sc.parallelize([
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0)),
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0))])

        schema = StructType([
            StructField("features", ArrayType(DoubleType(), False), False),
            StructField("label", ArrayType(DoubleType(), False), False)])
        df = self.sqlContext.createDataFrame(data, schema)
        nnModel = estimator.fit(df)
        assert nnModel.getBatchSize() == 4

        res = nnModel.transform(df)
        assert type(res).__name__ == 'DataFrame'
        res.registerTempTable("nnModelDF")  # Compatible with spark 1.6
        results = self.sqlContext.table("nnModelDF")

        count = results.rdd.count()
        data = results.rdd.collect()

        for i in range(count):
            row_label = data[i][1]
            row_prediction = data[i][2]
            assert_allclose(row_label[0], row_prediction[0], atol=0, rtol=1e-1)
            assert_allclose(row_label[1], row_prediction[1], atol=0, rtol=1e-1)

    def test_nnEstimator_fit_with_non_default_featureCol(self):
        model = Sequential().add(Linear(2, 2))
        criterion = MSECriterion()
        estimator = NNEstimator(model, criterion, [2], [2]).setBatchSize(4)\
            .setLearningRate(0.01).setMaxEpoch(1) \
            .setFeaturesCol("abcd").setLabelCol("xyz").setPredictionCol("tt")

        data = self.sc.parallelize([
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0)),
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0))])

        schema = StructType([
            StructField("abcd", ArrayType(DoubleType(), False), False),
            StructField("xyz", ArrayType(DoubleType(), False), False)])
        df = self.sqlContext.createDataFrame(data, schema)
        nnModel = estimator.fit(df)

        res = nnModel.transform(df)
        assert type(res).__name__ == 'DataFrame'
        assert res.select("abcd", "xyz", "tt").count() == 4

    def test_NNModel_transform_with_nonDefault_featureCol(self):
        model = Sequential().add(Linear(2, 2))
        nnModel = NNModel(model, [2]).setFeaturesCol("abcd").setPredictionCol("dcba")

        data = self.sc.parallelize([
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0)),
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0))])

        schema = StructType([
            StructField("abcd", ArrayType(DoubleType(), False), False),
            StructField("xyz", ArrayType(DoubleType(), False), False)])
        df = self.sqlContext.createDataFrame(data, schema)
        res = nnModel.transform(df)
        assert type(res).__name__ == 'DataFrame'
        assert res.select("abcd", "dcba").count() == 4

    def test_nnclassifier_fit_nnclassifiermodel_transform(self):
        model = Sequential().add(Linear(2, 2))
        criterion = ClassNLLCriterion()
        classifier = NNClassifier(model, criterion, [2]).setBatchSize(4) \
            .setLearningRate(0.2).setMaxEpoch(40)
        data = self.sc.parallelize([
            ((2.0, 1.0), 1.0),
            ((1.0, 2.0), 2.0),
            ((2.0, 1.0), 1.0),
            ((1.0, 2.0), 2.0)])

        schema = StructType([
            StructField("features", ArrayType(DoubleType(), False), False),
            StructField("label", DoubleType(), False)])
        df = self.sqlContext.createDataFrame(data, schema)
        nnClassifierModel = classifier.fit(df)

        res = nnClassifierModel.transform(df)
        assert type(res).__name__ == 'DataFrame'
        res.registerTempTable("nnClassifierModelDF")
        results = self.sqlContext.table("nnClassifierModelDF")

        count = results.rdd.count()
        data = results.rdd.collect()

        for i in range(count):
            row_label = data[i][1]
            row_prediction = data[i][2]
            assert row_label == row_prediction

    def test_nnclassifier_in_pipeline(self):
        if self.sc.version.startswith("1"):
            from pyspark.mllib.linalg import Vectors
        else:
            from pyspark.ml.linalg import Vectors

        df = self.sqlContext.createDataFrame([(Vectors.dense([2.0, 1.0]), 1.0),
                                              (Vectors.dense([1.0, 2.0]), 2.0),
                                              (Vectors.dense([2.0, 1.0]), 1.0),
                                              (Vectors.dense([1.0, 2.0]), 2.0),
                                              ], ["features", "label"])

        scaler = MinMaxScaler().setInputCol("features").setOutputCol("scaled")
        model = Sequential().add(Linear(2, 2))
        criterion = ClassNLLCriterion()
        classifier = NNClassifier(model, criterion, [2]).setBatchSize(4) \
            .setLearningRate(0.01).setMaxEpoch(10).setFeaturesCol("scaled")

        pipeline = Pipeline(stages=[scaler, classifier])

        pipelineModel = pipeline.fit(df)

        res = pipelineModel.transform(df)
        assert type(res).__name__ == 'DataFrame'

if __name__ == "__main__":
    pytest.main()
