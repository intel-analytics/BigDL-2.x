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

import warnings

from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import *
from zoo.models.common import *
from zoo.common.utils import callZooFunc
from bigdl.util.common import Sample

if sys.version >= '3':
    long = int
    unicode = str


class AnomalyDetector(KerasZooModel):
    """
    The anomaly detector model for sequence data based on LSTM.

    # Arguments
    feature_shape: The input shape of features, including unroll_length and feature_size.
    hidden_layers: Units of hidden layers of LSTM.
    dropouts:     Fraction of the input units to drop out. Float between 0 and 1.
    """

    def __init__(self, feature_shape, hidden_layers=[8, 32, 15],
                 dropouts=[0.2, 0.2, 0.2], **kwargs):
        assert len(hidden_layers) == len(dropouts), \
            "sizes of dropouts and hidden_layers should be equal"
        if 'bigdl_type' in kwargs:
            kwargs.pop('bigdl_type')
            self.bigdl_type = kwargs.get("bigdl_type")
        else:
            self.bigdl_type = "float"
        if kwargs:
            raise TypeError('Wrong arguments for AnomalyDetector: ' + str(kwargs))
        self.feature_shape = feature_shape
        self.hidden_layers = hidden_layers
        self.dropouts = dropouts
        self.model = self.build_model()
        super(AnomalyDetector, self).__init__(None, self.bigdl_type,
                                              feature_shape,
                                              hidden_layers,
                                              dropouts,
                                              self.model)

    def build_model(self):
        model = Sequential()
        model.add(InputLayer(input_shape=self.feature_shape)) \
            .add(LSTM(input_shape=self.feature_shape, output_dim=self.hidden_layers[0],
                      return_sequences=True))

        for ilayer in range(1, len(self.hidden_layers) - 1):
            model.add(LSTM(output_dim=self.hidden_layers[ilayer], return_sequences=True)) \
                .add(Dropout(self.dropouts[ilayer]))

        model.add(LSTM(self.hidden_layers[-1], return_sequences=False)) \
            .add(Dropout(self.dropouts[-1]))

        model.add(Dense(output_dim=1))
        return model

    @staticmethod
    def load_model(path, weight_path=None, bigdl_type="float"):
        """
        Load an existing AnomalyDetector model (with weights).

        # Arguments
        path: The path for the pre-defined model.
              Local file system, HDFS and Amazon S3 are supported.
              HDFS path should be like 'hdfs://[host]:[port]/xxx'.
              Amazon S3 path should be like 's3a://bucket/xxx'.
        weight_path: The path for pre-trained weights if any. Default is None.
        """
        jmodel = callZooFunc(bigdl_type, "loadAnomalyDetector", path, weight_path)
        model = KerasZooModel._do_load(jmodel, bigdl_type)
        model.__class__ = AnomalyDetector
        return model

    def predict(self, x, batch_per_thread=8):
        """
        Precict on RDD[Sample].
        """
        results = callZooFunc(self.bigdl_type, "modelPredictRDD",
                              self.value,
                              x,
                              batch_per_thread)
        return results.map(lambda data: data.to_ndarray())

    @classmethod
    def unroll(cls, data_rdd, unroll_length, predict_step=1):
        """
        Unroll a rdd of arrays to prepare features and labels.

        # Arguments
        data_rdd: RDD[Array]. data to be unrolled, it holds original time series features
        unroll_length: Int. the length of precious values to predict future value.
        predict_step: Int. How many time steps to predict future value, default is 1.
        return: an rdd of FeatureLableIndex
        a simple example
                     data: (1,2,3,4,5,6); unrollLength: 2, predictStep: 1
                     features, label, index
                     (1,2), 3, 0
                     (2,3), 4, 1
                     (3,4), 5, 2
                     (4,5), 6, 3
        """
        result = callZooFunc("float", "unroll", data_rdd, unroll_length, predict_step)
        return cls._to_indexed_rdd(result)

    @classmethod
    def detect_anomalies(cls, ytruth, ypredict, anomaly_size):
        """
        # Arguments
        :param ytruth: RDD of float or double values. Truth to be compared.
        :param ypredict: RDD of float or double values. Predictions.
        :param anomaly_size: Int. The size to be considered as anomalies.
        :return: RDD of [ytruth, ypredict, anomaly], anomaly is None or ytruth
        """
        return callZooFunc("float", "detectAnomalies", ytruth, ypredict, anomaly_size)

    @staticmethod
    def standardScale(df):
        return callZooFunc("float", "standardScaleDF", df)

    @staticmethod
    def train_test_split(unrolled, test_size):
        cutPoint = unrolled.count() - test_size
        train = unrolled.filter(lambda x: x.index < cutPoint) \
            .map(lambda x: Sample.from_ndarray(np.array(x.feature), np.array(x.label)))
        test = unrolled.filter(lambda x: x.index >= cutPoint) \
            .map(lambda x: Sample.from_ndarray(np.array(x.feature), np.array(x.label)))
        return [train, test]

    @staticmethod
    def _to_indexed_rdd(unrolled_rdd):
        def row_to_feature(feature_str):
            feature = [x.split("|") for x in feature_str.split(",")]
            matrix = []
            for i in range(0, len(feature)):
                line = []
                for j in range(0, len(feature[0])):
                    line.append(float(feature[i][j]))
            matrix.append(line)
            return matrix

        return unrolled_rdd \
            .map(lambda y: FeatureLableIndex(row_to_feature(y[0]), float(y[1]), long(y[2])))


class FeatureLableIndex(object):
    """
    Each record should contain the following fields:
    feature: List[List[float]].
    label: float.
    index: long.
    """

    def __init__(self, feature, label, index, bigdl_type="float"):
        self.feature = feature
        self.label = label
        self.index = index
        self.bigdl_type = bigdl_type

    def __reduce__(self):
        return FeatureLableIndex, (self.feature, self.label, self.index)

    def __str__(self):
        return "FeatureLableIndex [feature: %s, label: %s, index: %s]" % (
            self.feature, self.label, self.index)
