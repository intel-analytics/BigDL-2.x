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


class DummyForecaster:

    def __init__(self, **kwargs):
        """
        Build a Dummy Forecast Model.
        """

    def fit(self, data, epochs=1, validation_data=None, metric="mse", batch_size=32, **kwargs):
        """
        Fit(Train) the dummy forecaster.

        :param data: [Different requirement subject to each model] or a TSDataset
        :param validation_data: [Different requirement subject to each model] or a TSDataset. Default to None.
        :param epochs: Number of epochs you want to train. Default to 1.
        :param metric: The metric for training data. Default to "mse".
        :param batch_size: Number of batch size you want to train. Default to 32.
        """


    def predict(self, data):
        """
        Predict using a trained forecaster.

        :param data: [Different requirement subject to each model] or TSDataset
        :return : predict result
        """

    def evaluate(self, data, metrics=['mse'], multioutput="raw_values"):
        """
        Evaluate using a trained forecaster.

        :param data: [Different requirement subject to each model] or a TSDataset
        :param metrics: A list contains metrics for test/valid data.
        :param multioutput: Defines aggregating of multiple output values.
               String in ['raw_values', 'uniform_average']. The value defaults to
               'raw_values'.
        """