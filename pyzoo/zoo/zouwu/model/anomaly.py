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

from abc import ABCMeta, abstractmethod
import math


class BaseDetector(metaclass=ABCMeta):
    """
    BaseDetector class.
    """
    @abstractmethod
    def detect(self, actual, predicted=None):
        """
        Detect anomalies
        @param actual: the data to check anomaly
        @param predicted: the predicted values for comparison
        """
        pass


class UncertaintyDetector(BaseDetector):
    """
    Detector according to Uncertainty
    """
    def __init__(self):
        pass

    def detect(self, actual, predicted):
        """
        Detect anomalies
        uncertainty is a by-product from the forecast model.
        @param actual: the data to check anomaly
        @param predicted: the predicted values *AND* corresponding uncertainty
        """
        pass


class ThresholdDetector(BaseDetector):

    def __init__(self, threshold=math.inf):
        """
        Initializer.
        @param threshold: and absolute value as threshold.
            actual_value should be within predicted_value+/-threshold.
            default is inf so no anomaly reported
        """
        self.abs_threshold = threshold

    def fit(self, train_actual, train_predicted, mode='uniform', percentage=0.01):
        """
        Use fit to calculate the threshold based on training data
        @param train_actual: the training data to check anomaly
        @param train_predicted: the predicted value for comparison
        @param mode: "uniform" is the only mode supported right now.
        @param percentage: if type = 'uniform', threshold = 1-percentage percentile of
            all differences as in uniform distribution
        """
        pass

    def detect(self, actual, predicted):
        """
        Detect anomalies using the threshold either from preset in initializer or from fit.
        @param actual: the data to check anomaly
        @param predicted: the predicted values
        """
        pass


class AutoEncoderDetector(BaseDetector):

    def __init__(self):
        """
        Initializer.
        """
        pass

    def detect(self, actual, predicted=None):
        """
        Detect anomalies by fitting the data with an auto-encoder and detect anomalies from reconstruction error.
        so train and detect in one method.
        @param actual: the data to check anomaly
        @param predicted:
        """
        if predicted is not None:
            #TODO raise error or give warnings
            raise ValueError("predicted values should be None as auto-encoder only applies to actual values")
        # ... fit auto-encoder model on actual value
        # ... detect outliers in reconstruction error using gaussian distribution
        pass
