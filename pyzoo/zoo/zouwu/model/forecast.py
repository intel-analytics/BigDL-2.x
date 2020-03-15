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
from zoo.automl.model.Seq2Seq import LSTMSeq2Seq as BaseSeq2Seq
from zoo.automl.model.VanillaLSTM import VanillaLSTM as BaseLSTM
from zoo.tfpark import KerasModel

import functools


class Forecaster:
    """
    Factory method for user to build or load models
    """
    @abstractmethod
    def __init__(self):
        """
        Forecaster now serves as a factory class
        """
        pass

    @staticmethod
    def load(model_file):
        """
        load a model from file, including format check
        """
        pass

    @staticmethod
    def build(model_type='lstm', horizon=1):
        """
        build a model from scratch
        """
        if model_type.lower() == 'lstm':
            return LSTMModel() \
                .set('horizon', horizon) \
                .build()
        elif model_type.lower() == 'seq2seq':
            return Seq2SeqModel() \
                .set('horizon', horizon) \
                .build()
        else:
            raise ValueError("model_type should be either \"lstm\" or \"seq2seq\"")


class BaseModel(metaclass=ABCMeta):
    """
    Base model for forecast models
    """
    def __init__(self, config=None):
        if config is not None:
            self.config = None
        else:
            self.config = {}

    def set(self, config, value):
        self.config[config] = value

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def train(self, train_x, train_y, validation_data, epochs, batch_size, distributed):
        pass

    @abstractmethod
    def predict(self, x, distributed, batch_per_thread):
        pass

    @abstractmethod
    def evaluate(self, x, y, metric, distributed, batch_per_thread):
        pass


class LSTMModel(BaseModel):
    """
    LSTM model
    """
    def __init__(self, horizon):
        super(LSTMModel, self).__init__()
        self.internal = None
        self.model = None

    def build(self):
        """
        Build model from scratch
        """
        # build model with TF/Keras
        self.internal = BaseLSTM(check_optional_config=False, future_seq_len=self.config.get('horizon', 1))
        internal_model = self.internal._build(mc=False, **self.config)
        self.model = KerasModel(internal_model)
        return self

    def check_model(self, func):
        """
        decorator function for model validation check
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # check model
            if self.model is None:
                raise Exception("model is not built properly, call build() before calling" + str(func.__name__))
            res = func(*args, **kwargs)
            return res

        return wrapper

    @check_model
    def fit(self, train_x, train_y, validation_data=None, epochs=5, batch_size=64, distributed=False):
        self.model.fit(train_x,
                       train_y,
                       validation_data=validation_data,
                       epochs=epochs,
                       batch_size=batch_size,
                       distributed=True)
        return self.model

    @check_model
    def evaluate(self, x, y, metric=['mse'], distributed=False, batch_per_thread=80):
        return self.model.evaluate(x, y,
                                   distributed=distributed,
                                   batch_per_thread=batch_per_thread)

    @check_model
    def predict(self, x, distributed=False, batch_per_thread=80):
        return self.model.predict(x,
                                  distributed=distributed,
                                  batch_per_thread=batch_per_thread)


class Seq2SeqModel(BaseModel):
    """
    Sequence To Sequence Model
    """
    def __init__(self, config=None):
        super(Seq2SeqModel,self).__init__(config)
        self.internal = None
        self.model = None

    def build(self):
        # build model with TF/Keras
        self.internal = BaseSeq2Seq(check_optional_config=False, future_seq_len=self.config.get('horizon', 1))
        internal_model = self.internal._build_train(mc=False, **self.config)
        # wrap keras model using TFPark, train and inference uses different part of model
        self.model = KerasModel(internal_model)
        return self

    def check_model(self, func):
        """
        decorator function for model validation check
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # check model
            if self.model is None:
                raise Exception("model is not built properly, call build() before calling" + str(func.__name__))
            res = func(*args, **kwargs)
            return res

        return wrapper

    @check_model
    def fit(self, train_x, train_y, validation_data=None, epochs=5, batch_size=64):
        self.model.fit(train_x,
                       train_y,
                       validation_data=validation_data,
                       epochs=epochs,
                       batch_size=batch_size,
                       distributed=True)
        return self.model

    @check_model
    def evaluate(self, x, y, metric, distributed, batch_per_thread):
        # predict first then evaluate
        pass

    @check_model
    def predict(self, x, distributed, batch_per_thread):
        # TODO need to decouple 1) and 2) steps in LSTMSeq2Seq._decode_sequence in order to add 1.1) below.
        #             1) build encoder and decoder model separately (in _decode_sequence)
        #             1.1) wrap encoder/decoder model using TFPark (need to add here)
        #             2) decode sequence using encoder and decoder model (in _decode_sequence)
        # get encoder, decoder from TFPark Keras model and build inference
        encoder, decoder = self.internal._build_inference(mc=False)
        keras_encoder = KerasModel(encoder)
        keras_decoder = KerasModel(decoder)
        ## decode sequence here
        # internal_model._decode_sequence(encoder, decoder, ...)
        pass


if __name__ == '__main__':
    # ... preprocessing data ...
    model = Forecaster.build(model_type='lstm',horizon=1)
    model.train()
    model.predict()
    model.evaluate()