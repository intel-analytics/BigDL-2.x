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
#
# MIT License
#
# Copyright (c) 2018 Roland Zimmermann
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import numpy as np
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import TruncatedNormal, Constant
import tensorflow.keras.backend as K

import tensorflow as tf
from zoo.automl.common.metrics import Evaluator
from zoo.automl.model.abstract import BaseModel
from zoo.automl.common.util import save_config


class AttentionRNNWrapper(Wrapper):
    """
        This class is modified based on
        https://github.com/zimmerrol/keras-utility-layer-collection/blob/master/kulc/attention.py.
        The idea of the implementation is based on the paper:
            "Effective Approaches to Attention-based Neural Machine Translation" by Luong et al.
        This layer is an attention layer, which can be wrapped around arbitrary RNN layers.
        This way, after each time step an attention vector is calculated
        based on the current output of the LSTM and the entire input time series.
        This attention vector is then used as a weight vector to choose special values
        from the input data. This data is then finally concatenated to the next input time step's
        data. On this a linear transformation in the same space as the input data's space
        is performed before the data is fed into the RNN cell again.
        This technique is similar to the input-feeding method described in the paper cited
    """

    def __init__(self, layer, weight_initializer="glorot_uniform", **kwargs):
        assert isinstance(layer, RNN)
        self.layer = layer
        self.supports_masking = True
        self.weight_initializer = weight_initializer

        super(AttentionRNNWrapper, self).__init__(layer, **kwargs)

    def _validate_input_shape(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "Layer received an input with shape {0} but expected a Tensor of rank 3.".format(
                    input_shape[0]))

    def build(self, input_shape):
        self._validate_input_shape(input_shape)

        self.input_spec = InputSpec(shape=input_shape)

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True

        input_dim = input_shape[-1]

        if self.layer.return_sequences:
            output_dim = self.layer.compute_output_shape(input_shape)[0][-1]
        else:
            output_dim = self.layer.compute_output_shape(input_shape)[-1]

        input_dim = input_dim.value
        output_dim = output_dim.value

        self._W1 = self.add_weight(shape=(input_dim, input_dim), name="{}_W1".format(self.name),
                                   initializer=self.weight_initializer)
        self._W2 = self.add_weight(shape=(output_dim, input_dim), name="{}_W2".format(self.name),
                                   initializer=self.weight_initializer)
        self._W3 = self.add_weight(shape=(2 * input_dim, input_dim), name="{}_W3".format(self.name),
                                   initializer=self.weight_initializer)
        self._b2 = self.add_weight(shape=(input_dim,), name="{}_b2".format(self.name),
                                   initializer=self.weight_initializer)
        self._b3 = self.add_weight(shape=(input_dim,), name="{}_b3".format(self.name),
                                   initializer=self.weight_initializer)
        self._V = self.add_weight(shape=(input_dim, 1), name="{}_V".format(self.name),
                                  initializer=self.weight_initializer)

        super(AttentionRNNWrapper, self).build()

    def compute_output_shape(self, input_shape):
        self._validate_input_shape(input_shape)

        return self.layer.compute_output_shape(input_shape)

    @property
    def trainable_weights(self):
        return self._trainable_weights + self.layer.trainable_weights

    @property
    def non_trainable_weights(self):
        return self._non_trainable_weights + self.layer.non_trainable_weights

    def step(self, x, states):
        h = states[1]
        # states[1] necessary?

        # equals K.dot(X, self._W1) + self._b2 with X.shape=[bs, T, input_dim]
        total_x_prod = states[-1]
        # comes from the constants (equals the input sequence)
        X = states[-2]

        # expand dims to add the vector which is only valid for this time step
        # to total_x_prod which is valid for all time steps
        hw = K.expand_dims(K.dot(h, self._W2), 1)
        additive_atn = total_x_prod + hw
        attention = K.softmax(K.dot(additive_atn, self._V), axis=1)
        x_weighted = K.sum(attention * X, [1])

        x = K.dot(K.concatenate([x, x_weighted], 1), self._W3) + self._b3

        h, new_states = self.layer.cell.call(x, states[:-2])

        return h, new_states

    def call(self, x, constants=None, mask=None, initial_state=None):
        # input shape: (n_samples, time (padded with zeros), input_dim)
        input_shape = self.input_spec.shape

        if self.layer.stateful:
            initial_states = self.layer.states
        elif initial_state is not None:
            initial_states = initial_state
            if not isinstance(initial_states, (list, tuple)):
                initial_states = [initial_states]

            base_initial_state = self.layer.get_initial_state(x)
            if len(base_initial_state) != len(initial_states):
                raise ValueError(
                    "initial_state does not have the correct length. Received length {0} "
                    "but expected {1}".format(len(initial_states), len(base_initial_state)))
            else:
                # check the state' shape
                for i in range(len(initial_states)):
                    # initial_states[i][j] != base_initial_state[i][j]:
                    if not initial_states[i].shape.is_compatible_with(base_initial_state[i].shape):
                        raise ValueError(
                            "initial_state does not match the default base state of the layer. "
                            "Received {0} but expected {1}".format(
                                [x.shape for x in initial_states],
                                [x.shape for x in base_initial_state]))
        else:
            initial_states = self.layer.get_initial_state(x)

        # print(initial_states)

        if not constants:
            constants = []

        constants += self.get_constants(x)

        last_output, outputs, states = K.rnn(
            self.step,
            x,
            initial_states,
            go_backwards=self.layer.go_backwards,
            mask=mask,
            constants=constants,
            unroll=self.layer.unroll,
            input_length=input_shape[1]
        )

        if self.layer.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.layer.states[i], states[i]))

        if self.layer.return_sequences:
            output = outputs
        else:
            output = last_output

            # Properly set learning phase
        if getattr(last_output, '_uses_learning_phase', False):
            output._uses_learning_phase = True
            for state in states:
                state._uses_learning_phase = True

        if self.layer.return_state:
            if not isinstance(states, (list, tuple)):
                states = [states]
            else:
                states = list(states)
            return [output] + states
        else:
            return output

    def get_constants(self, x):
        # add constants to speed up calculation
        constants = [x, K.dot(x, self._W1) + self._b2]

        return constants

    def get_config(self):
        config = {'weight_initializer': self.weight_initializer}
        base_config = super(AttentionRNNWrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MTNetKeras(BaseModel):

    def __init__(self, check_optional_config=False, future_seq_len=1):

        """
        Constructor of MTNet model
        """
        self.check_optional_config = check_optional_config
        self.config = None
        # config parameter
        self.time_step = None  # timestep
        self.cnn_height = None  # convolution window size (convolution filter height)` ?
        self.long_num = None  # the number of the long-term memory series
        self.ar_window = None  # the window size of ar model
        self.feature_num = None  # input's variable dimension (convolution filter width)
        self.output_dim = None  # output's variable dimension
        self.cnn_hid_size = None
        # last size is equal to en_conv_hidden_size, should be a list
        self.rnn_hid_sizes = None
        self.last_rnn_size = None
        self.cnn_dropout = None
        self.rnn_dropout = None
        self.lr = None
        self.batch_size = None
        self.loss = None

        self.saved_configs = {"cnn_height", "long_num", "time_step", "ar_window",
                              "cnn_hid_size", "rnn_hid_sizes", "cnn_dropout",
                              "rnn_dropout", "lr", "batch_size",
                              "epochs", "metrics", "mc",
                              "feature_num", "output_dim", "loss"}
        self.model = None
        self.metrics = None
        self.mc = None
        self.epochs = None

    def apply_config(self, rs=False, config=None):
        super()._check_config(**config)
        if rs:
            config_names = set(config.keys())
            assert config_names.issuperset(self.saved_configs)
            # assert config_names.issuperset(self.lr_decay_configs) or \
            #        config_names.issuperset(self.lr_configs)
        self.epochs = config.get("epochs")
        self.metrics = config.get("metrics", ["mean_squared_error"])
        self.mc = config.get("mc")
        self.feature_num = config["feature_num"]
        self.output_dim = config["output_dim"]
        self.time_step = config.get("time_step", 1)
        self.long_num = config.get("long_num", 7)
        self.ar_window = config.get("ar_window", 1)
        self.cnn_height = config.get("cnn_height", 1)
        self.cnn_hid_size = config.get("cnn_hid_size", 32)
        self.rnn_hid_sizes = config.get("rnn_hid_sizes", [16, 32])
        self.last_rnn_size = self.rnn_hid_sizes[-1]
        self.rnn_dropout = config.get("rnn_dropout", 0.2)
        self.cnn_dropout = config.get("cnn_dropout", 0.2)
        self.loss = config.get('loss', "mae")
        self.batch_size = config.get("batch_size", 64)
        self.lr = config.get('lr', 0.001)
        self._check_configs()

    def _check_configs(self):
        assert self.time_step >= 1, \
            "Invalid configuration value. 'time_step' must be larger than 1"
        assert self.time_step >= self.ar_window, \
            "Invalid configuration value. 'ar_window' must not exceed 'time_step'"
        assert isinstance(self.rnn_hid_sizes, list), \
            "Invalid configuration value. 'rnn_hid_sizes' must be a list of integers"
        # assert self.cnn_hid_size == self.last_rnn_size,\
        #     "Invalid configuration value. 'cnn_hid_size' must be equal to the last element of " \
        #     "'rnn_hid_sizes'"

    def build(self):
        """
        build MTNet model
        :param config:
        :return:
        """
        training = True if self.mc else None
        # long-term time series historical data inputs
        long_input = Input(shape=(self.long_num, self.time_step, self.feature_num))
        # short-term time series historical data
        short_input = Input(shape=(self.time_step, self.feature_num))

        # ------- no-linear component----------------
        # memory and context : (batch, long_num, last_rnn_size)
        memory = self.__encoder(long_input, num=self.long_num, name='memory', training=training)
        # memory = memory_model(long_input)
        context = self.__encoder(long_input, num=self.long_num, name='context', training=training)
        # context = context_model(long_input)
        # query: (batch, 1, last_rnn_size)
        query_input = Reshape((1, self.time_step, self.feature_num),
                              name='reshape_query')(short_input)
        query = self.__encoder(query_input, num=1, name='query', training=training)
        # query = query_model(query_input)

        # prob = memory * query.T, shape is (long_num, 1)
        query_t = Permute((2, 1))(query)
        prob = Lambda(lambda xy: tf.matmul(xy[0], xy[1]))([memory, query_t])
        prob = Softmax(axis=-1)(prob)
        # out is of the same shape of context: (batch, long_num, last_rnn_size)
        out = multiply([context, prob])
        # concat: (batch, long_num + 1, last_rnn_size)

        pred_x = concatenate([out, query], axis=1)
        reshaped_pred_x = Reshape((self.last_rnn_size * (self.long_num + 1),),
                                  name="reshape_pred_x")(pred_x)
        nonlinear_pred = Dense(units=self.output_dim,
                               kernel_initializer=TruncatedNormal(stddev=0.1),
                               bias_initializer=Constant(0.1),)(reshaped_pred_x)

        # ------------ ar component ------------
        if self.ar_window > 0:
            ar_pred_x = Reshape((self.ar_window * self.feature_num,),
                                name="reshape_ar")(short_input[:, -self.ar_window:])
            linear_pred = Dense(units=self.output_dim,
                                kernel_initializer=TruncatedNormal(stddev=0.1),
                                bias_initializer=Constant(0.1),)(ar_pred_x)
        else:
            linear_pred = 0
        y_pred = Add()([nonlinear_pred, linear_pred])
        self.model = Model(inputs=[long_input, short_input], outputs=y_pred)
        # lr decay
        # def lr_scheduler(epoch, r):
        #     max_lr = 0.03
        #     min_lr = 0.0001
        #     lr = min_lr + (max_lr - min_lr) * math.exp(-epoch / 60)
        #     return lr
        # callbacks = [tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)]
        # initial_lr = 0.003
        # rate = math.exp(-1 / 60)
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_lr,
        #     decay_steps=249,
        #     decay_rate=rate,
        #     staircase=True
        # )
        #
        # self.model.compile(loss="mae",
        #                    metrics=metrics,
        #                    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule))

        self.model.compile(loss=self.loss,
                           metrics=self.metrics,
                           optimizer=tf.keras.optimizers.Adam(lr=self.lr))

        return self.model

    def __encoder(self, input, num, name='Encoder', training=None):
        """
            Treat batch_size dimension and num dimension as one batch_size dimension
            (batch_size * num).
        :param input:  <batch_size, num, time_step, input_dim>
        :param num: the number of input time series data. For short term data, the num is 1.
        :return: the embedded of the input <batch_size, num, last_rnn_hid_size>
        """
        # input = Input(shape=(num, self.time_step, self.feature_num))
        batch_size_new = self.batch_size * num
        Tc = self.time_step - self.cnn_height + 1

        # CNN
        # reshaped input: (batch_size_new, time_step, feature_num, 1)
        reshaped_input = Lambda(lambda x:
                                K.reshape(x, (-1, self.time_step, self.feature_num, 1),),
                                name=name+'reshape_cnn')(input)
        # output: <batch_size_new, conv_out, 1, en_conv_hidden_size>
        cnn_out = Conv2D(filters=self.cnn_hid_size,
                         kernel_size=(self.cnn_height, self.feature_num),
                         padding="valid",
                         kernel_initializer=TruncatedNormal(stddev=0.1),
                         bias_initializer=Constant(0.1),
                         activation="relu")(reshaped_input)
        cnn_out = Dropout(self.cnn_dropout)(cnn_out, training=training)

        rnn_input = Lambda(lambda x:
                           K.reshape(x, (-1, num, Tc, self.cnn_hid_size)),)(cnn_out)

        # use AttentionRNNWrapper
        rnn_cells = [GRUCell(h_size, activation="relu", dropout=self.rnn_dropout)
                     for h_size in self.rnn_hid_sizes]

        attention_rnn = AttentionRNNWrapper(RNN(rnn_cells),
                                            weight_initializer=TruncatedNormal(stddev=0.1))

        outputs = []
        for i in range(num):
            input_i = rnn_input[:, i]
            # input_i = (batch, conv_hid_size, Tc)
            input_i = Permute((2, 1), input_shape=[Tc, self.cnn_hid_size])(input_i)
            # output = (batch, last_rnn_hid_size)
            output_i = attention_rnn(input_i, training=training)
            # output = (batch, 1, last_rnn_hid_size)
            output_i = Reshape((1, -1))(output_i)
            outputs.append(output_i)
        if len(outputs) > 1:
            output = Lambda(lambda x: concatenate(x, axis=1))(outputs)
        else:
            output = outputs[0]
        return output

    def _reshape_input_x(self, x):
        long_term = np.reshape(x[:, : self.time_step * self.long_num],
                               [-1, self.long_num, self.time_step, x.shape[-1]])
        short_term = np.reshape(x[:, self.time_step * self.long_num:],
                                [-1, self.time_step, x.shape[-1]])
        return long_term, short_term

    def _pre_processing(self, x, validation_data=None):
        long_term, short_term = self._reshape_input_x(x)
        if validation_data:
            val_x, val_y = validation_data
            long_val, short_val = self._reshape_input_x(val_x)
            validation_data = ([long_val, short_val], val_y)
        return [long_term, short_term], validation_data

    def _add_config_attributes(self, config, **new_attributes):
        # new_attributes are among ["metrics", "epochs", "mc", "feature_num", "output_dim"]
        if self.config is None:
            self.config = config
        else:
            if config:
                raise ValueError("You can only pass new configuations for 'mc', 'epochs' and "
                                 "'metrics' during incremental fitting. "
                                 "Additional configs passed are {}".format(config))

        if new_attributes["metrics"] is None:
            del new_attributes["metrics"]
        self.config.update(new_attributes)

    def _check_input(self, x, y):
        input_feature_num = x.shape[-1]
        input_output_dim = y.shape[-1]
        if input_feature_num is None:
            raise ValueError("input x is None!")
        if input_output_dim is None:
            raise ValueError("input y is None!")

        if self.feature_num is not None and self.feature_num != input_feature_num:
            raise ValueError("input x has different feature number (the shape of last dimension) "
                             "{} with the fitted model, which is {}."
                             .format(input_feature_num, self.feature_num))
        if self.output_dim is not None and self.output_dim != input_output_dim:
            raise ValueError("input y has different prediction size (the shape of last dimension) "
                             "of {} with the fitted model, which is {}."
                             .format(input_output_dim, self.output_dim))
        return input_feature_num, input_output_dim

    def fit_eval(self, x, y, validation_data=None, mc=False, metrics=None,
                 epochs=10, verbose=0, **config):
        feature_num, output_dim = self._check_input(x, y)
        self._add_config_attributes(config, epochs=epochs, mc=mc, metrics=metrics,
                                    feature_num=feature_num, output_dim=output_dim)
        self.apply_config(config=self.config)
        processed_x, processed_validation_data = self._pre_processing(x, validation_data)

        # if model is not initialized, __build the model
        if self.model is None:
            st = time.time()
            self.build()
            end = time.time()
            if verbose == 1:
                print("Build model took {}s".format(end - st))

        st = time.time()
        hist = self.model.fit(processed_x, y, validation_data=processed_validation_data,
                              batch_size=self.batch_size,
                              epochs=self.epochs,
                              verbose=verbose)

        if verbose == 1:
            print("Fit model took {}s".format(time.time() - st))
        if validation_data is None:
            # get train metrics
            # results = self.model.evaluate(x, y)
            result = hist.history.get(self.metrics[0])[-1]
        else:
            result = hist.history.get('val_' + str(self.metrics[0]))[-1]
        return result

    def evaluate(self, x, y, metrics=['mse']):
        """
        Evaluate on x, y
        :param x: input
        :param y: target
        :param metric: a list of metrics in string format
        :return: a list of metric evaluation results
        """
        y_pred = self.predict(x)
        if y_pred.shape[1] == 1:
            multioutput = 'uniform_average'
        else:
            multioutput = 'raw_values'
        # y = np.squeeze(y, axis=2)
        return [Evaluator.evaluate(m, y, y_pred, multioutput=multioutput) for m in metrics]

    def predict(self, x, mc=False):
        input_x = self._reshape_input_x(x)
        return self.model.predict(input_x)

    def predict_with_uncertainty(self, x, n_iter=100):
        result = np.zeros((n_iter,) + (x.shape[0], self.output_dim))

        for i in range(n_iter):
            result[i, :, :] = self.predict(x, mc=True)

        prediction = result.mean(axis=0)
        uncertainty = result.std(axis=0)
        return prediction, uncertainty

    def save(self, model_path, config_path):
        self.model.save_weights(model_path)
        config_to_save = {"cnn_height": self.cnn_height,
                          "long_num": self.long_num,
                          "time_step": self.time_step,
                          "ar_window": self.ar_window,
                          "cnn_hid_size": self.cnn_hid_size,
                          "rnn_hid_sizes": self.rnn_hid_sizes,
                          "cnn_dropout": self.cnn_dropout,
                          "rnn_dropout": self.rnn_dropout,
                          "lr": self.lr,
                          "batch_size": self.batch_size,
                          # for fit eval
                          "epochs": self.epochs,
                          # todo: can not serialize metrics unless all elements are str
                          "metrics": self.metrics,
                          "mc": self.mc,
                          "feature_num": self.feature_num,
                          "output_dim": self.output_dim,
                          "loss": self.loss
                          }
        assert set(config_to_save.keys()) == self.saved_configs, \
            "The keys in config_to_save is not the same as self.saved_configs." \
            "Please keep them consistent"
        # if self.decay_epochs > 0:
        #     lr_decay_configs = {"min_lr": self.min_lr,
        #                         "max_lr": self.max_lr}
        #     assert set(lr_decay_configs.keys()) == self.lr_decay_configs, \
        #         "The keys in lr_decay_configs is not the same as self.lr_decay_configs." \
        #         "Please keep them consistent"
        #     config_to_save.update(lr_decay_configs)
        # else:
        #     lr_configs = {"lr": self.lr_value}
        #     assert set(lr_configs.keys()) == self.lr_configs, \
        #         "The keys in lr_configs is not the same as self.lr_configs." \
        #         "Please keep them consistent"
        #     config_to_save.update(lr_configs)

        save_config(config_path, config_to_save)

    def restore(self, model_path, **config):
        """
        restore model from file
        :param model_path: the model file
        :param config: the trial config
        """
        self.config = config
        self.apply_config(rs=True, config=config)
        self.build()
        self.model.load_weights(model_path)

    def _get_optional_parameters(self):
        return {
            "batch_size",
            "cnn_dropout",
            "rnn_dropout",
            "time_step",
            "cnn_height",
            "long_num",
            "ar_size",
            "loss",
            "cnn_hid_size",
            "rnn_hid_sizes",
            "lr"
        }

    def _get_required_parameters(self):
        return {
            "feature_num",
            "output_dim"
        }
