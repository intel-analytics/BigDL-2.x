import numpy as np
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import TruncatedNormal, Constant
import tensorflow.keras.backend as K
import math

# from zoo.automl.common.util import *
# from zoo.automl.common.metrics import Evaluator
# from tensorflow.python.ops import array_ops
# from tensorflow.python.util import nest
# from tensorflow.python.ops import nn_ops
# from tensorflow.python.ops import math_ops
import tensorflow as tf
# _Linear = core_rnn_cell._Linear
from zoo.automl.common.util import *
from zoo.automl.common.metrics import Evaluator
import pandas as pd
from zoo.automl.model.abstract import BaseModel


class AttentionRNNWrapper(Wrapper):
    """
        The AttentionRNNWrapper is a modified version of
        https://github.com/zimmerrol/keras-utility-layer-collection/blob/master/kulc/attention.py
        The idea of the implementation is based on the paper:
            "Effective Approaches to Attention-based Neural Machine Translation" by Luong et al.
        This layer is an attention layer, which can be wrapped around arbitrary RNN layers.
        This way, after each time step an attention vector is calculated
        based on the current output of the LSTM and the entire input time series.
        This attention vector is then used as a weight vector to choose special values
        from the input data. This data is then finally concatenated to the next input
        time step's data. On this a linear transformation in the same space as the input data's space
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
                    "initial_state does not have the correct length. Received length {0} but expected {1}".format(
                        len(initial_states), len(base_initial_state)))
            else:
                # check the state' shape
                for i in range(len(initial_states)):
                    if not initial_states[i].shape.is_compatible_with(base_initial_state[
                                                                          i].shape):  # initial_states[i][j] != base_initial_state[i][j]:
                        raise ValueError(
                            "initial_state does not match the default base state of the layer. Received {0} but expected {1}".format(
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


class AttentionRNN(Layer):
    def __init__(self, hidden_sizes, input_keep_prob, output_keep_prob, num, batch_size, Tc, **kwargs):
        super(AttentionRNN, self).__init__(**kwargs)
        # # tensorflow
        # rnns = [tf.nn.rnn_cell.GRUCell(h_size, activation=tf.nn.relu) for h_size in
        #         hidden_sizes]
        # # dropout
        # if input_keep_prob < 1 or output_keep_prob < 1:
        #     rnns = [tf.nn.rnn_cell.DropoutWrapper(rnn,
        #                                           input_keep_prob=input_keep_prob,
        #                                           output_keep_prob=output_keep_prob)
        #             for rnn in rnns]
        #
        # if len(rnns) > 1:
        #     self.rnns = tf.nn.rnn_cell.MultiRNNCell(rnns)
        # else:
        #     self.rnns = rnns[0]
        # keras
        rnn_cells = [GRUCell(h_size, activation="relu", dropout=1 - input_keep_prob)
                     for h_size in hidden_sizes]
        # output dropout of GRUCell is omitted since the output_keep_prob is 1 in all configs.
        self.rnns = StackedRNNCells(rnn_cells)
        self.hidden_sizes = hidden_sizes
        self.last_rnn_size = hidden_sizes[-1]
        self.num = num
        self.Tc = Tc

    def build(self, input_shape):
        # input_shape is <batch_size, Tc, hidden_size>
        Tc = input_shape[1]
        # hidden size equals self.last_rnn_size
        hidden_size = input_shape[2]
        self.v = self.add_weight(shape=(self.Tc, 1),
                                 initializer=TruncatedNormal(stddev=0.1),
                                 name="att_v")
        self.w = self.add_weight(shape=(self.last_rnn_size, self.Tc),
                                 initializer=TruncatedNormal(stddev=0.1),
                                 name="att_w")
        self.u = self.add_weight(shape=(self.Tc, self.Tc),
                                 initializer=TruncatedNormal(stddev=0.1),
                                 name="att_u")
        self.rnns.build(input_shape=[self.last_rnn_size])
        self.built = True

    def call(self, input):
        # print(input.shape)
        batch_size = tf.shape(input)[0]
        res_hstates = tf.TensorArray(tf.float32, self.num)
        for k in range(self.num):
            # <batch_size, en_conv_hidden_size, Tc>
            attr_input = tf.transpose(input[:, k], perm=[0, 2, 1])

            # <batch_size, last_rnn_hidden_size>
            # s_state = self.rnns.zero_state(self.batch_size, tf.float32)

            s_state = self.rnns.get_initial_state(batch_size=batch_size, dtype=tf.float32)
            if len(self.hidden_sizes) > 1:
                h_state = s_state[-1]
            else:
                h_state = s_state

            for t in range(self.Tc):
                # h(t-1) dot attr_w
                h_part = tf.matmul(h_state, self.w)
                # en_conv_hidden_size * <batch_size_new, 1>
                e_ks = tf.TensorArray(tf.float32, self.last_rnn_size)
                _, output = tf.while_loop(
                    lambda i, _: tf.less(i, self.last_rnn_size),
                    lambda i, output_ta: (i + 1, output_ta.write(i, tf.matmul(
                        tf.tanh(h_part + tf.matmul(attr_input[:, i], self.u)), self.v))),
                    [0, e_ks])
                # <batch_size, en_conv_hidden_size, 1>
                e_ks = tf.transpose(output.stack(), perm=[1, 0, 2])
                e_ks = tf.reshape(e_ks, shape=[-1, self.last_rnn_size])

                # <batch_size, en_conv_hidden_size>
                a_ks = tf.nn.softmax(e_ks)

                x_t = tf.matmul(tf.expand_dims(attr_input[:, :, t], -2), tf.matrix_diag(a_ks))
                # <batch_size, en_conv_hidden_size>
                x_t = tf.reshape(x_t, shape=[-1, self.last_rnn_size])

                h_state, s_state = self.rnns(x_t, s_state)

            res_hstates = res_hstates.write(k, h_state)

        return tf.transpose(res_hstates.stack(), perm=[1, 0, 2])


class MTNetKeras:

    def __init__(self, name='MTNet'):

        """
        Constructor of MTNet model
        """
        # self.config = config
        # self._get_configs()
        self.config = None
        # config parameter
        self.time_step = None  # timestep
        self.cnn_height = None  # convolution window size (convolution filter height)` ?
        self.long_num = None  # the number of the long-term memory series
        self.ar_window = None  # the window size of ar model
        self.feature_num = None  # input's variable dimension (convolution filter width)
        self.output_dim = None # output's variable dimension
        self.cnn_hid_size = None
        # last size is equal to en_conv_hidden_size, should be a list
        self.rnn_hid_sizes = None
        self.last_rnn_size = None
        self.dropout = None
        self.lr = None
        self.batch_size = None

        self.saved_configs = {"cnn_height", "long_num", "time_step", "ar_window",
                              "cnn_hid_size", "rnn_hid_sizes", "dropout", "lr", "batch_size", "epochs",
                              "feature_num", "output_dim"}
        self.model = None
        self.metric = None

    def _get_configs(self, rs=False, config=None):
        if rs:
            config_names = set(config.keys())
            assert config_names.issuperset(self.saved_configs)
            # assert config_names.issuperset(self.lr_decay_configs) or \
            #        config_names.issuperset(self.lr_configs)
            self.feature_num = config.get("feature_num")
            self.output_dim = config.get("output_dim")
        self.time_step = config.get("time_step", 1)
        self.long_num = config.get("long_num", 7)
        self.ar_window = config.get("ar_window", 1)
        self.cnn_height = config.get("cnn_height", 1)
        self.cnn_hid_size = config.get("cnn_hid_size", 32)
        self.rnn_hid_sizes = config.get("rnn_hid_sizes", [16, 32])
        self.last_rnn_size = self.rnn_hid_sizes[-1]
        self.dropout = config.get("dropout", 0.2)

        self.batch_size = config.get("batch_size", 64)
        self.lr = config.get('lr', 0.001)
        self.epochs = config.get('epochs', 10)
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

    def _get_len(self, x, y):
        self.feature_num = x.shape[-1]
        self.output_dim = y.shape[-1]

    def _build_train(self, mc=False, metrics=None):
        """
        build MTNet model
        :param config:
        :return:
        """
        # long-term time series historical data inputs
        long_input = Input(shape=(self.long_num, self.time_step, self.feature_num))
        # short-term time series historical data
        short_input = Input(shape=(self.time_step, self.feature_num))

        # ------- no-linear component----------------
        # memory and context : (batch, long_num, last_rnn_size)
        memory = self.__encoder(long_input, num=self.long_num, name='memory')
        # memory = memory_model(long_input)
        context = self.__encoder(long_input, num=self.long_num, name='context')
        # context = context_model(long_input)
        # query: (batch, 1, last_rnn_size)
        query_input = Reshape((1, self.time_step, self.feature_num), name='reshape_query')(short_input)
        query = self.__encoder(query_input, num=1, name='query')
        # query = query_model(query_input)

        # prob = memory * query.T, shape is (long_num, 1)
        query_t = Permute((2, 1))(query)
        prob = Lambda(lambda xy: tf.matmul(xy[0], xy[1]))([memory, query_t])
        prob = Softmax(axis=-1)(prob)
        # out is of the same shape of context: (batch, long_num, last_rnn_size)
        out = multiply([context, prob])
        # concat: (batch, long_num + 1, last_rnn_size)

        pred_x = concatenate([out, query], axis=1)
        reshaped_pred_x = Reshape((self.last_rnn_size * (self.long_num + 1),), name="reshape_pred_x")(pred_x)
        nonlinear_pred = Dense(units=self.output_dim,
                               kernel_initializer=TruncatedNormal(stddev=0.1),
                               bias_initializer=Constant(0.1),)(reshaped_pred_x)

        # ------------ ar component ------------
        if self.ar_window > 0:
            ar_pred_x = Reshape((self.ar_window * self.feature_num,), name="reshape_ar")(short_input[:, -self.ar_window:])
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
        initial_lr = 0.003
        rate = math.exp(-1 / 60)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_lr,
            decay_steps=249,
            decay_rate=rate,
            staircase=True
        )

        self.model.compile(loss="mae",
                           metrics=metrics,
                           optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule))

        # self.model.compile(loss="mae",
        #                    metrics=metrics,
        #                    optimizer=tf.keras.optimizers.Adam(lr=self.lr))

        return self.model

    def __encoder(self, input, num, name='Encoder'):
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
        cnn_out = Dropout(self.dropout)(cnn_out)

        rnn_input = Lambda(lambda x:
                                K.reshape(x, (-1, num, Tc, self.cnn_hid_size)),)(cnn_out)

        # hidden_sizes, input_keep_prob, output_keep_prob, num, batch_size, Tc
        output = AttentionRNN(hidden_sizes=self.rnn_hid_sizes,
                              input_keep_prob=1-self.dropout,
                              output_keep_prob=1,
                              num=num,
                              Tc=Tc,
                              batch_size=self.batch_size,
                              name=name + 'attention_rnn')(rnn_input)
        return output

        # rnn inputs
        # <batch_size, n, conv_out, en_conv_hidden_size>
        # rnn_input = Reshape((num, Tc, self.cnn_hid_size), name="first_reshape_{}".format(np.random.random_integers(0, 100)))(cnn_out)

        # rnn_cells = [GRUCell(h_size, activation="relu", dropout=self.dropout)
        #              for h_size in self.rnn_hid_sizes]
        # test_cell = GRUCell(self.last_rnn_size, activation="relu", dropout=self.dropout)
        # # output dropout of GRUCell is omitted since the output_keep_prob is 1 in all configs.
        # # rnn_cell = StackedRNNCells(rnn_cells)
        # # attention_rnn_cell = AttentionCellWrapper(test_cell)
        # # attention_rnn = RNN(attention_rnn_cell)
        #
        # attention_rnn = AttentionRNNWrapper(RNN(rnn_cells),
        #                                     weight_initializer=TruncatedNormal(stddev=0.1))
        #
        # outputs = []
        # for i in range(num):
        #     input_i = rnn_input[:, i]
        #     # input_i = (batch, conv_hid_size, Tc)
        #     input_i = Permute((2, 1), input_shape=[Tc, self.cnn_hid_size])(input_i)
        #     # output = (batch, last_rnn_hid_size)
        #     output_i = attention_rnn(input_i)
        #     # output = (batch, 1, last_rnn_hid_size)
        #     output_i = Reshape((1, -1))(output_i)
        #     outputs.append(output_i)
        # if len(outputs) > 1:
        #     output = Lambda(lambda x: concatenate(x, axis=1))(outputs)
        #     # print(output.shape)
        # else:
        #     output = outputs[0]
        # # encoder_model = Model(input, output, name='Encoder' + name)
        # return output

    def _gen_hist_inputs(self, x):
        long_term = np.reshape(x[:, : self.time_step * self.long_num],
                               [-1, self.long_num, self.time_step, self.feature_num])
        short_term = np.reshape(x[:, self.time_step * self.long_num:],
                                [-1, self.time_step, self.feature_num])
        return long_term, short_term

    def _pre_processing(self, x, y, validation_data):
        self._get_len(x, y)
        long_term, short_term = self._gen_hist_inputs(x)
        if validation_data:
            val_x, val_y = validation_data
            long_val, short_val = self._gen_hist_inputs(val_x)
            validation_data = ([long_val, short_val], val_y)
        return [long_term, short_term], y, validation_data

    def fit_eval(self, x, y, validation_data=None, mc=False, metrics=None,
                 epochs=100, verbose=0, config=None):
        x, y, validation_data = self._pre_processing(x, y, validation_data)
        if metrics is None:
            metrics = ['mean_absolute_error', tf.keras.metrics.RootMeanSquaredError()]
        self.config = config
        self._get_configs(config=self.config)
        # x, y, validation_data = self._pre_processing(x, y, validation_data)
        # if model is not initialized, __build the model
        if self.model is None:
            st = time.time()
            self._build_train(mc=mc, metrics=metrics)
            end = time.time()
            print("Build model took {}s".format(end - st))

        # st = time.time()
        # self.model.compile(loss="mae",
        #                    metrics=metrics,
        #                    optimizer=tf.keras.optimizers.Adam(lr=self.lr))
        # print("Compile model took {}s".format(time.time() - st))
        st = time.time()
        hist = self.model.fit(x, y, validation_data=validation_data,
                              batch_size=self.batch_size,
                              epochs=epochs)
        #
        # hist = self.model.fit_generator(generator=train_data,
        #                                 validation_data=validation_data,
        #                                 )
        print("Fit model took {}s".format(time.time() - st))
        if validation_data is None:
            # get train metrics
            # results = self.model.evaluate(x, y)
            result = hist.history.get(metrics[0])[-1]
        else:
            result = hist.history.get('val_' + str(metrics[0]))[-1]
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
        # input_x = self._gen_hist_inputs(x)
        # return self.model.predict(input_x)
        return self.model.predict(x)

    def save(self, model_path, config_path):
        self.model.save(model_path)
        config_to_save = {"cnn_height": self.cnn_height,
                          "long_num": self.long_num,
                          "time_step": self.time_step,
                          "ar_window": self.ar_window,
                          "cnn_hid_size": self.cnn_hid_size,
                          "rnn_hid_sizes": self.rnn_hid_sizes,
                          "dropout": self.dropout,
                          "lr": self.lr,
                          "batch_size": self.batch_size,
                          "epochs": self.epochs,
                          "feature_num": self.feature_num,
                          "output_dim": self.output_dim
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
     if not self.built:
         # build model
         self._get_configs(rs=True, config=config)
         self.model = tf.keras.models.load_model(model_path)

     self.built = 1

    def _get_optional_configs(self):
        return {
            "batch_size",
            "dropout",
            "time_step",
            "filter_size",
            "long_num",
            "ar_size",
        }


# if __name__ == "__main__":
#     from zoo.automl.feature.time_sequence import TimeSequenceFeatureTransformer
#     from zoo.automl.common.util import split_input_df
#
#     import os
#     import pandas as pd
#
#     dataset_path = os.path.join("/home/shan/sources/automl-analytics-zoo/dist",
#                                 "bin/data/NAB/nyc_taxi/nyc_taxi.csv")
#     df = pd.read_csv(dataset_path)
#     # df = pd.read_csv('automl/data/nyc_taxi.csv')
#     future_seq_len = 1
#     model = MTNetKeras(check_optional_config=False, future_seq_len=future_seq_len)
#     train_df, val_df, test_df = split_input_df(df, val_split_ratio=0.1, test_split_ratio=0.1)
#     feature_transformer = TimeSequenceFeatureTransformer(future_seq_len=future_seq_len)
#     config = {
#         'selected_features': ['IS_WEEKEND(datetime)', 'MONTH(datetime)', 'IS_AWAKE(datetime)',
#                               'HOUR(datetime)'],
#         'batch_size': 64,
#         'epochs': 5,
#         "time_step": 6,
#         "long_num": 10,
#         "cnn_height": 2,
#         'ar_window': 2,
#         'dropout': 0.2,
#         # past_seq_len = (n + 1) * T
#     }
#     config['past_seq_len'] = (config['long_num'] + 1) * config['time_step']
#     x_train, y_train = feature_transformer.fit_transform(train_df, **config)
#     x_val, y_val = feature_transformer.transform(val_df, is_train=True)
#     x_test, y_test = feature_transformer.transform(test_df, is_train=True)
#     # y_train = np.c_[y_train, y_train/2]
#     # y_test = np.c_[y_test, y_test/2]
#     for i in range(1):
#         print("fit_eval:", model.fit_eval(x_train, y_train, validation_data=(x_val, y_val), **config))
#
#     print("evaluate:", model.evaluate(x_test, y_test))
#     y_pred = model.predict(x_test)
#
#     # dirname = "tmp"
#     # model_1 = MTNet(check_optional_config=False)
#     # save(dirname, model=model)
#     # restore(dirname, model=model_1, config=config)
#     # predict_after = model_1.predict(x_test)
#     # assert np.allclose(y_pred, predict_after), \
#     #     "Prediction values are not the same after restore: " \
#     #     "predict before is {}, and predict after is {}".format(y_pred,
#     #                                                            predict_after)
#     # new_config = {'epochs': 1}
#     # for i in range(2):
#     #     model_1.fit_eval(x_train, y_train, **new_config)
#     #     print("evaluate:", model_1.evaluate(x_test, y_test))
#     # import shutil
#     # shutil.rmtree("tmp")
#
#
#     from matplotlib import pyplot as plt
#
#     y_test = np.squeeze(y_test)
#     y_pred = np.squeeze(y_pred)
#
#
#     def plot_result(y_test, y_pred):
#         # target column of dataframe is "value"
#         # past sequence length is 50
#         # pred_value = pred_df["value"].values
#         # true_value = test_df["value"].values[50:]
#         fig, axs = plt.subplots()
#
#         axs.plot(y_pred, color='red', label='predicted values')
#         axs.plot(y_test, color='blue', label='actual values')
#         axs.set_title('the predicted values and actual values (for the test data)')
#
#         plt.xlabel('test data index')
#         plt.ylabel('number of taxi passengers')
#         plt.legend(loc='upper left')
#         plt.savefig("MTNet_result_keras.png")
#
#
#     plot_result(y_test, y_pred)