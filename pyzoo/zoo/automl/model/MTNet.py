import time

import math

from zoo.automl.model.abstract import BaseModel
import numpy as np
import pandas as pd
from zoo.automl.common.metrics import Evaluator
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress some warnings
import tensorflow as tf
from zoo.automl.common.util import *


class MTNet(BaseModel):
    # source code from https://github.com/Maple728/MTNet

    def __init__(self, check_optional_config=True, future_seq_len=1):
        """
        Constructor, setting up necessary parameters
        """
        # config parameter
        self.future_seq_len = future_seq_len
        self.K = self.future_seq_len
        self.T = None  # timestep
        self.W = None  # convolution window size (convolution filter height)` ?
        self.n = None  # the number of the long-term memory series
        self.highway_window = None  # the window size of ar model
        self.D = None  # input's variable dimension (convolution filter width)
        self.en_conv_hidden_size = None
        # last size is equal to en_conv_hidden_size, should be a list
        self.en_rnn_hidden_sizes = None
        self.input_keep_prob_value = None
        self.output_keep_prob_value = None
        self.lr_value = None

        self.past_seq_len = None
        self.batch_size = None
        self.metric = None
        self.epochs = None
        self.decay_epochs = None
        self.max_lr = None
        self.min_lr = None

        self.built = 0  # indicating if a new model needs to be build
        self.sess = None  # the session

        # graph component assignment
        # placeholders
        self.X = None
        self.Q = None
        self.Y = None
        self.input_keep_prob = None
        self.output_keep_prob = None
        self.lr = None
        # non-placeholders
        self.y_pred = None
        self.loss = None
        self.train_op = None

        self.check_optional_config = check_optional_config
        self.saved_configs = {"en_conv_hidden_size", "en_rnn_hidden_sizes", "highway_window",
                              "input_keep_prob", "output_keep_prob", "D", "K", "T", "n", "W",
                              "past_seq_len", "metric", "epochs", "decay_epochs", "batch_size"}
        self.lr_decay_configs = {"min_lr", "max_lr"}
        self.lr_configs = {"lr"}

    def __encoder(self, origin_input_x, n, input_keep_prob, output_keep_prob,
                  strides=[1, 1, 1, 1], padding='VALID',
                  activation_func=tf.nn.relu, scope='Encoder'):
        """
            Treat batch_size dimension and n dimension as one batch_size dimension (batch_size * n).
        :param input_x:  <batch_size, n, T, D>
        :param strides: convolution stride
        :param padding: convolution padding
        :param scope: encoder scope
        :return: the embedded of the input_x <batch_size, n, last_rnn_hid_size>
        """
        # constant
        scope = 'Encoder_' + scope
        batch_size_new = self.batch_size * n
        Tc = self.T - self.W + 1
        last_rnn_hidden_size = self.en_rnn_hidden_sizes[-1]

        # reshape input_x : <batch_size * n, T, D, 1>
        input_x = tf.reshape(origin_input_x, shape=[-1, self.T, self.D, 1])

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # cnn parameters
            with tf.variable_scope('CNN', reuse=tf.AUTO_REUSE):
                w_conv1 = tf.get_variable('w_conv1',
                                          shape=[self.W, self.D, 1, self.en_conv_hidden_size],
                                          dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
                b_conv1 = tf.get_variable('b_conv1', shape=[self.en_conv_hidden_size],
                                          dtype=tf.float32,
                                          initializer=tf.constant_initializer(0.1))

                # <batch_size_new, Tc, 1, en_conv_hidden_size>
                h_conv1 = activation_func(
                    tf.nn.conv2d(input_x, w_conv1, strides, padding=padding) + b_conv1)
                # if input_keep_prob < 1:
                h_conv1 = tf.nn.dropout(h_conv1, input_keep_prob)

            # tmporal attention layer and gru layer
            # rnns
            rnns = [tf.nn.rnn_cell.GRUCell(h_size, activation=activation_func) for h_size in
                    self.en_rnn_hidden_sizes]
            # dropout
            # if input_keep_prob < 1 or output_keep_prob < 1:
            rnns = [tf.nn.rnn_cell.DropoutWrapper(rnn,
                                                  input_keep_prob=input_keep_prob,
                                                  output_keep_prob=output_keep_prob)
                    for rnn in rnns]

            if len(rnns) > 1:
                rnns = tf.nn.rnn_cell.MultiRNNCell(rnns)
            else:
                rnns = rnns[0]

            # attention layer

            # attention weights
            attr_v = tf.get_variable('attr_v', shape=[Tc, 1], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
            attr_w = tf.get_variable('attr_w', shape=[last_rnn_hidden_size, Tc], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
            attr_u = tf.get_variable('attr_u', shape=[Tc, Tc], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))

            # rnn inputs
            # <batch_size, n, Tc, en_conv_hidden_size>
            rnn_input = tf.reshape(h_conv1, shape=[-1, n, Tc, self.en_conv_hidden_size])

            # n * <batch_size, last_rnns_size>
            res_hstates = tf.TensorArray(tf.float32, n)
            for k in range(n):
                # <batch_size, en_conv_hidden_size, Tc>
                attr_input = tf.transpose(rnn_input[:, k], perm=[0, 2, 1])

                # <batch_size, last_rnn_hidden_size>
                s_state = rnns.zero_state(self.batch_size, tf.float32)
                if len(self.en_rnn_hidden_sizes) > 1:
                    h_state = s_state[-1]
                else:
                    h_state = s_state

                for t in range(Tc):
                    # attr_v = tf.Variable(tf.truncated_normal(shape=[Tc, 1],
                    #                                          stddev=0.1, dtype=tf.float32),
                    #                      name='attr_v')
                    # attr_w = tf.Variable(tf.truncated_normal(shape=[last_rnn_hidden_size, Tc],
                    #                                          stddev=0.1, dtype=tf.float32),
                    #                      name='attr_w')
                    # attr_u = tf.Variable(tf.truncated_normal(shape=[Tc, Tc], stddev=0.1,
                    #                                          dtype=tf.float32), name='attr_u')

                    # h(t-1) dot attr_w
                    h_part = tf.matmul(h_state, attr_w)

                    # en_conv_hidden_size * <batch_size_new, 1>
                    e_ks = tf.TensorArray(tf.float32, self.en_conv_hidden_size)
                    _, output = tf.while_loop(
                        lambda i, _: tf.less(i, self.en_conv_hidden_size),
                        lambda i, output_ta: (i + 1, output_ta.write(i, tf.matmul(
                            tf.tanh(h_part + tf.matmul(attr_input[:, i], attr_u)), attr_v))),
                        [0, e_ks])
                    # : tf.while_loop(cond, body, loop_vars ) :
                    # : TENSOR.write(idx, content) : write content at index
                    # <batch_size, en_conv_hidden_size, 1>
                    e_ks = tf.transpose(output.stack(), perm=[1, 0, 2])
                    e_ks = tf.reshape(e_ks, shape=[-1, self.en_conv_hidden_size])

                    # <batch_size, en_conv_hidden_size>
                    a_ks = tf.nn.softmax(e_ks)

                    x_t = tf.matmul(tf.expand_dims(attr_input[:, :, t], -2), tf.matrix_diag(a_ks))
                    # <batch_size, en_conv_hidden_size>
                    x_t = tf.reshape(x_t, shape=[-1, self.en_conv_hidden_size])

                    h_state, s_state = rnns(x_t, s_state)

                res_hstates = res_hstates.write(k, h_state)

        return tf.transpose(res_hstates.stack(), perm=[1, 0, 2])

    def _build(self, scope='MTNet', **config):
        """
        build the model 
        set up configuration parameter and graph components
        :config:
        """
        tf.reset_default_graph()
        tf.logging.set_verbosity(tf.logging.ERROR)  # suppress warnings during building
        with tf.variable_scope(scope, reuse=False):
            # placeholders
            X = tf.placeholder(tf.float32, shape=[None, self.n, self.T, self.D])
            Q = tf.placeholder(tf.float32, shape=[None, self.T, self.D])
            Y = tf.placeholder(tf.float32, shape=[None, self.K])
            lr = tf.placeholder(tf.float32)
            input_keep_prob = tf.placeholder(tf.float32)
            output_keep_prob = tf.placeholder(tf.float32)

            # ------- no-linear component----------------
            last_rnn_hid_size = self.en_rnn_hidden_sizes[-1]
            # <batch_size, n, en_rnn_hidden_sizes>
            m_is = self.__encoder(X, self.n, input_keep_prob, output_keep_prob, scope='m')
            c_is = self.__encoder(X, self.n, input_keep_prob, output_keep_prob, scope='c')
            # <batch_size, 1, en_rnn_hidden_sizes>
            u = self.__encoder(tf.reshape(Q, shape=[-1, 1, self.T, self.D]), 1,
                               input_keep_prob, output_keep_prob, scope='in')

            p_is = tf.matmul(m_is, tf.transpose(u, perm=[0, 2, 1]))

            # using softmax
            p_is = tf.squeeze(p_is, axis=[-1])
            p_is = tf.nn.softmax(p_is)
            # <batch_size, n, 1>
            p_is = tf.expand_dims(p_is, -1)

            # using sigmoid
            # p_is = tf.nn.sigmoid(p_is)

            # for summary
            # p_is_mean, _ = tf.metrics.mean_tensor(p_is, updates_collections='summary_ops',
            #                                       name='p_is')
            # tf.summary.histogram('p_is', p_is_mean)

            # <batch_size, n, en_rnn_hidden_sizes> =
            # <batch_size, n, en_rnn_hidden_sizes> * < batch_size, n, 1>
            o_is = tf.multiply(c_is, p_is)

            # :: last_rnn_hid_size * (self.n + 1) :: is the concatenation of (12)
            # following variables are a single layer vanilla net
            pred_w = tf.get_variable('pred_w', shape=[last_rnn_hid_size * (self.n + 1), self.K],
                                     dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
            pred_b = tf.get_variable('pred_b', shape=[self.K],
                                     dtype=tf.float32, initializer=tf.constant_initializer(0.1))

            pred_x = tf.concat([o_is, u], axis=1)
            pred_x = tf.reshape(pred_x, shape=[-1, last_rnn_hid_size * (self.n + 1)])

            # <batch_size, D>
            y_pred = tf.matmul(pred_x, pred_w) + pred_b

            # ------------ ar component ------------
            with tf.variable_scope('AutoRegression'):
                if self.highway_window > 0:  # highway_window is basically (s^ar -1)
                    highway_ws = tf.get_variable('highway_ws',
                                                 shape=[self.highway_window * self.D, self.K],
                                                 dtype=tf.float32,
                                                 initializer=tf.truncated_normal_initializer(
                                                     stddev=0.1))
                    highway_b = tf.get_variable('highway_b', shape=[self.K], dtype=tf.float32,
                                                initializer=tf.constant_initializer(0.1))

                    highway_x = tf.reshape(Q[:, -self.highway_window:],
                                           shape=[-1, self.highway_window * self.D])
                    y_pred_l = tf.matmul(highway_x, highway_ws) + highway_b

                    # y_pred_l = tf.matmul(Q[:, -1], highway_ws[0]) + highway_b
                    # _, y_pred_l = tf.while_loop(lambda i, _ : tf.less(i, self.highway_window),
                    #                             lambda i, acc : (i + 1,
                    #                                              tf.matmul(Q[:, self.T - i - 1],
                    #                                                        highway_ws[i]) + acc),
                    #                             loop_vars = [1, y_pred_l])
                    y_pred += y_pred_l

        # metrics summary
        # mae_loss, _ = tf.metrics.mean_absolute_error(Y, y_pred,
        #                                              updates_collections = 'summary_ops',
        #                                              name = 'mae_metric')
        # tf.summary.scalar('mae_loss', mae_loss)

        # rmse_loss, _ = tf.metrics.root_mean_squared_error(Y, y_pred,
        #                                                   updates_collections = 'summary_ops',
        #                                                   name = 'rmse_metric')
        # tf.summary.scalar("rmse_loss", rmse_loss)

        # statistics_vars = tf.get_collection(tf.GraphKeys.METRIC_VARIABLES)
        # statistics_vars_initializer = tf.variables_initializer(var_list = statistics_vars)

        loss = tf.losses.absolute_difference(Y, y_pred)
        with tf.name_scope('Train'):
            train_op = tf.train.AdamOptimizer(lr).minimize(loss)

        # assignment
        self.X = X
        self.Q = Q
        self.Y = Y
        self.input_keep_prob = input_keep_prob
        self.output_keep_prob = output_keep_prob
        self.lr = lr
        self.y_pred = y_pred
        self.loss = loss
        self.train_op = train_op

        self.built = 1
        # self.reset_statistics_vars = statistics_vars_initializer
        # self.merged_summary = tf.summary.merge_all()
        # self.summary_updates = tf.get_collection('summary_ops')

    def _get_feed_dict(self, one_batch, is_train, mc=False):
        """
        set necessary placeholder
        :one_batch: a batch of data in tuple (X, Q, Y), prepared by self._preprocessing()
        :is_train: 
        """
        if is_train:
            fd = {self.X: one_batch[0],
                  self.Q: one_batch[1],
                  self.Y: one_batch[2],
                  self.input_keep_prob: self.input_keep_prob_value,
                  self.output_keep_prob: self.output_keep_prob_value,
                  self.lr: self.lr_value}
        elif mc:
            fd = {self.X: one_batch[0],
                  self.Q: one_batch[1],
                  self.input_keep_prob: self.input_keep_prob_value,
                  self.output_keep_prob: self.output_keep_prob_value}
        else:
            fd = {self.X: one_batch[0],
                  self.Q: one_batch[1],
                  self.input_keep_prob: 1.0,
                  self.output_keep_prob: 1.0}
        return fd

    def _preprocessing(self, x_train, y_train, validation_data):
        """
        prepare historical data, pack into packages for training and validation data
        :x_train:
        :y_train:
        :validation_data:
        """
        batch_data_train = self._prepare_batches(x_train, y_train)
        if len(batch_data_train) == 0:
            raise ValueError("A batch size of {} is too large for training data (len = {})."
                             .format(self.batch_size, len(x_train)))
        if validation_data is not None:
            # evaluate on validation set
            x_val, y_val = validation_data
            batch_data_val = self._prepare_batches(x_val, y_val)
            if len(batch_data_val) == 0:
                raise ValueError("A batch size of {} is too large for validation data (len = {})."
                                 .format(self.batch_size, len(x_val)))
        else:
            y_val = y_train
            batch_data_val = batch_data_train
        return batch_data_train, batch_data_val, y_val

    def _gen_X_q(self, x):
        X = np.reshape(x[:, : self.T * self.n], [-1, self.n, self.T, self.D])
        q = np.reshape(x[:, self.T * self.n:], [-1, self.T, self.D])
        return X, q

    def _prepare_batches(self, x, y=None, fill_last_batch=False):
        """
        prepare batch data for model input
        :x: <record_num, (n + 1) * T, D>
        :y: <record_num, K>. y is None in prediction.
        :return: a list of batch data. One batch data is (X_batch, q_batch, y_batch).
                 Note that y_batch is None if input y is None.
                 X_batch: <batch_size, n, T, D>;
                 q_batch: <batch_size, T, D>;
                 y_batch: <batch_size, K>
        """
        # generate X and q
        X, q = self._gen_X_q(x)

        # split X, q and y into batches
        batch_size = self.batch_size
        total_length = q.shape[0] // batch_size
        if y is not None:
            batch_data = [(X[batch_size * i:batch_size * (i + 1)],
                           q[batch_size * i:batch_size * (i + 1)],
                           y[batch_size * i:batch_size * (i + 1)]) for i in range(0, total_length)]
            # print(X.shape, q.shape, y.shape)
        else:
            batch_data = [(X[batch_size * i:batch_size * (i + 1)],
                           q[batch_size * i:batch_size * (i + 1)],
                           None) for i in range(0, total_length)]
            # print(X.shape, q.shape)

        # fill last_batch for validation/testing
        if fill_last_batch:
            last_piece_idx = total_length * batch_size
            fill_length = batch_size - (len(q) - last_piece_idx)
            # fill_last_batch is only valid in validation/test mode, in which case, y is None.
            X_last_piece, q_last_piece = X[last_piece_idx:], q[last_piece_idx:]
            X_last_batch = np.pad(X_last_piece, ((0, fill_length), (0, 0), (0, 0), (0, 0)),
                                  mode='constant', constant_values=0)
            q_last_batch = np.pad(q_last_piece, ((0, fill_length), (0, 0), (0, 0)), mode='constant',
                                  constant_values=0)
            batch_data.append((X_last_batch, q_last_batch, None))
        return batch_data

    def _set_config(self, x=None, rs=False, **config):
        """
        read out configurations, used at the beginning of self.fit_eval()
        :config:
        """
        if rs:
            config_names = set(config.keys())
            assert config_names.issuperset(self.saved_configs)
            assert config_names.issuperset(self.lr_decay_configs) or \
                config_names.issuperset(self.lr_configs)
            self.D = config.get('D')  # input's variable dimension (convolution filter width)
            self.K = config.get('K')  # output's variable dimension
        else:
            super()._check_config(**config)
            self.D = x.shape[-1]
            # self.K = future_seq_len is set in constructor
        self.T = config.get("T", 1)
        self.W = config.get('W', 1)
        self.n = config.get('n', 7)
        self.highway_window = config.get('highway_window', 1)
        self.en_conv_hidden_size = config.get('en_conv_hidden_size', 32)
        self.en_rnn_hidden_sizes = config.get('en_rnn_hidden_sizes', [16, 32])
        self.input_keep_prob_value = config.get('input_keep_prob', 0.8)
        self.output_keep_prob_value = config.get('output_keep_prob', 1.0)
        assert (self.highway_window <= self.T), \
            "Invalid configuration value. 'highway_window' must not exceed 'T'"
        assert (self.W <= self.T), "invalid configuration value. 'W' must not exceed 'T'"
        self.past_seq_len = config["past_seq_len"]
        assert self.past_seq_len == (self.n + 1) * self.T, "past_seq_len should equal (n + 1) * T,"\
                                                           "Currently, past_seq_len:{}; n:{}; T:{}"\
                                                           "You can add or adjust values" \
                                                           " of the three configurations."\
            .format(self.past_seq_len, self.n, self.T)

        self.batch_size = config.get('batch_size', 100)
        self.metric = config.get('metric', 'mse')

        self._set_running_configs(**config)

    def _set_running_configs(self, **config):
        self.epochs = config.get('epochs', 10)
        assert self.epochs >= 1, "The number of epochs should not be less than 1."\
                                 "Please check the value of 'epochs' you passed in configs"

        # learning rate decay
        self.decay_epochs = config.get('decay_epochs', 0)
        if self.decay_epochs > 0:
            if self.decay_epochs > self.epochs:
                raise ValueError("the value of decay epochs: {} is larger than epochs: {}. "
                                 "Please adjust your configuration"
                                 .format(self.decay_epochs, self.epochs))
            self.max_lr = config.get('max_lr', 0.003)
            self.min_lr = config.get('min_lr', 0.0001)
            assert self.min_lr < self.max_lr, \
                "min_lr: {} should be smaller than max_lr: {} for learning rate decay. " \
                "Please check your configs.".format(self.min_lr, self.max_lr)
        else:
            self.lr_value = config.get('lr', 0.001)

    def _open_sess(self):
        """
        return session if exists, open a session otherwise
        :return: self.sess
        """
        if self.sess is None:
            self.sess = tf.Session()
        return self.sess

    def close_sess(self):
        """
        [!!!] not actually used in code
        """
        self.sess.close()

    def fit_eval(self, x, y, validation_data=None, mc=False, verbose=0, **config):
        """
        fit for one iteration, a session will be open inside this function
        :param x: 3-d array in format (no. of samples, past sequence length, 2+feature length),
        in the last dimension, the 1st col is the time index (data type needs to be numpy datetime
        type, e.g. "datetime64"),
        the 2nd col is the target value (data type should be numeric)
        :param y: 2-d numpy array in format (no. of samples, future sequence length)
        if future sequence length > 1,
        or 1-d numpy array in format (no. of samples, ) if future sequence length = 1
        :param validation_data: tuple in format (x_test,y_test), data used for validation.
        If this is specified, validation result will be the optimization target for automl.
        Otherwise, train metric will be the optimization target.
        :param config: optimization hyper parameters
        :return: the resulting metric 
        """
        if not self.built:
            # use the same config for different iterations.
            self._set_config(x, **config)
            if verbose > 0:
                print("building")
                build_st = time.time()
            self._build(**config)
            if verbose > 0:
                print("Build model took {}s".format(time.time() - build_st))
            sess = self._open_sess()
            sess.run(tf.global_variables_initializer())

        batch_data_train, batch_data_val, y_val = self._preprocessing(x, y, validation_data)

        # mainly for incremental training, in which case, **config is the new running configs
        self._set_running_configs(**config)

        for i in range(self.epochs):
            if self.decay_epochs > 0:
                self.lr_value = self.min_lr + \
                                (self.max_lr - self.min_lr) * math.exp(-i / self.decay_epochs)
            # print("lr value is", self.lr_value)
            loss = self._run_one_epoch_train(batch_data_train)
            if verbose > 0:
                print("Epoch", i, "loss:", loss)
            if i != 0 and i % 5 == 0 or i == self.epochs - 1:
                val_y_pred = self._run_one_epoch_predict(batch_data_val, mc=mc)
                val_y_true = y_val[:len(val_y_pred)]
                metric_result = Evaluator.evaluate(self.metric, val_y_true, val_y_pred,
                                                   multioutput='uniform_average')
                if verbose > 0:
                    print("Evaluation on epoch {}: {}: {}"
                          .format(i, self.metric, metric_result))
        return metric_result

    def _run_one_epoch_train(self, batch_data):
        loss_list = []
        for ds in batch_data:
            fd = self._get_feed_dict(ds, is_train=True)
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict=fd)
            loss_list.append(loss)
        return np.mean(loss_list)

    def _run_one_epoch_predict(self, batch_data, mc=False):
        sess = self._open_sess()
        y_pred_list = []
        for ds in batch_data:
            fd = self._get_feed_dict(ds, is_train=False, mc=mc)
            pred = sess.run(self.y_pred, feed_dict=fd)
            y_pred_list.append(pred)
        y_pred = np.concatenate(y_pred_list)
        return y_pred

    def predict(self, x, mc=False):
        """
        Prediction on x. Opens a session and restore parameter
        :param x: input
        :return: predicted y (expected dimension = 2)
        """
        batch_data = self._prepare_batches(x, fill_last_batch=True)
        y_pred = self._run_one_epoch_predict(batch_data, mc=mc)
        return y_pred[:len(x)]

    def evaluate(self, x, y, metrics=['mse'], multioutput='raw_values'):
        """
        Evaluate on x, y
        :param x: input
        :param y: target
        :param metrics: a list of metrics in string format
        :return: a list of metric evaluation results
        """
        y_pred = self.predict(x)
        if y_pred.shape[1] == 1:
            multioutput = 'uniform_average'
        # y = np.squeeze(y, axis=2)
        return [Evaluator.evaluate(m, y, y_pred, multioutput=multioutput) for m in metrics]

    def predict_with_uncertainty(self, x, n_iter=100):
        result = np.zeros((n_iter,) + (x.shape[0], self.future_seq_len))

        for i in range(n_iter):
            result[i, :, :] = self.predict(x, mc=True)

        prediction = result.mean(axis=0)
        uncertainty = result.std(axis=0)
        return prediction, uncertainty

    def save(self, model_path, config_path):
        """
        save model to file.
        :param model_path: the model file path to be saved to.
        :param config_path: the config file path to be saved to.
        :return:
        """
        sess = self._open_sess()
        saver = tf.train.Saver()
        saver.save(sess, model_path)

        config_to_save = {"en_conv_hidden_size": self.en_conv_hidden_size,
                          "en_rnn_hidden_sizes": self.en_rnn_hidden_sizes,
                          "highway_window": self.highway_window,
                          "input_keep_prob": self.input_keep_prob_value,
                          "output_keep_prob": self.output_keep_prob_value,
                          "D": self.D,
                          "K": self.K,
                          "T": self.T,
                          "n": self.n,
                          "W": self.W,
                          "past_seq_len": self.past_seq_len,
                          "metric": self.metric,
                          "epochs": self.epochs,
                          "decay_epochs": self.decay_epochs,
                          "batch_size": self.batch_size}
        assert set(config_to_save.keys()) == self.saved_configs,\
            "The keys in config_to_save is not the same as self.saved_configs." \
            "Please keep them consistent"
        if self.decay_epochs > 0:
            lr_decay_configs = {"min_lr": self.min_lr,
                                "max_lr": self.max_lr}
            assert set(lr_decay_configs.keys()) == self.lr_decay_configs,\
                "The keys in lr_decay_configs is not the same as self.lr_decay_configs." \
                "Please keep them consistent"
            config_to_save.update(lr_decay_configs)
        else:
            lr_configs = {"lr": self.lr_value}
            assert set(lr_configs.keys()) == self.lr_configs, \
                "The keys in lr_configs is not the same as self.lr_configs." \
                "Please keep them consistent"
            config_to_save.update(lr_configs)

        save_config(config_path, config_to_save)

    def restore(self, model_path, **config):
        """
        restore model from file
        :param model_path: the model file
        :param config: the trial config
        """
        if not self.built:
            # build model
            self._set_config(rs=True, **config)
            self._build(**config)
        sess = self._open_sess()
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        self.built = 1

    def _get_required_parameters(self):
        return {'past_seq_len'}

    def _get_optional_parameters(self):
        return {}


if __name__ == "__main__":
    from zoo.automl.feature.time_sequence import TimeSequenceFeatureTransformer
    from zoo.automl.common.util import split_input_df

    dataset_path = os.getenv("ANALYTICS_ZOO_HOME") + "/bin/data/NAB/nyc_taxi/nyc_taxi.csv"
    df = pd.read_csv(dataset_path)
    # df = pd.read_csv('automl/data/nyc_taxi.csv')
    future_seq_len = 2
    model = MTNet(check_optional_config=False, future_seq_len=future_seq_len)
    train_df, val_df, test_df = split_input_df(df, val_split_ratio=0.1, test_split_ratio=0.1)
    feature_transformer = TimeSequenceFeatureTransformer(future_seq_len=future_seq_len)
    config = {
        'selected_features': ['IS_WEEKEND(datetime)', 'MONTH(datetime)', 'IS_AWAKE(datetime)',
                              'HOUR(datetime)'],
        'batch_size': 100,
        'epochs': 10,
        "T": 4,
        "n": 3,
        "W": 2,
        'highway_window': 2,
        # 'decay_epochs': 10,
        # 'input_keep_prob': 1.0
        # past_seq_len = (n + 1) * T
    }
    config['past_seq_len'] = (config['n'] + 1) * config['T']
    x_train, y_train = feature_transformer.fit_transform(train_df, **config)
    x_val, y_val = feature_transformer.transform(val_df, is_train=True)
    x_test, y_test = feature_transformer.transform(test_df, is_train=True)
    # y_train = np.c_[y_train, y_train/2]
    # y_test = np.c_[y_test, y_test/2]
    for i in range(2):
        print("fit_eval:", model.fit_eval(x_train, y_train,
                                          validation_data=(x_val, y_val), verbose=1, **config))

    print("evaluate:", model.evaluate(x_test, y_test))
    y_pred = model.predict(x_test)

    dirname = "tmp"
    model_1 = MTNet(check_optional_config=False)
    save(dirname, model=model)
    restore(dirname, model=model_1, config=config)
    predict_after = model_1.predict(x_test)
    assert np.allclose(y_pred, predict_after), \
        "Prediction values are not the same after restore: " \
        "predict before is {}, and predict after is {}".format(y_pred,
                                                               predict_after)
    new_config = {'epochs': 5}
    model_1.fit_eval(x_train, y_train, **new_config)
    print("evaluate:", model_1.evaluate(x_test, y_test))

    import shutil
    shutil.rmtree("tmp")

    from matplotlib import pyplot as plt

    y_test = np.squeeze(y_test)
    y_pred = np.squeeze(y_pred)


    def plot_result(y_test, y_pred):
        # target column of dataframe is "value"
        # past sequence length is 50
        # pred_value = pred_df["value"].values
        # true_value = test_df["value"].values[50:]
        fig, axs = plt.subplots()

        axs.plot(y_pred, color='red', label='predicted values')
        axs.plot(y_test, color='blue', label='actual values')
        axs.set_title('the predicted values and actual values (for the test data)')

        plt.xlabel('test data index')
        plt.ylabel('number of taxi passengers')
        plt.legend(loc='upper left')
        plt.savefig("MTNet_result.png")


    plot_result(y_test, y_pred)
