from zoo.automl.model.abstract import BaseModel
import numpy as np
import pandas as pd 
from zoo.automl.common.metrics import Evaluator
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress some warnings
import tensorflow as tf
from zoo.automl.common.util import *

class MTNet(BaseModel):
    ''' source code from https://github.com/Maple728/MTNet'''
    def __init__(self,  check_optional_config=True, future_seq_len=1):
        '''
        Constructer, setting up necessary parameters
        '''
        # [[[WIP]]] currently  future_seq_len is implemented only for 1
        #if future_seq_len!=1:
        #    raise ValueError("Invalid future_seq_len. Only future_seq_len=1 is implemented currently")
        self.K = future_seq_len  # this configuration is previously used as output dimension in source code, curretnly change to be output's future_seq_length
        # config parameter
        self.T = None # timestep
        self.W = None # convolution window size (convolution filter height)` ?
        self.n = None # the number of the long-term memory series
        self.highway_window = None  # the window size of ar model

        self.D = None # input's variable dimension (convolution filter width)

        self.en_conv_hidden_size = None
        self.en_rnn_hidden_sizes = None  # last size is equal to en_conv_hidden_size, should be a list
        self.input_keep_prob_value = None
        self.output_keep_prob_value = None
        self.lr_value = None

        self.batch_size = None
        self.metric = None
        self.built = 0 # indicating if a new model needs to be build
        self.sess = None # the session

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

        # self.reset_statistics_vars = None
        # self.merged_summary = tf.summary.merge_all()
        # self.summary_updates = tf.get_collection('summary_ops')

    def __encoder(self, origin_input_x, n, strides = [1, 1, 1, 1], padding = 'VALID', activation_func = tf.nn.relu, scope = 'Encoder'):
        '''
            Treat batch_size dimension and n dimension as one batch_size dimension (batch_size * n).
        :param input_x:  <batch_size, n, T, D>
        :param strides: convolution stride
        :param padding: convolution padding
        :param scope: encoder scope
        :return: the embedded of the input_x <batch_size, n, last_rnn_hid_size>
        '''
        # constant
        scope = 'Encoder_' + scope
        batch_size_new = self.batch_size * n
        Tc = self.T - self.W + 1
        last_rnn_hidden_size = self.en_rnn_hidden_sizes[-1]

        # reshape input_x : <batch_size * n, T, D, 1>
        input_x = tf.reshape(origin_input_x, shape = [-1, self.T, self.D, 1])

        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
            # cnn parameters
            with tf.variable_scope('CNN', reuse = tf.AUTO_REUSE):
                w_conv1 = tf.get_variable('w_conv1', shape = [self.W, self.D, 1, self.en_conv_hidden_size], dtype = tf.float32, initializer = tf.truncated_normal_initializer(stddev = 0.1))
                b_conv1 = tf.get_variable('b_conv1', shape = [self.en_conv_hidden_size], dtype = tf.float32, initializer = tf.constant_initializer(0.1))

                # <batch_size_new, Tc, 1, en_conv_hidden_size>
                h_conv1 = activation_func(tf.nn.conv2d(input_x, w_conv1, strides, padding = padding) + b_conv1)
                if self.input_keep_prob_value < 1:
                    h_conv1 = tf.nn.dropout(h_conv1, self.input_keep_prob_value)


            # tmporal attention layer and gru layer
            # rnns
            rnns = [tf.nn.rnn_cell.GRUCell(h_size, activation = activation_func) for h_size in self.en_rnn_hidden_sizes]
            # dropout
            if self.input_keep_prob_value < 1 or self.output_keep_prob_value < 1:
                rnns = [tf.nn.rnn_cell.DropoutWrapper(rnn,
                                                      input_keep_prob = self.input_keep_prob_value,
                                                      output_keep_prob = self.output_keep_prob_value)
                        for rnn in rnns]

            if len(rnns) > 1:
                rnns = tf.nn.rnn_cell.MultiRNNCell(rnns)
            else:
                rnns = rnns[0]

            # attention layer

            # attention weights
            attr_v = tf.get_variable('attr_v', shape = [Tc, 1], dtype = tf.float32, initializer = tf.truncated_normal_initializer(stddev = 0.1))
            attr_w = tf.get_variable('attr_w', shape = [last_rnn_hidden_size, Tc], dtype = tf.float32, initializer = tf.truncated_normal_initializer(stddev = 0.1))
            attr_u = tf.get_variable('attr_u', shape = [Tc, Tc], dtype = tf.float32, initializer = tf.truncated_normal_initializer(stddev = 0.1))

            # rnn inputs
            # <batch_size, n, Tc, en_conv_hidden_size>
            rnn_input = tf.reshape(h_conv1, shape=[-1, n, Tc, self.en_conv_hidden_size])

            # n * <batch_size, last_rnns_size>
            res_hstates = tf.TensorArray(tf.float32, n)
            for k in range(n):
                # <batch_size, en_conv_hidden_size, Tc>
                attr_input = tf.transpose(rnn_input[:, k], perm = [0, 2, 1])

                # <batch_size, last_rnn_hidden_size>
                s_state = rnns.zero_state(self.batch_size, tf.float32)
                if len(self.en_rnn_hidden_sizes) > 1:
                    h_state = s_state[-1]
                else:
                    h_state = s_state

                for t in range(Tc):
                    # attr_v = tf.Variable(tf.truncated_normal(shape=[Tc, 1], stddev=0.1, dtype=tf.float32), name='attr_v')
                    # attr_w = tf.Variable(tf.truncated_normal(shape=[last_rnn_hidden_size, Tc], stddev=0.1, dtype=tf.float32), name='attr_w')
                    # attr_u = tf.Variable(tf.truncated_normal(shape=[Tc, Tc], stddev=0.1, dtype=tf.float32), name='attr_u')

                    # h(t-1) dot attr_w
                    h_part = tf.matmul(h_state, attr_w)

                    # en_conv_hidden_size * <batch_size_new, 1>
                    e_ks = tf.TensorArray(tf.float32, self.en_conv_hidden_size)
                    _, output = tf.while_loop(lambda i, _ : tf.less(i, self.en_conv_hidden_size),
                                              lambda i, output_ta : (i + 1, output_ta.write(i, tf.matmul(tf.tanh( h_part + tf.matmul(attr_input[:, i], attr_u) ), attr_v))),
                                              [0, e_ks])
                                              # : tf.while_loop(cond, body, loop_vars ) :
                                              # : TENSOR.write(idx, content) : write content at index
                    # <batch_size, en_conv_hidden_size, 1>
                    e_ks = tf.transpose(output.stack(), perm = [1, 0, 2])
                    e_ks = tf.reshape(e_ks, shape = [-1, self.en_conv_hidden_size])

                    # <batch_size, en_conv_hidden_size>
                    a_ks = tf.nn.softmax(e_ks)

                    x_t = tf.matmul( tf.expand_dims(attr_input[:, :, t], -2), tf.matrix_diag(a_ks))
                    # <batch_size, en_conv_hidden_size>
                    x_t = tf.reshape(x_t, shape = [-1, self.en_conv_hidden_size])

                    h_state, s_state = rnns(x_t, s_state)

                res_hstates = res_hstates.write(k, h_state)

        return tf.transpose(res_hstates.stack(), perm = [1, 0, 2])

    def _build(self, scope='MTNet', **config ):
        ''' 
        build the model 
        set up configuration parameter and graph components
        :config:
        '''
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.logging.set_verbosity( tf.compat.v1.logging.ERROR) # suppress warnings during building  SO ANNOYING!
        with tf.variable_scope(scope, reuse = False):
            # placeholders
            X = tf.placeholder(tf.float32, shape = [None, self.n, self.T, self.D])
            Q = tf.placeholder(tf.float32, shape = [None, self.T, self.D])
            Y = tf.placeholder(tf.float32, shape = [None, self.K])
            lr = tf.placeholder(tf.float32)
            input_keep_prob = tf.placeholder(tf.float32)
            output_keep_prob = tf.placeholder(tf.float32)

            # ------- no-linear component----------------
            last_rnn_hid_size = self.en_rnn_hidden_sizes[-1]
            # <batch_size, n, en_rnn_hidden_sizes>u
            m_is = self.__encoder(X, self.n, scope = 'm')
            c_is = self.__encoder(X, self.n, scope = 'c')
            # <batch_size, 1, en_rnn_hidden_sizes>
            u = self.__encoder(tf.reshape(Q, shape = [-1, 1, self.T, self.D]), 1, scope = 'in')

            p_is = tf.matmul(m_is, tf.transpose(u, perm = [0, 2, 1]))

            # using softmax
            p_is = tf.squeeze(p_is, axis = [-1])
            p_is = tf.nn.softmax(p_is)
            # <batch_size, n, 1>
            p_is = tf.expand_dims(p_is, -1)

            # using sigmoid
            # p_is = tf.nn.sigmoid(p_is)

            # for summary
            #p_is_mean, _ = tf.metrics.mean_tensor(p_is, updates_collections = 'summary_ops', name = 'p_is')
            #tf.summary.histogram('p_is', p_is_mean)

            # <batch_size, n, en_rnn_hidden_sizes> = <batch_size, n, en_rnn_hidden_sizes> * <batch_size, n, 1>
            o_is = tf.multiply(c_is, p_is)

            # :: last_rnn_hid_size * (self.n + 1) :: is the concatination of (12)
            # following varaibles are a single layer vanilla net
            pred_w = tf.get_variable('pred_w', shape = [last_rnn_hid_size * (self.n + 1), self.K],
                                     dtype = tf.float32, initializer = tf.truncated_normal_initializer(stddev = 0.1))
            pred_b = tf.get_variable('pred_b', shape = [self.K],
                                     dtype = tf.float32, initializer = tf.constant_initializer(0.1))
            
            pred_x = tf.concat([o_is, u], axis = 1)
            pred_x = tf.reshape(pred_x, shape = [-1, last_rnn_hid_size * (self.n + 1)])

            # <batch_size, D>
            y_pred = tf.matmul(pred_x, pred_w) + pred_b

            # ------------ ar component ------------
            with tf.variable_scope('AutoRegression'):
                if self.highway_window > 0: # highway_window is basically (s^ar -1) 
                    highway_ws = tf.get_variable('highway_ws', shape = [self.highway_window * self.D, self.K],
                                                dtype = tf.float32,
                                                initializer = tf.truncated_normal_initializer(stddev = 0.1))
                    highway_b = tf.get_variable('highway_b', shape = [self.K], dtype = tf.float32,
                                                initializer = tf.constant_initializer(0.1))

                    highway_x = tf.reshape(Q[:, -self.highway_window:], shape = [-1, self.highway_window * self.D])
                    y_pred_l = tf.matmul(highway_x, highway_ws) + highway_b

                    # y_pred_l = tf.matmul(Q[:, -1], highway_ws[0]) + highway_b
                    # _, y_pred_l = tf.while_loop(lambda i, _ : tf.less(i, self.highway_window),
                    #                             lambda i, acc : (i + 1, tf.matmul(Q[:, self.T - i - 1], highway_ws[i]) + acc),
                    #                             loop_vars = [1, y_pred_l])
                    y_pred += y_pred_l


        # metrics summary
        #mae_loss, _ = tf.metrics.mean_absolute_error(Y, y_pred, updates_collections = 'summary_ops', name = 'mae_metric')
        #tf.summary.scalar('mae_loss', mae_loss)

        # rmse_loss, _ = tf.metrics.root_mean_squared_error(Y, y_pred, updates_collections = 'summary_ops', name = 'rmse_metric')
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

    def _get_feed_dict(self, one_batch, is_train):
        '''
        set necessary placeholder
        :one_batch: a batch of data in tuple (X, Q, Y), prepared by self._preprocessing()
        :is_train: 
        '''
        if is_train:
            fd = {self.X : one_batch[0],
                  self.Q : one_batch[1],
                  self.Y : one_batch[2],
                  self.input_keep_prob : self.input_keep_prob_value,
                  self.output_keep_prob : self.output_keep_prob_value,
                  self.lr : self.lr_value}
        else:
            fd = {self.X : one_batch[0],
                  self.Q : one_batch[1],
                  self.input_keep_prob : 1.0,
                  self.output_keep_prob :1.0 }
        return fd

    def _preprocessing(self, x_train, y_train, validation_data):
        '''
        prepare historical data, pack into packages for training and validation data
        :x_train:
        :y_train:
        :validation_data:
        :return: (batches_data_train, batches_data_val), y_train
        '''
        # set data length
        self.D = x_train.shape[-1]
        # self.K = y_train.shape[-1]
        # preprocess training
        batch_data_train = self._prepare_batches(x_train, y_train, fill_with_samples=True)
        # for training, fill y_train with additional samples
        batch_size = self.batch_size
        total_length = x_train.shape[0] // batch_size
        y_train = np.r_[ y_train[:total_length*batch_size ], batch_data_train[-1][-1] ]
        # preprocess validation
        if validation_data is not None:
            x_val, y_val = validation_data
            batch_data_val = self._prepare_batches(x_val, y_val)
            return (batch_data_train, batch_data_val), y_train
        else:
            return (batch_data_train, ), y_train

    def _prepare_batches(self, x, y=None, fill_with_samples=False):
        '''
        prepare historical data for train/validation/testing
        :x:
        :y: if y is None
        :return: [ (X_batch[0], q_batch[0], y_batch[0]),  (X_batch[1], q_batch[1], y_batch[1]), ... ] for y not None
                         [ (X_batch[0], q_batch[0], None]),  (X_batch[1], q_batch[1],None), ... ] for y is None
        '''
        unroll_length = self.T
        history_length = self.n
        # generate q and X
        q = x[:, unroll_length*history_length:]
        X = x[:, :unroll_length*history_length]
        X = X.reshape(-1, history_length, unroll_length, self.D)
        # pack into batches
        batch_size = self.batch_size
        total_length = q.shape[0] // batch_size
        if y is not None:
            batch_data = [ (X[batch_size*i:batch_size*(i+1)], q[batch_size*i:batch_size*(i+1)], y[batch_size*i:batch_size*(i+1)]) for i in range(0, total_length) ]
        else:
            batch_data = [ (X[batch_size*i:batch_size*(i+1)], q[batch_size*i:batch_size*(i+1)], None) for i in range(0, total_length) ]
        
        # fill last_batch according to input parameter
        last_piece_idx = total_length*batch_size
        fill_length = batch_size - (len(q) - last_piece_idx)
        X_last_piece, q_last_piece = X[last_piece_idx:], q[last_piece_idx:] 
        if fill_with_samples: # when in training, fill with samples 
            y_last_batch = y[last_piece_idx:]
            samples_idx = np.random.permutation(q.shape[0])[:fill_length]
            X_last_batch = np.r_[X_last_piece, X[samples_idx] ]
            q_last_batch = np.r_[q_last_piece, q[samples_idx] ]
            y_last_batch = np.r_[y_last_batch, y[samples_idx] ]
            batch_data.append((X_last_batch, q_last_batch, y_last_batch))
        else: # when not training, fill with zero and get afterwards, don't care about y
            X_last_batch = np.pad(X_last_piece, ( (0,fill_length),(0,0),(0,0),(0,0) ), mode='constant', constant_values=0)
            q_last_batch = np.pad(q_last_piece, ( (0,fill_length),(0,0),(0,0) ), mode='constant', constant_values=0)
            batch_data.append((X_last_batch, q_last_batch, None))
        return batch_data

    def _set_config(self, store=False, **config):
        ''' 
        read out configurations, used at the beginning of self.fit_eval()
        :config:
        '''
        super()._check_config(**config)
        self.past_seq_len = config.get("past_seq_len", 2)
        assert(self.past_seq_len >= 2), "Invalid configuration value. `past_seq_len` must be greater that 2 in order to have historical data."
        self.W = config.get('W', 1)
        # check consistency if  past_seq_len==(n+1)*T
        self.n = config.get('n', None)
        self.T = config.get('T', None)
        if self.n is None and self.T is None: # if both are not specified, set up default value
            self.n = 1 
        if self.n is not None: # for n is specified or both are not specified 
            if self.past_seq_len % (self.n+1) == 0:
                T = self.past_seq_len // (self.n+1) 
                if self.T is not None: # T is also specifed, check consistency
                    if self.T != T:
                        raise ValueError("input `past_seq_len`, `T`, `n` does not satisfy past_seq_len=T*(n+1). Please check the consistency of configuration.")
                else:
                    self.T = T
            else:
                raise ValueError("input `past_seq_len` is not divisable by `n+1`, this will cause error in MTNet. Please change check the configuration.")
        else: # only T is specified
            if self.past_seq_len % self.T == 0:
                self.n = self.past_seq_len // self.T - 1
        self.highway_window = config.get('highway_window', 1)
        self.en_conv_hidden_size = config.get('en_conv_hidden_size', 16)
        self.en_rnn_hidden_sizes = config.get('en_rnn_hidden_sizes', [16, 16])
        self.input_keep_prob_value = config.get('input_keep_prob', 0.8)
        self.output_keep_prob_value = config.get('output_keep_prob', 1.0)
        assert(self.highway_window <= self.T), "Invalid configuration value. `highway_window` must not exceed `T` "
        assert(self.W <= self.T), "Invalid configuration value. `W` must not exceed `T` "
        if store: # when storing
            self.D = config.get('D')  # input's variable dimension (convolution filter width)
            self.K = config.get('K') # output's variable dimension

        self.lr_value = config.get('lr', 0.001)
        self.batch_size = config.get('batch_size', 100)

        self.metric = config.get('metric', 'mean_squared_error')

    def _open_sess(self):
        ''' 
        return session if exists, open a session otherwise
        :return: self.sess
        '''
        if self.sess is None:
            self.sess = tf.Session()
        return self.sess

    def close_sess(self):
        '''
        [!!!] not actually used in code
        '''
        self.sess.close()

    def fit_eval(self, x, y, validation_data=None, verbose=0, **config):
        ''' 
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
        '''
        self._set_config(**config)
        if verbose > 0:
            print("preprocessing")
        batches_data, y_train = self._preprocessing(x, y, validation_data)
        if not self.train_op:
            if verbose > 0:
                print("building")
            self._build(**config)
            sess = self._open_sess()
            sess.run(tf.global_variables_initializer())
        else:
            sess = self._open_sess()
        epochs = config.get('epochs', 10)
        for i in range(epochs):
            if verbose > 0:
                print("start epoch {}".format(i) )
            # sess.run(model.reset_statistics_vars) # reset statistics variables
            for ds in batches_data[0]:
                # print("\tstart new batch")
                fd = self._get_feed_dict(ds, True)
                _, loss, pred = self.sess.run([self.train_op, self.loss, self.y_pred], feed_dict = fd)
        if validation_data is None:
            # evaluate on training set
            y_pred = self._predict_batches(batches_data[0], sess)
            metric_result = Evaluator.evaluate(self.metric, y_train, y_pred) # expected to be an array
        else:
            # evaluate on validation set
            y_pred = self._predict_batches(batches_data[1], sess)
            y_val = validation_data[1]
            y_pred = y_pred[:len(y_val)] # predicted validation data is padded actual validation data 
            metric_result = Evaluator.evaluate(self.metric, y_val, y_pred) # expected to be an array
        return np.mean(metric_result)

    def _predict_batches(self, batch_data, sess):
        '''
        prepare data on a batch_data returned by self._preprocessing/self._prepare_batches()
        '''
        sess = self._open_sess()
        y_pred_list = []
        for ds in batch_data:
            fd = self._get_feed_dict(ds, False)
            (y_pred, ) = sess.run([self.y_pred], feed_dict = fd)
            y_pred_list.append(y_pred)
        result = np.concatenate(y_pred_list)
        return result

    def predict(self, x):
        """
        Prediction on x. Opens a session and restore parameter
        :param x: input
        :return: predicted y (expected dimension = 2)
        """
        batch_data = self._prepare_batches(x)
        y_pred = self._predict_batches(batch_data, self._open_sess())
        return  y_pred[:len(x)]

    def evaluate(self, x, y, metric=['mean_squared_error']):
        """
        Evaluate on x, y
        :param x: input
        :param y: target
        :param metric: a list of metrics in string format
        :return: a list of metric evaluation results
        """
        y_pred = self.predict(x)
        # y = np.squeeze(y, axis=2)
        # return [Evaluator.evaluate(m, y, y_pred) for m in metric]
        return Evaluator.evaluate(self.metric, y, y_pred)

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
                          "input_keep_prob_value": self.input_keep_prob_value,
                          "output_keep_prob_value": self.output_keep_prob_value,
                          "D": self.D,
                          "K": self.K,
                          "T": self.T,
                          "n": self.n, 
                          "W": self.W,
                          "metric": self.metric,
                          "batch_size": self.batch_size}
        save_config(config_path, config_to_save)

    def restore(self, model_path, **config):
        """
        restore model from file
        :param model_path: the model file
        :param config: the trial config
        """
        print(config)
        if not self.built:
            # build model
            self._set_config(True, **config)
            self._build(**config)
        sess = self._open_sess()
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
                
    def _get_required_parameters(self):
        return {}
    def _get_optional_parameters(self):
        return {}

if __name__=="__main__":
    from zoo.automl.feature.time_sequence import TimeSequenceFeatureTransformer
    from zoo.automl.common.util import split_input_df
    df = pd.read_csv('automl/data/nyc_taxi.csv')
    future_seq_len = 2
    model = MTNet(check_optional_config=False, future_seq_len=future_seq_len)
    train_df, val_df, test_df = split_input_df(df, val_split_ratio=0.1, test_split_ratio=0.1)
    feature_transformer = TimeSequenceFeatureTransformer( future_seq_len=future_seq_len)
    config = {
        # 'input_shape_x': x_train.shape[1],
        # 'input_shape_y': x_train.shape[-1],
        'selected_features': ['IS_WEEKEND(datetime)', 'MONTH(datetime)', 'IS_AWAKE(datetime)',
                              'HOUR(datetime)'],
        'batch_size': 100,
        'epochs': 2,
        "past_seq_len": 6,
        "T": 2,
        #"n":2,
        "highway_window": 2
    }
    x_train, y_train = feature_transformer.fit_transform(train_df, **config)
    x_test, y_test = feature_transformer.transform(test_df, is_train=True)
    print("fit_eval:", model.fit_eval(x_train, y_train, validation_data=(x_test, y_test), **config) )
    #print("fit_eval:", model.fit_eval(x_train, y_train, None, **config) )
    print("evaluate:", model.evaluate(x_test, y_test))
    y_pred = model.predict(x_test)
    print(y_pred.shape)

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