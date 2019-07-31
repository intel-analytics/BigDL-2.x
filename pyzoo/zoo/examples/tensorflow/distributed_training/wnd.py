# -*- coding: utf-8 -*-
#####################
# create by jinzitian
# date : 2019-03-20
#####################
import tensorflow as tf
import math
import os

# from ..metric.metric import auc
from config_reader import config_reader

class WideAndDeep(object):


    def __init__(self, fields_and_wide_data_dict, wide_and_deep_conf, FLAGS, train_or_test):
        #batch_size = FLAGS.batch_size if(train_or_test == 'train') else FLAGS.test_batch_size
        #wide_and_deep_conf, fields_and_wide_data_dict = self.get_conf_and_hdfs_tfrecords(hdfs_conf_path, hdfs_path, batch_size)
        #print(fields_and_wide_data_dict)
        self.build_wide_and_deep(fields_and_wide_data_dict, wide_and_deep_conf, FLAGS, train_or_test)


    def parse(self, serialized_example, continuous_dim, field_name_list):
        feature_dict = {
                        'label': tf.FixedLenFeature([1], tf.float32),  #[]表示标量,[3]则表示长度为3,[1]表示长度为1,batch_size * 1
                        'wide': tf.FixedLenFeature([1],tf.string),
                        'continuous': tf.FixedLenFeature([continuous_dim], tf.float32)
                       }
        #将field 稀疏特征加入解析
        for field_name in field_name_list:
            feature_dict[field_name] = tf.FixedLenFeature([1],tf.string)

        features = tf.parse_single_example(
                serialized_example,
                features= feature_dict
                )
        return features


    def dataset_train_generator(self, filename, batch_size, continuous_dim, field_name_list):
        #filename can be a list or a single file
        dataset = tf.data.TFRecordDataset(filename)
        #shuffle区是缓冲洗牌区，每次从shuffle区随机取一个样本到batch，再从样本集里补充一个数据到shuffle区，因此将shuffle区设置>为1可以变为顺序取样本
        dataset = dataset.map(lambda x:self.parse(x, continuous_dim, field_name_list)).repeat().shuffle(1000).batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        features_dict = iterator.get_next()
        return features_dict


    def get_conf(self, hdfs_conf_path):

        '''
        return {"field_name_list":["field_1","field_2","field_3"],
                "field_1":20,
                "field_2":20,
                "field_3":20,
                "continuous":10,
                "wide":1000}
        '''
        print(hdfs_conf_path)
        file = tf.gfile.Open(hdfs_conf_path, 'r')
        field_name_list = []
        conf = {}
        for line in file:
            info = line.split(' ')
            conf[info[0]] = int(info[1])
            if 'field' in info[0]:
                field_name_list.append(info[0])
        conf['field_name_list'] = field_name_list
        print(conf)
        return conf


    def get_conf_and_hdfs_tfrecords(self, hdfs_conf_path, hdfs_path, batch_size):
        '''
        需要利用脚本找到active name node 然后再加上hdfs的路径，来得到完整的hdfs路径hdfs_path
        hdfs:// + hadoop-jy-namenode02:8020 + /user/uaa/test/jinzitian/tf_records/mnist-dataset/
        '''
        file_name_list = tf.gfile.ListDirectory(hdfs_path)
        hdfs_file_list = [ hdfs_path + file for file in file_name_list ]
        #hdfs_file_list = [r"/home/intel/train_data/part-r-00490"]
        conf_info = self.get_conf(hdfs_conf_path)
        continuous_dim = conf_info['continuous']
        field_name_list = conf_info['field_name_list']
        self.field_name_list = field_name_list
        fields_and_wide_data_dict = self.dataset_train_generator(hdfs_file_list, batch_size, continuous_dim, field_name_list)
        return conf_info, fields_and_wide_data_dict


    def deep_model(self, embedding_and_continuous, hidden1_units, hidden2_units, hidden3_units):
        deep_var = []
        #embedding和连续特征拼接后作为输入
        input_len = int(embedding_and_continuous.shape[1])
        with tf.name_scope("hidden1"):
            weights = tf.get_variable(name="weights1", shape=[input_len, hidden1_units], initializer=tf.random_normal_initializer(stddev=math.sqrt(2.0 / (float(input_len)))))
            biases = tf.get_variable(name="biases1", shape=[hidden1_units], initializer = tf.zeros_initializer())
            hidden1 = tf.nn.relu(tf.matmul(embedding_and_continuous, weights)) + biases
            deep_var.append(weights)
            deep_var.append(biases)
        with tf.name_scope("hidden2"):
            weights = tf.get_variable(name="weights2", shape=[hidden1_units, hidden2_units], initializer=tf.random_normal_initializer(stddev=math.sqrt(2.0 / (float(hidden1_units)))))
            biases = tf.get_variable(name="biases2", shape=[hidden2_units], initializer = tf.zeros_initializer())
            hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
            deep_var.append(weights)
            deep_var.append(biases)
        with tf.name_scope("hidden3"):
            weights = tf.get_variable(name="weights3", shape=[hidden2_units, hidden3_units], initializer=tf.random_normal_initializer(stddev=math.sqrt(2.0 / (float(hidden2_units)))))
            biases = tf.get_variable(name="biases3", shape=[hidden3_units], initializer = tf.zeros_initializer())
            hidden3 = tf.nn.relu(tf.matmul(hidden2, weights) + biases)
            deep_var.append(weights)
            deep_var.append(biases)
        with tf.name_scope("deep_output"):
            weights = tf.get_variable(name="weights4", shape=[hidden3_units, 1], initializer=tf.random_normal_initializer(stddev=math.sqrt(2.0 / (float(hidden3_units)))))
            biases = tf.get_variable(name="biases4", shape=[1], initializer = tf.zeros_initializer())
            output = tf.nn.relu(tf.matmul(hidden3, weights) + biases)
            deep_var.append(weights)
            deep_var.append(biases)
        return output, deep_var


    def deep_model_new(self, embedding_and_continuous, hidden_dim_list):
        deep_var = []
        #embedding和连续特征拼接后作为输入
        input_len = int(embedding_and_continuous.shape[1])
        all_dim_list = [input_len] + hidden_dim_list + [1]
        input_matrix = embedding_and_continuous
        for i in range(1, len(all_dim_list)):
            hidden_name = "hidden" + str(i)
            weight_name = "weights" + str(i)
            biases_name = "biases" + str(i)

            with tf.name_scope(hidden_name):
                                #均匀分布tf.random_uniform_initializer(-0.001, 0.001)，正太分布tf.random_normal_initializer(stddev=0.01)
                weights = tf.get_variable(name=weight_name, shape=[all_dim_list[i-1], all_dim_list[i]], initializer=tf.random_normal_initializer(stddev=0.001))
                biases = tf.get_variable(name=biases_name, shape=[all_dim_list[i]], initializer = tf.zeros_initializer())
                input_matrix = tf.nn.relu(tf.matmul(input_matrix, weights)) + biases
                deep_var.append(weights)
                deep_var.append(biases)
        return input_matrix, deep_var


    def wide_model(self, wide_dim, input_data_sparse):
        wide_var = []
        #离散和交叉特征作为输入
        with tf.name_scope("wide_output"):
            weights = tf.get_variable("weights", shape = [wide_dim, 1], initializer=tf.random_normal_initializer(stddev=0.001))
            biases = tf.get_variable(name="biases", shape=[1], initializer = tf.zeros_initializer())
            #这里变相用sparse embedding lookup的方式实现矩阵乘法
            output = tf.nn.embedding_lookup_sparse(weights, input_data_sparse, None, combiner="sum") + biases
            wide_var.append(weights)
            wide_var.append(biases)
        #self.wide_weights = weights
        return output, wide_var


    def desne_str_to_int_sparse(self, str_dense_tensor, spl):
        #shape [batch,1]
        flat_tensor = tf.reshape(str_dense_tensor,(-1,))
        str_sparse_tensor = tf.string_split(flat_tensor, spl)
        int_sparse_tensor = tf.SparseTensor(str_sparse_tensor.indices, tf.string_to_number(str_sparse_tensor.values, tf.int64), str_sparse_tensor.dense_shape)
        return int_sparse_tensor

    def dense_to_sparse(self, dense_tensor):
        idx = tf.where(tf.not_equal(dense_tensor, -1))

        sparse_tensor = tf.SparseTensor(idx, tf.gather_nd(dense_tensor, idx), tf.shape(dense_tensor, out_type=tf.int64))
        #print(sparse_tensor)
        return sparse_tensor

    def build_wide_and_deep(self, fields_and_wide_dict, wide_and_deep_conf, FLAGS, train_or_test):
        """
        fields_and_wide_dict中存储从tf.records中获取数据
            1、deep侧特征：fields的sparse特征，连续特征；
            2、wide侧特征：离散和交叉特征的sparse数据；
            3、label数据；
        wide_and_deep_conf存储了field维度等信息
        FLAGS存储了各种超参数信息
        """
        y = fields_and_wide_dict['label']
        wide_dense = fields_and_wide_dict['wide']#from tf.train.shuffle_batch
        continuous_feature = fields_and_wide_dict['continuous']
        wide_dim = wide_and_deep_conf['wide']
        field_names_list = wide_and_deep_conf['field_name_list']
        self.label = y
        self.wide_sparse_feature = wide_dense
        self.continuous_feature = continuous_feature
        self.field_sparse_feature_dict = {}

        embedding_deep_var_list = []
        #将离散特征分field进行embedding并与连续特征拼接
        field_embedding_dict = {}
        with tf.name_scope("field_embedding"):
            for field_name in field_names_list:
                field_dim = wide_and_deep_conf[field_name]
                field_embeddings = tf.get_variable(name=field_name, shape=[field_dim, FLAGS.embedding_dim], initializer=tf.random_normal_initializer(stddev=0.001))
                field_dense_str = fields_and_wide_dict[field_name]#from tf.train.shuffle_batch
                self.field_sparse_feature_dict[field_name] = field_dense_str
                field_embedding = tf.nn.embedding_lookup_sparse(field_embeddings, self.dense_to_sparse(field_dense_str), None, combiner="sum")
                '''
                可以在生成tfrecord的时候生成一个对应大小的[1,1,1,1,1]的sparse的weight数据，帮助完成mean操作，sum则不需要
                field_embedding = tf.nn.embedding_lookup_sparse(field_embeddings, field_sparse_feature, fields_and_wide_dict[field_name+"_weights"], combiner="mean")
                '''
                field_embedding_dict[field_name] = field_embedding
                embedding_deep_var_list.append(field_embeddings)

        #shape = batch * (embedding_dim * field_num + continuous_feature_num)
        embedding_and_continuous = tf.concat([field_embedding_dict[i] for i in field_names_list] + [continuous_feature], axis = 1)

        deep_output, deep_var_list = self.deep_model_new(embedding_and_continuous, config_reader(FLAGS.hidden_struct)['hidden_dim_list'])
        #deep_output, deep_var_list = self.deep_model(embedding_and_continuous, FLAGS.hidden1_dim, FLAGS.hidden2_dim, FLAGS.hidden3_dim)
        wide_output, wide_var_list = self.wide_model(wide_dim, self.dense_to_sparse(wide_dense))

        # 使用 LR 将两个模型组合在一起
        deep_weight = tf.get_variable(name="deep_weight", shape=[1, 1], initializer=tf.random_normal_initializer(stddev=0.001))
        bias = tf.get_variable(name="wide_and_deep_bias", shape=[1], initializer = tf.zeros_initializer())
        deep_var_list.append(deep_weight)
        deep_var_list.extend(embedding_deep_var_list)
        wide_var_list.append(bias)

        prediction = tf.matmul(deep_output, deep_weight) + wide_output + bias

        #shape = batch * 1
        prob = tf.nn.sigmoid(prediction)
        self.score = prob

        #交叉熵损失
        #loss = tf.reduce_mean(tf.reduce_sum(y * -tf.log(prob) + (1-y) * -tf.log(1-prob), 1))
        loss = tf.reduce_mean(tf.reduce_sum(y * -tf.log(prob+1e-12) + (1-y) * -tf.log(1-prob+1e-12), 1))
        #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=prediction))

        '''
        手动计算l1损失
        regloss = 0
        for name in wide_var_name_list:
            w_vec = tf.get_variable(name)
            regloss += tf.contrib.layers.l1_regularizer(0.001)(w_vec)
        '''
        self.loss = loss

        if train_or_test == 'train':

            '''
            self.ftrl_lr_rate = tf.placeholder(tf.float32,[])
            self.ada_lr_rate = tf.placeholder(tf.float32,[])
            wide_optimizer = tf.train.FtrlOptimizer(self.ftrl_lr_rate, l1_regularization_strength = FLAGS.FTRL_l1_rate)
            deep_optimizer = tf.train.AdagradOptimizer(self.ada_lr_rate, 1e-7)
            wide_grad_and_vars = wide_optimizer.compute_gradients(loss, var_list=wide_var_list)
            deep_grad_and_vars = deep_optimizer.compute_gradients(loss, var_list=deep_var_list)
            wide_opt = wide_optimizer.apply_gradients(wide_grad_and_vars)
            deep_opt = deep_optimizer.apply_gradients(deep_grad_and_vars)

            self.wide_opt = wide_opt
            self.deep_opt = deep_opt
            '''

            '''
            self.ftrl_lr_rate = tf.placeholder(tf.float32,[])
            self.adam_lr_rate = tf.placeholder(tf.float32,[])
            wide_optimizer = tf.train.FtrlOptimizer(self.ftrl_lr_rate, l1_regularization_strength = FLAGS.FTRL_l1_rate)
            deep_optimizer = tf.train.AdamOptimizer(self.adam_lr_rate)
            wide_grad_and_vars = wide_optimizer.compute_gradients(loss, var_list=wide_var_list)
            deep_grad_and_vars = deep_optimizer.compute_gradients(loss, var_list=deep_var_list)
            wide_opt = wide_optimizer.apply_gradients(wide_grad_and_vars)
            deep_opt = deep_optimizer.apply_gradients(deep_grad_and_vars)
            self.wide_opt = wide_opt
            self.deep_opt = deep_opt
            '''

            self.adam_lr_rate = tf.placeholder(tf.float32,[])
            optimizer = tf.train.AdamOptimizer(self.adam_lr_rate)
            self.train_op = optimizer.minimize(loss)





if __name__ == '__main__':


    pass


