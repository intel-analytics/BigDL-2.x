# -*- coding: utf-8 -*-
#####################
# create by jinzitian
# date : 2019-03-20
#####################
import argparse
import sys
import six

import numpy as np
import tensorflow as tf
from zoo.common import set_core_number
from zoo.common.nncontext import init_nncontext, init_spark_conf
from zoo.pipeline.api.net import TFDataset, TFOptimizer
from zoo.pipeline.api.keras.metrics import AUC
# from zoo.pipeline.api.keras.optimizers import EpochStep
from bigdl.optim.optimizer import MaxEpoch
# from deep_model.model.wnd import WideAndDeep
from zoo.examples.tensorflow.distributed_training.wnd import WideAndDeep
# from deep_model.util.util import Unbuffered
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#保证即时打印
# sys.stdout = Unbuffered(sys.stdout)
# sys.stderr = Unbuffered(sys.stderr)


# import tensorflow as tf
# import numpy as np
#
# example = tf.SparseTensor(indices=[[0], [1], [2]], values=[3, 6, 9], dense_shape=[3])
# vocabulary_size = 10
# embedding_size = 3
# # var = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0])
# var = np.random.random_sample((vocabulary_size, embedding_size))
# embeddings = tf.Variable(var)
#
# embed = tf.nn.embedding_lookup_sparse(embeddings, example, None)
# prob = tf.nn.sigmoid(embed)
# loss = tf.reduce_sum(-tf.log(prob+1e-12))
# optimizer = tf.train.FtrlOptimizer(0.01, l1_regularization_strength = 0.01)
# wide_grad_and_vars = optimizer.compute_gradients(loss, var_list=embeddings)
# grad = tf.unsorted_segment_sum(wide_grad_and_vars[0][0].values, wide_grad_and_vars[0][0].indices,
#                                wide_grad_and_vars[0][0].dense_shape[0])
#
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print(sess.run(wide_grad_and_vars))
#     print(sess.run(grad))

# t = 0

def get_conf(hdfs_conf_path):
        file = tf.gfile.Open(hdfs_conf_path, 'r')
        field_name_list = []
        conf = {}
        for line in file:
            info = line.split(' ')
            conf[info[0]] = int(info[1])
            if 'field' in info[0]:
                field_name_list.append(info[0])
        conf['field_name_list'] = field_name_list
        #print(conf)
	return conf


# TF Tensor eval is very slow
def parse_single_record(example, continuous_dim, field_name_list):
            sess = tf.InteractiveSession()
            name_to_features = {
               "label": tf.FixedLenFeature([1], tf.float32),
               "wide": tf.FixedLenFeature([1], tf.string),
               "continuous": tf.FixedLenFeature([continuous_dim], tf.float32)
            }
            for field_name in field_name_list:
                name_to_features[field_name] = tf.FixedLenFeature([1], tf.string)
            result = tf.parse_single_example(example, name_to_features)
            result_dict = {}
            for key, value in result.items():
                res = value.eval()
                if key not in ["label", "continuous"]:
                        res = res[0]
                        assert isinstance(res, six.string_types)
                        res = [int(i) for i in res.split(",")]
                        res = np.array(res)
                result_dict[key] = res
            sess.close()
            return result_dict


def generate_data():
    feed_dict = {}
    for feature in ["wide"] + field_name_list:
        feed_dict[feature] = np.random.randint(0, conf_info[feature], size=(30,))
    feed_dict["continuous"] = np.random.random(size=(continuous_dim,))
    feed_dict["label"] = np.random.randint(0, 2, size=(1,))
    return feed_dict


def parse_example(example, field_name_list):
        result = tf.train.Example.FromString(example)
        result_dict = {}
        for field in field_name_list + ["wide"]:  # str as bytes
                str_content = result.features.feature[field].bytes_list.value[0]
                ids = [int(i) for i in str_content.split(",")]
                result_dict[field] = np.array(ids)
        for field in ["continuous", "label"]:  # float array
                content = np.float32(result.features.feature[field].float_list.value)
                result_dict[field] = content
        return result_dict


def pad_record(record, length_dict):
        for key, value in length_dict.items():
                before = record[key]
                pad_length = value[0] - len(before)
                if pad_length > 0:
                        after = np.concatenate((before, [-1]*pad_length))
                elif pad_length < 0:
                        after = before[:value[0]]
                else:
                        after = before
                record[key] = after
        return record


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    '''
    output一般用来模型普通save路基或者是tensorboard的数据save路径
    '''
    parser.add_argument('--output_dir', type=str,
                       help='Directory for storing output data')
    parser.add_argument('--model_dir', type=str,
                       help='Directory for storing model')
    parser.add_argument('--model_version', type=str, default='1',
                       help='Model version of export model')
    parser.add_argument('--epoch_num', type=int, default=7,
                       help='epoch_num')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='batch_size')
    parser.add_argument('--test_batch_size', type=int, default=1,
                       help='test_batch_size')
    parser.add_argument('--embedding_dim', type=int, default=110,
                       help='embedding_dim')
    parser.add_argument('--hidden_struct', type=str, default='{"hidden_dim_list":[1024,512,256]}',
                       help='whole struct is [input_dim] + hidden_dim_list + [1]')
    parser.add_argument('--FTRL_learning_rate', type=float, default=0.02,
                       help='FTRL_learning_rate')
    parser.add_argument('--FTRL_l1_rate', type=float, default=0.0,
                       help='FTRL_l1_rate')
    #parser.add_argument('--AdaGrad_learning_rate', type=float, default=0.0001,
    #                   help='AdaGrad_learning_rate')
    parser.add_argument('--Adam_learning_rate', type=float, default=0.0003,
                       help='Adam_learning_rate')
    parser.add_argument('--target_date', type=str,
                       help='target_date')
    FLAGS, unparsed = parser.parse_known_args()

    #动态获取active node节点,生成完整hdfs路径
    nameservice = "hadoop-bdwg-dt4"
    hdfs_conf_path = 'hdfs://' + nameservice + '/user/intel/mba_rec/deep_model/wide_and_deep_normalize/{0}/conf_save_path/part-00000'.format(FLAGS.target_date)
    hdfs_train_path = 'hdfs://' + nameservice + '/user/intel/mba_rec/deep_model/wide_and_deep_normalize/{0}/train_data/'.format(FLAGS.target_date)
    hdfs_test_path = 'hdfs://' + nameservice + '/user/intel/mba_rec/deep_model/wide_and_deep_normalize/{0}/test_data/'.format(FLAGS.target_date)
    FLAGS.output_dir = 'hdfs://' + nameservice + '/user/intel/mba_rec/deep_model/wide_and_deep_normalize/{0}/output_dir/'.format(FLAGS.target_date)
    FLAGS.model_dir = 'hdfs://' + nameservice + '/user/intel/mba_rec/deep_model/wide_and_deep_normalize/{0}/model_dir/'.format(FLAGS.target_date)

    #打印基本信息
    print("nameservice is %s" % nameservice)
    print("model_dir is %s" % FLAGS.model_dir)
    print("output_dir is %s" % FLAGS.output_dir)
    print("target_date is %s" % FLAGS.target_date)
    print("hdfs_conf_path is %s" % hdfs_conf_path)
    print("hdfs_train_path is %s" % hdfs_train_path)
    print("hdfs_test_path is %s" % hdfs_test_path)

    conf = init_spark_conf().setAppName("Wide And Deep").set("spark.memory.fraction", "0.9") #.set("spark.executorEnv.TF_DISABLE_MKL", "1")#.set("spark.memory.fraction", "0.9")
    sc = init_nncontext(conf)
    set_core_number(2)


    # conf_info = get_conf(hdfs_conf_path)
    conf_info = {}
    # conf_info["continuous"] = 131
    # conf_info["wide"] = 1003803
    conf_info["continuous"] = 2
    conf_info["wide"] = 10
    conf_info["field_name_list"] = ['field_2', 'field_5', 'field_8', 'field_7', 'field_1',
                                                'field_4', 'field_6', 'field_9', 'field_0', 'field_3']
    # conf_info["field_8"] = 32
    # conf_info["field_9"] = 47046
    # conf_info["field_6"] = 39210
    # conf_info["field_7"] = 10401
    # conf_info["field_4"] = 526
    # conf_info["field_5"] = 183763
    # conf_info["field_2"] = 2312
    # conf_info["field_3"] = 522856
    # conf_info["field_0"] = 196885
    # conf_info["field_1"] = 781

    conf_info["field_8"] = 3
    conf_info["field_9"] = 4
    conf_info["field_6"] = 3
    conf_info["field_7"] = 1
    conf_info["field_4"] = 5
    conf_info["field_5"] = 3
    conf_info["field_2"] = 2
    conf_info["field_3"] = 5
    conf_info["field_0"] = 2
    conf_info["field_1"] = 7

    continuous_dim = conf_info['continuous']
    field_name_list = conf_info['field_name_list']
    batch_size = FLAGS.batch_size

    #hdfs_train_path = "file:///home/intel/train_data/"
    #hdfs_test_path = "file:///home/intel/test_data/"
    # TODO: replace record_rdd as RDD[String]
    # record_rdd = sc.newAPIHadoopFile(hdfs_train_path, "org.tensorflow.hadoop.io.TFRecordFileInputFormat", keyClass="org.apache.hadoop.io.BytesWritable",valueClass="org.apache.hadoop.io.NullWritable").map(lambda record: bytes(record[0]))
    # print(record_rdd.getNumPartitions())
    # data_rdd = record_rdd.map(lambda example: parse_example(example, field_name_list))
    data = [0 for i in range(2)]
    data_dumy_rdd = sc.parallelize(data)
    data_rdd = data_dumy_rdd.map(lambda x: generate_data())


    # test_record_rdd = sc.newAPIHadoopFile(hdfs_test_path, "org.tensorflow.hadoop.io.TFRecordFileInputFormat", keyClass="org.apache.hadoop.io.BytesWritable",valueClass="org.apache.hadoop.io.NullWritable").map(lambda record: bytes(record[0]))
    # test_rdd = test_record_rdd.map(lambda example: parse_example(example, field_name_list))
    test_data = [0 for i in range(1)]
    test_dumy_rdd = sc.parallelize(test_data)
    test_rdd = data_dumy_rdd.map(lambda x: generate_data())

    features_dict = {}
    features_dict["label"] = (tf.float32, [1])
    features_dict["wide"] = (tf.int64, [None])
    features_dict["continuous"] = (tf.float32, [continuous_dim])
    for field_name in field_name_list:
        features_dict[field_name] = (tf.int64, [None])
    train_dataset = TFDataset.from_rdd(data_rdd,
                                       features=features_dict,
                                       batch_size=batch_size,
                                       val_rdd=test_rdd)
    fields_and_wide_dict = train_dataset.tensors[0]
    #print(fields_and_wide_dict)

    model = WideAndDeep(fields_and_wide_dict, conf_info, FLAGS, "train")
    loss = model.loss
    #from zoo.pipeline.api.keras.optimizers import Adam
    #optim_method = Adam(FLAGS.Adam_learning_rate/0.75, schedule=EpochStep(1, 0.75))
    from bigdl.optim.optimizer import Adam
    optim_method = Adam(FLAGS.Adam_learning_rate)

    from zoo.pipeline.api.keras.optimizers import SparseAdagrad
    # optimizer = TFOptimizer.from_loss(loss, optim_method, val_outputs=[model.score], val_labels=[model.label], val_method=AUC(), session_config=tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=2))
    # optimizer = TFOptimizer.from_loss(loss, optim_method, val_outputs=[model.score], val_labels=[model.label], val_method=AUC())
    optimizer = TFOptimizer.from_loss(loss, optim_method, val_outputs=[model.score], val_labels=[model.label],
                                      val_method=AUC(), sparse_optim_method=SparseAdagrad())
    optimizer.optimize(end_trigger=MaxEpoch(FLAGS.epoch_num))
