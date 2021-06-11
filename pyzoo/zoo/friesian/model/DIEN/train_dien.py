import argparse
import time

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from zoo.orca.data.file import exists, makedirs
from zoo.orca.learn.tf.estimator import Estimator
from zoo.util.tf import load_tf_checkpoint
from utils import *
from zoo.orca import init_orca_context, stop_orca_context

def train(config, train_data, test_data, n_uid, n_mid, n_cat, batch_size, epoch=10, lr=0.001,
          model_type='DNN', use_bf16=False, seed=3, snapshot_dir='snapshot'):

    cpkts_dir = os.path.join(snapshot_dir, 'cpkts/')
    if not exists(cpkts_dir): makedirs(cpkts_dir)
    snapshot_path = cpkts_dir + "cpkt_noshuffle_" + str(seed) + '_' + model_type

    tf.reset_default_graph()
    with tf.Session(config=config) as sess:
        set_random_seed(seed)
        model = build_model(model_type, n_uid, n_mid, n_cat, lr, use_bf16=use_bf16, training=True)
        [inputs, feature_cols] = align_input_features(model)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        estimator = Estimator.from_graph(inputs=inputs, outputs=[model.y_hat],
                                         labels=[model.target_ph], loss=model.loss,
                                         optimizer=model.optim, sess=sess, model_dir=snapshot_dir,
                                         metrics={'loss': model.loss, 'accuracy': model.accuracy})

        estimator.fit(train_data, epochs=epoch, batch_size=batch_size, feature_cols=feature_cols,
                      label_cols=['label'], validation_data=test_data)

        estimator.save_tf_checkpoint(snapshot_path)

def test(config, test_data, batch_size, n_uid, n_mid, n_cat, model_type='DNN', use_bf16=False, seed=3,
         snapshot_dir='snapshot'):
    cpkts_dir = os.path.join(snapshot_dir, 'cpkts/')
    if not exists(cpkts_dir): raise Exception("no checkpoint dir")
    snapshot_path = cpkts_dir + "cpkt_noshuffle_" + str(seed) + '_' + model_type

    tf.reset_default_graph()
    with tf.Session(config=config) as sess:
        set_random_seed(seed)
        model = build_model(model_type, n_uid, n_mid, n_cat, lr=0.001, use_bf16=use_bf16, training=False)
        [inputs, feature_cols] = align_input_features(model)
        load_tf_checkpoint(sess, snapshot_path)

        estimator = Estimator.from_graph(inputs=inputs, outputs=[model.y_hat],
                                         labels=[model.target_ph], loss=model.loss,
                                         optimizer=model.optim, sess=sess,model_dir=snapshot_dir,
                                         metrics={'loss': model.loss, 'accuracy': model.accuracy})

        result = estimator.evaluate(test_data, batch_size, feature_cols=feature_cols, label_cols=['label'])
        print('test result:', result)

        prediction_df = estimator.predict(test_data,feature_cols=feature_cols)
        prediction_df.cache()
        transform_label = udf(lambda x: int(x[0]), IntegerType())
        prediction_df = prediction_df.withColumn('label', transform_label(col('label')))
        prediction_df.printSchema()
        prediction_df.select(['label', 'prediction']).show(10, False)
        evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction",
                                                  labelCol="label",
                                                  metricName="areaUnderROC")
        auc = evaluator.evaluate(prediction_df)
        print("AUC score is: ", auc)


if __name__ == '__main__':
    time_start = time.time()
    parser = argparse.ArgumentParser(description='Tensorflow DIEN Training/Inference')
    # parameters
    parser.add_argument('--cluster_mode', type=str, default="local",
                      help='The cluster mode, such as local, yarn or standalone.')
    parser.add_argument('--master', type=str, default=None,
                      help='The master url, only used when cluster mode is standalone.')
    parser.add_argument('--executor_cores', type=int, default=8,
                      help='The executor core number.')
    parser.add_argument('--executor_memory', type=str, default="160g",
                        help='The executor memory.')
    parser.add_argument('--model', default='DIEN', type=str,
                        help='model type: DIEN-gru-att-augru (default)')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epoch', default=1, type=int, help='train epoch')
    parser.add_argument('--mode', default='train', type=str, choices=['train', 'test'],
                        help='run mode')
    parser.add_argument('--batch_size', default=8000, type=int, help='batch size')
    parser.add_argument('--dir', default='snapshot', type=str,
                        help='snapshot directory name (default: snapshot)')
    parser.add_argument('--data_dir', type=str, help='data directory')
    args = parser.parse_args()

    if args.cluster_mode == "local":
        sc = init_orca_context("local", cores=args.executor_cores, memory=args.executor_memory)
    elif args.cluster_mode == "spark-submit":
        sc = init_orca_context("spark-submit")
    spark = SparkSession(sparkContext=sc)
    tfconfig = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=args.executor_cores)

    train_data, test_data, n_uid, n_mid, n_cat = load_dien_data(spark, args.data_dir)

    if args.mode == 'train':
        train(config=tfconfig, train_data=train_data, test_data=test_data,
              n_uid=n_uid, n_mid=n_mid, n_cat=n_cat,
              batch_size=args.batch_size,
              epoch=args.epoch, lr = args.lr,
              model_type=args.model,
              use_bf16=use_bf16, seed=SEED, snapshot_dir=args.dir)
        # if args.mode == 'test':
        test(config=tfconfig, test_data=test_data,
             n_uid=n_uid, n_mid=n_mid, n_cat=n_cat,
             batch_size=args.batch_size, model_type=args.model,
             use_bf16=use_bf16, seed=SEED, snapshot_dir=args.dir)
    else:
        print('only tain or test are provided')

    time_end = time.time()
    print('perf { total time: %f }' % (time_end - time_start))
    stop_orca_context()
