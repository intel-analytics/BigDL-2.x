from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import urllib
from itertools import islice

import tensorflow as tf

from zoo import init_nncontext
from zoo.tfpark import TFDataset, ZooOptimizer, TFEstimator

dftrain = pd.read_csv('/home/yang/sources/datasets/titanic/train.csv')
dfeval = pd.read_csv('/home/yang/sources/datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')


CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

_CSV_COLUMNS = ['survived','sex','age','n_siblings_spouses','parch','fare','class','deck','embark_town','alone']
_CSV_COLUMN_DEFAULTS = [0,'male',22.0,1,0,7.25,'Third','unknown','Southampton','n']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

sc = init_nncontext()

def make_input_fn(data_path, batch_size=-1, batch_per_thread=-1):

  def parse_csv(line):
    columns = tf.io.decode_csv(line, _CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    labels = features.pop('survived')
    return features, labels

  def input_function():
    rdd = sc.textFile(data_path).mapPartitionsWithIndex(lambda idx, it: islice(it, 1, None) if idx == 0 else it)
    ds = TFDataset.from_string_rdd(rdd, batch_size=batch_size, batch_per_thread=batch_per_thread)
    ds = ds.map(lambda x: parse_csv(x[0]))
    return ds
  return input_function

train_input_fn = make_input_fn("/home/yang/sources/datasets/titanic/train.csv", batch_size=32)
eval_input_fn = make_input_fn("/home/yang/sources/datasets/titanic/eval.csv", batch_per_thread=32)



# print(ds.output_types)
# for feature_batch, label_batch in ds.take(1):
#   print('Some feature keys:', list(feature_batch.keys()))
#   print()
#   print('A batch of class:', feature_batch['class'].numpy())
#   print()
#   print('A batch of Labels:', label_batch.numpy())

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns,
                                           optimizer=ZooOptimizer(tf.train.FtrlOptimizer(0.2)))
zoo_est = TFEstimator(linear_est.model_fn, model_dir="/tmp/estimator/linear")
zoo_est.train(train_input_fn, steps=200)
# result = zoo_est.evaluate(eval_input_fn, ["acc"])
# print(result)