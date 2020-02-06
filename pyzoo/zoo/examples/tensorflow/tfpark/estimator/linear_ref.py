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
_CSV_COLUMN_DEFAULTS = [[0],['male'],[22.0],[1],[0],[7.25],['Third'],['unknown'],['Southampton'],['n']]
_CSV_COLUMN_DEFAULTS = [0,'male',22.0,1,0,7.25,'Third','unknown','Southampton','n']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    if shuffle:
      ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).repeat(num_epochs)
    return ds
  return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns, model_dir="/tmp/estimator/linear")
# linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

print(result)