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

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import tensorflow as tf
from bigdl.optim.optimizer import Adam, MaxEpoch
from zoo.common.nncontext import *
from zoo.pipeline.api.net import TFDataset, TFOptimizer
from tokenization import FullTokenizer
from modeling import BertConfig, get_assignment_map_from_checkpoint
from run_classifier import MrpcProcessor, convert_examples_to_features, create_model


def to_list_numpy(feature):
    return [np.array(feature.input_ids), np.array(feature.input_mask),
            np.array(feature.segment_ids), np.array(feature.label_id)]


# Model and data files
bert_base_dir = "/home/kai/Downloads/bert_models/uncased_L-12_H-768_A-12/"
bert_config_file = bert_base_dir + "bert_config.json"
vocab_file = bert_base_dir + "vocab.txt"
init_checkpoint = bert_base_dir + "bert_model.ckpt"
data_dir = "/home/kai/Downloads/glue_data/MRPC/"

# Options
do_lower_case = False
max_seq_length = 128
train_batch_size = 32
eval_batch_size = 8
predict_batch_size = 8
learning_rate = 1e-6
nb_epoch = 50


sc = init_nncontext("BERT Classification Example")

# Data preparation and preprocessing
processor = MrpcProcessor()
label_list = processor.get_labels()
tokenizer = FullTokenizer(vocab_file, do_lower_case)
examples = processor.get_train_examples(data_dir)
features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer)
features = [to_list_numpy(feature) for feature in features]
feature_rdd = sc.parallelize(features)
train_dataset = TFDataset.from_rdd(feature_rdd,
                                   names=["input_ids", "input_mask", "segment_ids", "label_id"],
                                   shapes=[[max_seq_length], [max_seq_length], [max_seq_length], [1]],
                                   types=[tf.int32, tf.int32, tf.int32, tf.int32],
                                   batch_size=train_batch_size)

# Model loading and construction
input_ids, input_mask, segment_ids, label_ids = train_dataset.tensors
bert_config = BertConfig.from_json_file(bert_config_file)
loss, per_example_loss, logits, probabilities = create_model(bert_config, True, input_ids, input_mask, segment_ids,
                                                             label_ids, len(label_list), False)
tvars = tf.trainable_variables()
initialized_variable_names = {}
scaffold_fn = None
if init_checkpoint:
    assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
tf.logging.info("**** Trainable Variables ****")
for var in tvars:
    init_string = ""
    if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

# Training
optimizer = TFOptimizer(loss, Adam(learning_rate))
optimizer.optimize(end_trigger=MaxEpoch(nb_epoch))

print("Finished")
