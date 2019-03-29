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

from zoo.tfpark.estimator import *
from bert import modeling


def _bert_model_fn(features, labels, mode, params):
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    bert_config = modeling.BertConfig.from_json_file(params["bert_config"])
    bert_model = modeling.BertModel(
        config=bert_config,
        is_training=(mode == tf.estimator.ModeKeys.TRAIN),
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=params["use_one_hot_embeddings"])
    output_layer = bert_model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value
    output_weights = tf.get_variable(
        "output_weights", [params["num_labels"], hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable(
        "output_bias", [params["num_labels"]], initializer=tf.zeros_initializer())
    with tf.variable_scope("loss"):
        if mode == tf.estimator.ModeKeys.TRAIN:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)

        tvars = tf.trainable_variables()
        if params["init_checkpoint"]:
            assignment_map, initialized_variable_names =\
                modeling.get_assignment_map_from_checkpoint(tvars, params["init_checkpoint"])
            tf.train.init_from_checkpoint(params["init_checkpoint"], assignment_map)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return TFEstimatorSpec(mode=mode, predictions=probabilities)
        else:
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(labels, depth=params["num_labels"], dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
            return TFEstimatorSpec(mode=mode, predictions=probabilities, loss=loss)


class BERTClassifier(TFEstimator):
    def __init__(self, num_labels, bert_config, init_checkpoint=None,
                 use_one_hot_embeddings=False, optimizer=None, model_dir=None):
        super(BERTClassifier, self).__init__(
            model_fn=_bert_model_fn,
            optimizer=optimizer,
            model_dir=model_dir,
            params={
                "num_labels": num_labels,
                "bert_config": bert_config,
                "init_checkpoint": init_checkpoint,
                "use_one_hot_embeddings": use_one_hot_embeddings
            })
