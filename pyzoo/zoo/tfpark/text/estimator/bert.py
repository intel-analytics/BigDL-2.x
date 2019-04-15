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

from bigdl.util.common import get_node_and_core_number
from zoo.tfpark.estimator import *
from bert import modeling


def _bert_model(features, labels, mode, params):
    """
    Return an instance of BertModel and one can take its different outputs to
    perform specific tasks.
    """
    input_ids = features["input_ids"]
    if "input_mask" in features:
        input_mask = features["input_mask"]
    else:
        input_mask = None
    if "token_type_ids" in features:
        token_type_ids = features["token_type_ids"]
    else:
        token_type_ids = None
    bert_config = modeling.BertConfig.from_json_file(params["bert_config_file"])
    bert_model = modeling.BertModel(
        config=bert_config,
        is_training=(mode == tf.estimator.ModeKeys.TRAIN),
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=token_type_ids,
        use_one_hot_embeddings=params["use_one_hot_embeddings"])
    tvars = tf.trainable_variables()
    if params["init_checkpoint"]:
        assignment_map, initialized_variable_names = \
            modeling.get_assignment_map_from_checkpoint(tvars, params["init_checkpoint"])
        tf.train.init_from_checkpoint(params["init_checkpoint"], assignment_map)
    return bert_model


def _bert_classifier_model_fn(features, labels, mode, params):
    """
    Model function for BERTClassifier.

    :param features: Dict of feature tensors. Must include the key "input_ids".
    :param labels: Label tensor for training.
    :param mode: 'train', 'eval' or 'infer'.
    :param params: Must include the key "num_classes".
    :return: TFEstimatorSpec.
    """
    output_layer = _bert_model(features, labels, mode, params).get_pooled_output()
    hidden_size = output_layer.shape[-1].value
    output_weights = tf.get_variable(
        "output_weights", [params["num_classes"], hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable(
        "output_bias", [params["num_classes"]], initializer=tf.zeros_initializer())
    with tf.variable_scope("loss"):
        if mode == tf.estimator.ModeKeys.TRAIN:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        if mode == tf.estimator.ModeKeys.PREDICT or mode == tf.estimator.ModeKeys.EVAL:
            return TFEstimatorSpec(mode=mode, predictions=probabilities)
        else:
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(labels, depth=params["num_classes"], dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
            return TFEstimatorSpec(mode=mode, loss=loss)


def bert_input_fn(rdd, max_seq_length, batch_size, labels=None,
                  features={"input_ids", "input_mask", "token_type_ids"}):
    """
    Takes an RDD to create the input function for BERT related TFEstimators.
    For training and evaluation, each element in rdd should be a tuple: (dict of features, label).
    For prediction, each element in rdd should be a dict of features.
    """
    assert features.issubset({"input_ids", "input_mask", "token_type_ids"})
    features_dict = {}
    for feature in features:
        features_dict[feature] = (tf.int32, [max_seq_length])
    if labels is None:
        labels = (tf.int32, [])

    def input_fn(mode):
        if mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.TRAIN:
            return TFDataset.from_rdd(rdd,
                                      features=features_dict,
                                      labels=labels,
                                      batch_size=batch_size)
        else:
            node_num, core_num = get_node_and_core_number()
            return TFDataset.from_rdd(rdd,
                                      features=features_dict,
                                      batch_per_thread=batch_size // (node_num * core_num))
    return input_fn


class BERTBaseEstimator(TFEstimator):
    """
    The base class for BERT related TFEstimators.
    Common arguments:
    bert_config_file, init_checkpoint, use_one_hot_embeddings, optimizer, model_dir.

    For its subclass:
    - One can add additional arguments and access them via `params`.
    - One can utilize `_bert_model` to create model_fn and `bert_input_fn` to create input_fn.
    """
    def __init__(self, model_fn, bert_config_file, init_checkpoint=None,
                 use_one_hot_embeddings=False, optimizer=None, model_dir=None, **kwargs):
        params = {"bert_config_file": bert_config_file,
                  "init_checkpoint": init_checkpoint,
                  "use_one_hot_embeddings": use_one_hot_embeddings}
        for k, v in kwargs.items():
            params[k] = v
        super(BERTBaseEstimator, self).__init__(
            model_fn=model_fn,
            optimizer=optimizer,
            model_dir=model_dir,
            params=params)


class BERTClassifier(BERTBaseEstimator):
    """
    A pre-built TFEstimator that takes the hidden state of the first token to do classification.

    :param num_classes: Positive int. The number of classes to be classified.
    :param bert_config_file: The path to the json file for BERT configurations.
    :param init_checkpoint: The path to the initial checkpoint of the pre-trained BERT model if any.
                            Default is None.
    :param use_one_hot_embeddings: Boolean. Whether to use one-hot for word embeddings.
                                   Default is False.
    :param optimizer: The optimizer used to train the estimator. It can either be an instance of
                      tf.train.Optimizer or the corresponding string representation.
                      Default is None if no training is involved.
    :param model_dir: The output directory for model checkpoints to be written if any.
                      Default is None.
    """
    def __init__(self, num_classes, bert_config_file, init_checkpoint=None,
                 use_one_hot_embeddings=False, optimizer=None, model_dir=None):
        super(BERTClassifier, self).__init__(
            model_fn=_bert_classifier_model_fn,
            bert_config_file=bert_config_file,
            init_checkpoint=init_checkpoint,
            use_one_hot_embeddings=use_one_hot_embeddings,
            optimizer=optimizer,
            model_dir=model_dir,
            num_classes=num_classes)
