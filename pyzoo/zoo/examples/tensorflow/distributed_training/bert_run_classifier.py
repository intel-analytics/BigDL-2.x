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

from optparse import OptionParser

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


def generate_tf_dataset(examples, seq_len, batch_size):
    features = convert_examples_to_features(examples, label_list, seq_len, tokenizer)
    features = [to_list_numpy(feature) for feature in features]
    rdd = getOrCreateSparkContext().parallelize(features)
    return TFDataset.from_rdd(rdd,
                              names=["input_ids", "input_mask", "segment_ids", "label_id"],
                              shapes=[[seq_len], [seq_len], [seq_len], [1]],
                              types=[tf.int32, tf.int32, tf.int32, tf.int32],
                              batch_size=batch_size)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--bert_base_dir", dest="bert_base_dir")
    parser.add_option("--data_dir", dest="data_dir")
    parser.add_option("--output_dir", dest="output_dir")
    parser.add_option("--batch_size", dest="batch_size", type=int, default=112)
    parser.add_option("--max_seq_length", dest="max_seq_length", type=int, default=128)
    parser.add_option("-e", "--nb_epoch", dest="nb_epoch", type=int, default=3)
    parser.add_option("-l", "--learning_rate", dest="learning_rate", type=float, default=2e-5)

    (options, args) = parser.parse_args(sys.argv)
    # Model and data files
    bert_config_file = options.bert_base_dir + "/bert_config.json"
    vocab_file = options.bert_base_dir + "/vocab.txt"
    init_checkpoint = options.bert_base_dir + "/bert_model.ckpt"

    sc = init_nncontext("BERT MRPC Classification Example")

    # Data preparation and preprocessing
    processor = MrpcProcessor()
    label_list = processor.get_labels()
    tokenizer = FullTokenizer(vocab_file, False)
    train_examples = processor.get_train_examples(options.data_dir)
    train_dataset = generate_tf_dataset(train_examples, options.max_seq_length, options.batch_size)
    eval_examples = processor.get_dev_examples(options.data_dir)
    eval_dataset = generate_tf_dataset(eval_examples, options.max_seq_length, options.batch_size)

    # Model loading and construction
    input_ids, input_mask, segment_ids, label_ids = train_dataset.tensors
    label_ids = tf.squeeze(label_ids)
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
    optimizer = TFOptimizer(loss, Adam(options.learning_rate))
    optimizer.optimize(end_trigger=MaxEpoch(options.nb_epoch))
    saver = tf.train.Saver()
    saver.save(optimizer.sess, options.output_dir + "/model")

    print("Finished")
