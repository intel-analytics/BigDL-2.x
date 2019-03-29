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
from zoo.common.nncontext import *
from zoo.pipeline.api.net import TFDataset
from zoo.tfpark.text.estimator import BERTClassifier
from bert.tokenization import FullTokenizer
from bert.run_classifier import MrpcProcessor, convert_examples_to_features


def feature_to_input(feature):
    res = dict()
    res["input_ids"] = np.array(feature.input_ids)
    res["input_mask"] = np.array(feature.input_mask)
    res["token_type_ids"] = np.array(feature.segment_ids)
    return res, np.array(feature.label_id)


def input_fn_builder(examples, label_list, max_seq_length, tokenizer, batch_size):
    features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer)
    features = [feature_to_input(feature) for feature in features]
    rdd = sc.parallelize(features)

    def input_fn(mode):
        if mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.TRAIN:
            return TFDataset.from_rdd(rdd,
                                      features={"input_ids": (tf.int32, [max_seq_length]),
                                                "input_mask": (tf.int32, [max_seq_length]),
                                                "token_type_ids": (tf.int32, [max_seq_length])},
                                      labels=(tf.int32, []),
                                      batch_size=batch_size)
        else:
            return TFDataset.from_rdd(rdd.map(lambda x: x[0]),
                                      features={"input_ids": (tf.int32, [max_seq_length]),
                                                "input_mask": (tf.int32, [max_seq_length]),
                                                "token_type_ids": (tf.int32, [max_seq_length])},
                                      batch_per_thread=4)
    return input_fn


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--bert_base_dir", dest="bert_base_dir")
    parser.add_option("--data_dir", dest="data_dir")
    parser.add_option("--output_dir", dest="output_dir")
    parser.add_option("--batch_size", dest="batch_size", type=int, default=32)
    parser.add_option("--max_seq_length", dest="max_seq_length", type=int, default=128)
    parser.add_option("-e", "--nb_epoch", dest="nb_epoch", type=int, default=3)
    parser.add_option("-l", "--learning_rate", dest="learning_rate", type=float, default=2e-5)

    (options, args) = parser.parse_args(sys.argv)
    sc = init_nncontext("BERT MRPC Classification Example")

    processor = MrpcProcessor()
    label_list = processor.get_labels()
    tokenizer = FullTokenizer(options.bert_base_dir + "/vocab.txt")
    train_examples = processor.get_train_examples(options.data_dir)
    train_input_fn = input_fn_builder(train_examples, label_list, options.max_seq_length, tokenizer, options.batch_size)
    eval_examples = processor.get_dev_examples(options.data_dir)
    eval_input_fn = input_fn_builder(eval_examples, label_list, options.max_seq_length, tokenizer, options.batch_size)
    test_examples = processor.get_test_examples(options.data_dir)
    test_input_fn = input_fn_builder(test_examples, label_list, options.max_seq_length, tokenizer, options.batch_size)

    estimator = BERTClassifier(len(label_list), bert_config_file=options.bert_base_dir + "/bert_config.json",
                               init_checkpoint=options.bert_base_dir + "/bert_model.ckpt",
                               optimizer=tf.train.AdamOptimizer(options.learning_rate),
                               model_dir=options.output_dir)
    estimator.train(train_input_fn, steps=len(train_examples)*options.nb_epoch//options.batch_size)
    result = estimator.evaluate(eval_input_fn, eval_methods=["acc"])
    print(result)
    predictions = estimator.predict(test_input_fn)
    for prediction in predictions.take(5):
        print(prediction)

    print("Finished")
