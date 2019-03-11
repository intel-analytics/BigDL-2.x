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

import argparse
import pickle
from os import path

from sklearn.metrics import classification_report
from tensorflow.python.keras.utils import to_categorical

from nlp_architect.data.intent_datasets import SNIPS
from nlp_architect.utils.generic import one_hot
from nlp_architect.utils.io import validate, validate_existing_directory, \
    validate_existing_filepath, validate_parent_exists
from nlp_architect.utils.metrics import get_conll_scores
from zoo.common.nncontext import *
from zoo.tfpark import IntentNER


def validate_input_args():
    validate((args.b, int, 1, 100000000))
    validate((args.e, int, 1, 100000000))
    validate((args.sentence_length, int, 1, 10000))
    validate((args.token_emb_size, int, 1, 10000))
    validate((args.intent_hidden_size, int, 1, 10000))
    validate((args.lstm_hidden_size, int, 1, 10000))
    validate((args.tagger_dropout, float, 0, 1))
    model_info_path = path.join(path.dirname(path.realpath(__file__)), str(args.model_info_path))
    validate_parent_exists(model_info_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=12,
                        help='Batch size')
    parser.add_argument('-e', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--dataset_path', type=validate_existing_directory, required=True,
                        help='dataset directory')
    parser.add_argument('--sentence_length', type=int, default=30,
                        help='Max sentence length')
    parser.add_argument('--token_emb_size', type=int, default=100,
                        help='Token features embedding vector size')
    parser.add_argument('--intent_hidden_size', type=int, default=100,
                        help='Intent detection LSTM hidden size')
    parser.add_argument('--lstm_hidden_size', type=int, default=150,
                        help='Slot tags LSTM hidden size')
    parser.add_argument('--tagger_dropout', type=float, default=0.5,
                        help='Slot tags dropout value')
    parser.add_argument('--embedding_model', type=validate_existing_filepath,
                        help='Path to word embedding model file')
    parser.add_argument('--use_cudnn', default=False, action='store_true',
                        help='use CUDNN based LSTM cells')
    parser.add_argument('--export_path', type=str, default='/tmp/mtl/',
                        help='Export TensorFlow path')
    parser.add_argument('--model_info_path', type=str, default='model_info.dat',
                        help='Path for saving model topology')
    args = parser.parse_args()
    validate_input_args()

    conf = init_spark_conf().setMaster("local[1]").setAppName("Intent Extraction and NER example")
    sc = init_nncontext(conf)

    # load dataset
    print('Loading dataset')
    dataset = SNIPS(path=args.dataset_path,
                    sentence_length=args.sentence_length)

    train_x, train_char, train_i, train_y = dataset.train_set
    test_x, test_char, test_i, test_y = dataset.test_set
    intent_truth = test_i

    test_y = to_categorical(test_y, dataset.label_vocab_size)
    train_y = to_categorical(train_y, dataset.label_vocab_size)
    train_i = one_hot(train_i, len(dataset.intents_vocab))
    test_i = one_hot(test_i, len(dataset.intents_vocab))

    model = IntentNER(dataset.word_len,
                      dataset.label_vocab_size,
                      dataset.intent_size,
                      dataset.word_vocab_size,
                      dataset.char_vocab_size,
                      word_emb_dims=args.token_emb_size,
                      tagger_lstm_dims=args.lstm_hidden_size,
                      dropout=args.tagger_dropout)
    model.fit([train_x, train_char], [train_i, train_y], batch_size=args.b,
              distributed=True, epochs=args.e)
    # [total_loss, intent_loss, ner_loss, intent_acc, ner_acc]
    eval_res = model.evaluate([test_x, test_char], [test_i, test_y], distributed=True)
    predictions = model.predict([test_x, test_char], distributed=True)

    print("Intent evaluation results: ")
    intent_classes = sorted([(k, v) for k, v in dataset.intents_vocab.vocab.items()], key=lambda k: k[1])
    print(classification_report(intent_truth, np.argmax(np.stack([x[0] for x in predictions]), axis=1),
                                target_names=[i[0] for i in intent_classes], labels=[i[1] for i in intent_classes]))
    print("NER evaluation results: ")
    eval = get_conll_scores(np.stack([x[1] for x in predictions]), test_y,
                            {v: k for k, v in dataset.tags_vocab.vocab.items()})
    print(eval)
    sc.stop()
