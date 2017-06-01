#
# Copyright 2016 The BigDL Authors.
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

from dataset import news20
import imdb
from nn.layer import *
from nn.criterion import *
from optim.optimizer import *
from util.common import *
import numpy as np
import itertools

padding_value = 1
start_char = 2
oov_char = 3
index_from = 3

# pad([1, 2, 3, 4, 5], 0, 6)
def pad_sequence(x, fill_value, length):
    """
    Pads each sequence to the same length
    :param x: a sequence
    :param fill_value: pad the sequence with this value
    :param length: pad sequence to the length

    :return: the padded sequence
    """
    if len(x) >= length:
        return x[(len(x) - length):]
    else:
        return [fill_value] * (length - len(x)) + x

def replace_oov(x, oov_char, max_words):
    """
    Replace the words out of vocabulary with `oov_char`
    :param x: a sequence
    :param max_words: the max number of words to include
    :param oov_char: words out of vocabulary because of exceeding the `max_words`
        limit will be replaced by this character

    :return: The replaced sequence
    """
    return [oov_char if w >= max_words else w for w in x]

def to_sample(features, label):
    """
    Wrap the `features` and `label` to a training sample object
    :param features: features of a sample
    :param label: label of a sample

    :return: a sample object including features and label
    """
    return Sample.from_ndarray(np.array(features, dtype='float'), np.array(label))

def build_model(w2v):
    model = Sequential()

    embedding = LookupTable(max_words, embedding_dim)
    embedding.set_weights([w2v])
    print('lookupTable weight: ', embedding.get_weights())
    model.add(embedding)
    if model_type.lower() == "gru":
        model.add(Recurrent()
                  .add(GRU(embedding_dim, 128, p))) \
            .add(Select(2, -1))
    elif model_type.lower() == "lstm":
        model.add(Recurrent()
                  .add(LSTM(embedding_dim, 128, p)))\
            .add(Select(2, -1))
    elif model_type.lower() == "bi_lstm":
        model.add(BiRecurrent(CAddTable())
                  .add(LSTM(embedding_dim, 128, p)))\
            .add(Select(2, -1))
    elif model_type.lower() == "cnn":
        model.add(Transpose([(2, 3)]))\
            .add(Dropout(0.2))\
            .add(Reshape([embedding_dim, 1, sequence_len]))\
            .add(SpatialConvolution(embedding_dim, 128, 5, 1))\
            .add(ReLU())\
            .add(SpatialMaxPooling(sequence_len - 5 + 1, 1, 1, 1))\
            .add(Reshape([128]))
    elif model_type.lower() == "cnn_lstm":
        model.add(Transpose([(2, 3)]))\
            .add(Dropout(0.2))\
            .add(Reshape([embedding_dim, 1, sequence_len])) \
            .add(SpatialConvolution(embedding_dim, 64, 5, 1)) \
            .add(ReLU()) \
            .add(SpatialMaxPooling(4, 1, 1, 1)) \
            .add(Squeeze(3)) \
            .add(Transpose([(2, 3)])) \
            .add(Recurrent()
                 .add(LSTM(64, 128, p))) \
            .add(Select(2, -1))

    model.add(Linear(128, 100))\
        .add(Dropout(0.2))\
        .add(ReLU())\
        .add(Linear(100, 1))\
        .add(Sigmoid())

    return model


def train(sc,
          batch_size,
          sequence_len, max_words, embedding_dim):
    print('Processing text dataset')
    (x_train, y_train), (x_test, y_test) = imdb.load_imdb()
    print('finished processing text')
    print('training set length: ', len(x_train))
    print('testing set length: ', len(x_test))
    word_idx = imdb.get_word_index()
    idx_word = {v:k for k,v in word_idx.items()}


    train_rdd = sc.parallelize(zip(x_train, y_train), 2) \
        .map(lambda (x, y): ([start_char] + [w + index_from for w in x] , y))\
        .map(lambda (x, y): (replace_oov(x, oov_char, max_words), y))\
        .map(lambda (x, y): (pad_sequence(x, padding_value, sequence_len), y))\
        .map(lambda (x, y): to_sample(x, y))
    test_rdd = sc.parallelize(zip(x_test, y_test), 2) \
        .map(lambda (x, y): ([start_char] + [w + index_from for w in x], y))\
        .map(lambda (x, y): (replace_oov(x, oov_char, max_words), y))\
        .map(lambda (x, y): (pad_sequence(x, padding_value, sequence_len), y))\
        .map(lambda (x, y): to_sample(x, y))

    glove = news20.get_glove_w2v(source_dir='/tmp/.bigdl/dataset', dim=embedding_dim)
    w2v = [glove.get(idx_word.get(i - index_from), np.random.uniform(-0.05, 0.05, embedding_dim))
           for i in xrange(1, max_words + 1)]
    w2v = np.array(list(itertools.chain(*np.array(w2v, dtype='float'))), dtype='float') \
        .reshape([max_words, embedding_dim])

    optimizer = Optimizer(
        model=build_model(w2v),
        training_rdd=train_rdd,
        criterion=BCECriterion(),
        end_trigger=MaxEpoch(max_epoch),
        batch_size=batch_size,
        optim_method=Adam())

    print("modelmodel: ", sequence_len, " ", batch_size, " ", embedding_dim)
    optimizer.set_validation(
        batch_size=batch_size,
        val_rdd=test_rdd,
        trigger=EveryEpoch(),
        val_method=["Top1Accuracy"]
    )
    train_model = optimizer.optimize()

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-a", "--action", dest="action", default="train")
    parser.add_option("-b", "--batch_size", dest="batch_size", default="128")
    parser.add_option("-e", "--embedding_dim", dest="embedding_dim", default="50")  # noqa
    parser.add_option("-m", "--max_epoch", dest="max_epoch", default="15")
    parser.add_option("-s", "--sequence_len", dest="sequence_len", default="500")
    parser.add_option("--max_words", dest="max_words", default="10000")
    parser.add_option("--model", dest="model_type", default="lstm")
    parser.add_option("-p", "--p", dest="p", default="0.0")

    (options, args) = parser.parse_args(sys.argv)
    if options.action == "train":
        batch_size = int(options.batch_size)
        embedding_dim = int(options.embedding_dim)
        max_epoch = int(options.max_epoch)
        p = float(options.p)
        model_type = options.model_type
        sequence_len = int(options.sequence_len)
        max_words = int(options.max_words)
        sc = SparkContext(appName="text_classifier",
                          conf=create_spark_conf())
        init_engine()
        train(sc,
              batch_size,
              sequence_len, max_words, embedding_dim)
    elif options.action == "test":
        pass
