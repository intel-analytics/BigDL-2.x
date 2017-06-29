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

# Still in experimental stage!

import itertools
import re
from optparse import OptionParser

from bigdl.dataset import sentence
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.util.common import Sample

def prepare_data(sc, folder, vocabsize):
    train_file = folder + "train.txt"
    val_file = folder + "val.txt"

    train_sentences_rdd = sc.textFile(train_file) \
        .map(lambda line: sentence.sentences_split(line))
    pad_sent = train_sentences_rdd.flatMap(lambda x: x). \
        map(lambda sent: sentence.sentences_bipadding(sent))
    tokens = pad_sent.map(lambda pad: sentence.sentence_tokenizer(pad))
    max_len = tokens.map(lambda x: len(x)).max()
    print("max length %s" % max_len)

    words = tokens.flatMap(lambda x: x)
    print("%s words and %s sentences processed in train data" % (words.count(), tokens.count()))

    val_sentences_rdd = sc.textFile(val_file) \
        .map(lambda line: sentence.sentences_split(line))
    val_pad_sent = val_sentences_rdd.flatMap(lambda x: x). \
        map(lambda sent: sentence.sentences_bipadding(sent))
    val_tokens = val_pad_sent.map(lambda pad: sentence.sentence_tokenizer(pad))
    val_max_len = val_tokens.map(lambda x: len(x)).max()
    print("val max length %s" % val_max_len)

    val_words = val_tokens.flatMap(lambda x: x)
    print("%s words and %s sentences processed in validation data" % (val_words.count(), val_tokens.count()))

    sort_words = words.map(lambda w: (w, 1)) \
                .reduceByKey(lambda a, b: a + b) \
                .sortBy(lambda w_c: w_c[1])
    vocabulary = np.array(sort_words.map(lambda w: w[0]).collect())

    fre_len = vocabulary.size
    if vocabsize > fre_len:
        length = fre_len
    else:
        length = vocabsize
    discard_vocab = vocabulary[: fre_len-length]
    used_vocab = vocabulary[fre_len-length: fre_len]
    used_vocab_size = used_vocab.size
    index = np.arange(used_vocab_size)
    index2word = dict(enumerate(used_vocab))
    word2index = dict(zip(used_vocab, index))
    total_vocab_len = used_vocab_size + 1
    startIdx = word2index.get("SENTENCESTART")
    endIdx = word2index.get("SENTENCEEND")

    def text2labeled(sent):
        indexes = [word2index.get(x, used_vocab_size) for x in sent]
        data = indexes[0: -1]
        label = indexes[1: len(indexes)]
        return data, label

    def labeled2onehotformat(labeled_sent):
        label = [x+1 for x in labeled_sent[1]]
        size = len(labeled_sent[0])
        feature_onehot = np.zeros(size * total_vocab_len, dtype='int').reshape(
            [size, total_vocab_len])
        for i, el in enumerate(labeled_sent[0]):
            feature_onehot[i, el] = 1
        return feature_onehot, label

    def padding(features, label):
        pad_len = max_len - len(label)
        print("max_len: %s pad_len: %s" % (max_len, pad_len))
        padded_label = (label + [startIdx] * max_len)[:max_len]
        feature_padding = np.zeros((pad_len, total_vocab_len), dtype=np.int)
        feature_padding[:, endIdx + 1] = np.ones(pad_len)
        padded_feautres = np.concatenate((features, feature_padding), axis=0)
        return padded_feautres, padded_label

    sample_rdd = tokens.map(lambda sentence_te: text2labeled(sentence_te)) \
        .map(lambda labeled_sent: labeled2onehotformat(labeled_sent)) \
        .map(lambda x: padding(x[0], x[1])) \
        .map(lambda vectors_label: Sample.from_ndarray(vectors_label[0], np.array(vectors_label[1])))

    val_sample_rdd = val_tokens.map(lambda sentence_t: text2labeled(sentence_t)) \
        .map(lambda labeled_sent: labeled2onehotformat(labeled_sent)) \
        .map(lambda x: padding(x[0], x[1])) \
        .map(lambda vectors_label: Sample.from_ndarray(vectors_label[0], np.array(vectors_label[1])))

    return sample_rdd, val_sample_rdd, total_vocab_len

def build_model(input_size, hidden_size, output_size):
    model = Sequential()
    model.add(Recurrent()
              .add(RnnCell(input_size, hidden_size, Tanh())))\
        .add(TimeDistributed(Linear(hidden_size, output_size)))
    model.reset()
    return model

if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("-f", "--folder", dest="folder", default="./")
    parser.add_option("-b", "--batchSize", dest="batchSize", default="128")
    parser.add_option("--learningRate", dest="learningrate", default="0.1")
    parser.add_option("--momentum", dest="momentum", default="0.0")
    parser.add_option("--weightDecay", dest="weight_decay", default="0.0")
    parser.add_option("--dampening", dest="dampening", default="0.0")
    parser.add_option("--hiddenSize", dest="hidden_size", default="40")
    parser.add_option("--vocabSize", dest="vob_size", default="4000")
    parser.add_option("--maxEpoch", dest="max_epoch", default="30")

    (options, args) = parser.parse_args(sys.argv)

    batch_size = int(options.batchSize)
    learningrate = float(options.learningrate)
    momentum = float(options.momentum)
    weight_decay = float(options.weight_decay)
    dampening = float(options.dampening)
    hidden_size = int(options.hidden_size)
    vob_size = int(options.vob_size)
    max_epoch = int(options.max_epoch)
    folder = options.folder

    sc = SparkContext(appName="simplernn_example",
                      conf=create_spark_conf())
    init_engine()

    (train_rdd, val_rdd, vob_size) = prepare_data(sc, folder, vob_size)

    optimizer = Optimizer(
        model=build_model(vob_size, hidden_size, vob_size),
        training_rdd=train_rdd,
        criterion=TimeDistributedCriterion(CrossEntropyCriterion(), size_average=True),
        batch_size=batch_size,
        optim_method=SGD(learningrate=learningrate, weightdecay=weight_decay,
                         momentum=momentum, dampening=dampening),
        end_trigger=MaxEpoch(max_epoch)
    )

    optimizer.set_validation(
        batch_size=batch_size,
        val_rdd=val_rdd,
        trigger=EveryEpoch(),
        val_method=["Loss"],
        criterion="TimeDistributedCriterion",
        embedded_cri="CrossEntropyCriterion"
    )

    train_model = optimizer.optimize()
    sc.stop()
