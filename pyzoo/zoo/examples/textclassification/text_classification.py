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

import itertools
import re
import datetime as dt
from optparse import OptionParser

from bigdl.optim.optimizer import *
from zoo.common.nncontext import init_nncontext
from zoo.examples.textclassification.news20 import *
from zoo.models.textclassification import TextClassifier
from zoo.pipeline.api.keras.objectives import SparseCategoricalCrossEntropy
from zoo.pipeline.api.keras.metrics import Accuracy


def text_to_words(review_text):
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    return words


def analyze_texts(data_rdd):
    def index(w_c_i):
        ((word, frequency), i) = w_c_i
        return word, (i + 1, frequency)
    return data_rdd.flatMap(lambda text_label: text_to_words(text_label[0])) \
        .map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b) \
        .sortBy(lambda word_frequency: - word_frequency[1]).zipWithIndex() \
        .map(lambda word_frequency_i: index(word_frequency_i)).collect()


def pad(l, fill_value, width):
    if len(l) >= width:
        return l[0: width]
    else:
        l.extend([fill_value] * (width - len(l)))
        return l


def to_vec(token, w2v_bc, embedding_dim):
    if token in w2v_bc:
        return w2v_bc[token]
    else:
        return pad([], 0, embedding_dim)


def to_sample(vectors, label, embedding_dim):
    flatten_features = list(itertools.chain(*vectors))
    features = np.array(flatten_features, dtype='float').reshape(
        [sequence_len, embedding_dim])
    return Sample.from_ndarray(features, np.array(label))


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--data_path", dest="data_path", default="/tmp/text_data")
    parser.add_option("--partition_num", dest="partition_num", default="4")
    parser.add_option("--token_length", dest="token_length", default="200")
    parser.add_option("--sequence_length", dest="sequence_length", default="500")
    parser.add_option("--max_words_num", dest="max_words_num", default="5000")
    parser.add_option("--encoder", dest="encoder", default="cnn")
    parser.add_option("--encoder_output_dim", dest="encoder_output_dim", default="256")
    parser.add_option("--training_split", dest="training_split", default="0.8")
    parser.add_option("-b", "--batch_size", dest="batch_size", default="128")
    parser.add_option("--nb_epoch", dest="nb_epoch", default="20")
    parser.add_option("-l", "--learning_rate", dest="learning_rate", default="0.01")
    parser.add_option("--log_dir", dest="log_dir", default="/tmp/.bigdl")
    parser.add_option("--model", dest="model")

    (options, args) = parser.parse_args(sys.argv)
    data_path = options.data_path
    token_length = int(options.token_length)
    sequence_len = int(options.sequence_length)
    max_words_num = int(options.max_words_num)
    training_split = float(options.training_split)
    batch_size = int(options.batch_size)

    sc = init_nncontext("Text Classification Example")

    print('Processing text dataset...')
    texts = get_news20(base_dir=data_path)
    text_data_rdd = sc.parallelize(texts, options.partition_num)

    word_meta = analyze_texts(text_data_rdd)
    # Remove the top 10 words roughly. You might want to fine tune this.
    word_meta = dict(word_meta[10: max_words_num])
    word_mata_broadcast = sc.broadcast(word_meta)

    word2vec = get_glove(base_dir=data_path, dim=token_length)
    # Ignore those unknown words.
    filtered_word2vec = dict((w, v) for w, v in word2vec.items() if w in word_meta)
    filtered_word2vec_broadcast = sc.broadcast(filtered_word2vec)

    tokens_rdd = text_data_rdd.map(lambda text_label:
                                   ([w for w in text_to_words(text_label[0]) if
                                     w in word_mata_broadcast.value], text_label[1]))
    padded_tokens_rdd = tokens_rdd.map(lambda tokens_label:
                                       (pad(tokens_label[0], "##", sequence_len), tokens_label[1]))
    vector_rdd = padded_tokens_rdd.map(lambda tokens_label:
                                       ([to_vec(w, filtered_word2vec_broadcast.value, token_length)
                                         for w in tokens_label[0]], tokens_label[1]))
    sample_rdd = vector_rdd.map(
        lambda vectors_label: to_sample(vectors_label[0], vectors_label[1], token_length))

    train_rdd, val_rdd = sample_rdd.randomSplit([training_split, 1-training_split])

    if options.model:
        model = TextClassifier.load_model(options.model)
    else:
        model = TextClassifier(CLASS_NUM, token_length, sequence_len,
                               options.encoder, int(options.encoder_output_dim))

    optimizer = Optimizer(
        model=model,
        training_rdd=train_rdd,
        criterion=SparseCategoricalCrossEntropy(),
        end_trigger=MaxEpoch(int(options.nb_epoch)),
        batch_size=batch_size,
        optim_method=Adagrad(learningrate=float(options.learning_rate), learningrate_decay=0.001))
    optimizer.set_validation(
        batch_size=batch_size,
        val_rdd=val_rdd,
        trigger=EveryEpoch(),
        val_method=[Accuracy()])

    log_dir = options.log_dir
    app_name = 'adam-' + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_summary = TrainSummary(log_dir=log_dir, app_name=app_name)
    train_summary.set_summary_trigger("Parameters", SeveralIteration(50))
    val_summary = ValidationSummary(log_dir=log_dir, app_name=app_name)
    optimizer.set_train_summary(train_summary)
    optimizer.set_val_summary(val_summary)

    optimizer.optimize()

    # Predict for probability distributions
    results = model.predict(val_rdd)
    results.take(5)
    # Predict for labels
    result_classes = model.predict_classes(val_rdd)
    print("First five class predictions (label starts from 0):")
    for res in result_classes.take(5):
        print(res)

    sc.stop()
