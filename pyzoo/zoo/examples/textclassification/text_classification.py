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

import re
import datetime as dt
from optparse import OptionParser

from bigdl.optim.optimizer import *
from zoo.common.nncontext import init_nncontext
from zoo.examples.textclassification.news20 import get_news20
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
    parser.add_option("--log_dir", dest="log_dir", default="/tmp/.zoo")
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
    texts, class_num = get_news20(base_dir=data_path)
    text_data_rdd = sc.parallelize(texts, options.partition_num)

    word_meta = analyze_texts(text_data_rdd)
    # Remove the top 10 words roughly. You might want to fine tune this.
    word_meta = dict(word_meta[10: max_words_num])
    word_mata_broadcast = sc.broadcast(word_meta)

    indexed_rdd = text_data_rdd\
        .map(lambda text_label:
             ([word_mata_broadcast.value[w][0] for w in text_to_words(text_label[0]) if
               w in word_mata_broadcast.value], text_label[1]))\
        .map(lambda tokens_label:
             (pad(tokens_label[0], 0, sequence_len), tokens_label[1]))
    sample_rdd = indexed_rdd.map(
        lambda features_label:
            Sample.from_ndarray(np.array(features_label[0]), features_label[1]))
    train_rdd, val_rdd = sample_rdd.randomSplit([training_split, 1 - training_split])

    if options.model:
        model = TextClassifier.load_model(options.model)
    else:
        if not (token_length == 50 or token_length == 100
                or token_length == 200 or token_length == 300):
            raise ValueError('token_length for GloVe can only be 50, 100, 200, 300, but got '
                             + str(token_length))
        embedding_file = data_path + "/glove.6B/glove.6B." + str(token_length) + "d.txt"
        word_index = {w: i_f[0] for w, i_f in word_meta.items()}
        model = TextClassifier(class_num, embedding_file, word_index, sequence_len,
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
