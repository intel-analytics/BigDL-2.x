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

import datetime as dt
from optparse import OptionParser

from bigdl.optim.optimizer import *
from zoo.common.nncontext import init_nncontext
from zoo.feature.text import TextSet
from zoo.models.textclassification import TextClassifier


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

    text_set = TextSet.read(path=data_path+"/20news-18828/", sc=sc, min_partitions=4)
    print('Processing text dataset...')
    transformed = text_set.tokenize().normalize()\
        .word2idx(remove_topN=10, max_words_num=max_words_num)\
        .shape_sequence(len=sequence_len).gen_sample()
    train_set, val_set = transformed.random_split([training_split, 1 - training_split])

    if options.model:
        model = TextClassifier.load_model(options.model)
    else:
        if not (token_length == 50 or token_length == 100
                or token_length == 200 or token_length == 300):
            raise ValueError('token_length for GloVe can only be 50, 100, 200, 300, but got '
                             + str(token_length))
        embedding_file = data_path + "/glove.6B/glove.6B." + str(token_length) + "d.txt"
        word_index = transformed.get_word_index()
        model = TextClassifier(20, embedding_file, word_index, sequence_len,
                               options.encoder, int(options.encoder_output_dim)).model

    # model.compile(optimizer=Adagrad(learningrate=float(options.learning_rate), learningrate_decay=0.001),
    #               loss="sparse_categorical_crossentropy",
    #               metrics=['accuracy'])
    # log_dir = options.log_dir
    # app_name = 'adam-' + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    # model.set_tensorboard(log_dir, app_name)
    # model.fit(train_set, batch_size=batch_size, nb_epoch=int(options.nb_epoch), validation_data=val_set)
    predict_set = model.predict(val_set)
    # Get the first five prediction probability distributions
    res = predict_set.get_predicts()

    sc.stop()
