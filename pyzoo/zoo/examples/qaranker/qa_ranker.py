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

import sys
from optparse import OptionParser

from bigdl.optim.optimizer import SGD
from zoo.common.nncontext import init_nncontext
from zoo.feature.common import Relations
from zoo.feature.text import TextSet
from zoo.models.textmatching import KNRM
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import TimeDistributed


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--data_path", dest="data_path")
    parser.add_option("--embedding_file", dest="embedding_file")
    parser.add_option("--question_length", dest="question_length", default="10")
    parser.add_option("--answer_length", dest="answer_length", default="40")
    parser.add_option("--partition_num", dest="partition_num", default="4")
    parser.add_option("-b", "--batch_size", dest="batch_size", default="200")
    parser.add_option("-e", "--nb_epoch", dest="nb_epoch", default="30")
    parser.add_option("-l", "--learning_rate", dest="learning_rate", default="0.001")
    parser.add_option("-m", "--model", dest="model")
    parser.add_option("--output_path", dest="output_path")

    (options, args) = parser.parse_args(sys.argv)
    sc = init_nncontext("QARanker Example")

    q_set = TextSet.read_csv(options.data_path + "/question_corpus.csv",
                             sc, int(options.partition_num)).tokenize().normalize()\
        .word2idx(min_freq=2).shape_sequence(int(options.question_length))
    a_set = TextSet.read_csv(options.data_path+"/answer_corpus.csv",
                             sc, int(options.partition_num)).tokenize().normalize()\
        .word2idx(min_freq=2, existing_map=q_set.get_word_index())\
        .shape_sequence(int(options.answer_length))

    train_relations = Relations.read(options.data_path + "/relation_train.csv",
                                     sc, int(options.partition_num))
    train_set = TextSet.from_relation_pairs(train_relations, q_set, a_set)
    validate_relations = Relations.read(options.data_path + "/relation_valid.csv",
                                        sc, int(options.partition_num))
    validate_set = TextSet.from_relation_lists(validate_relations, q_set, a_set)

    if options.model:
        knrm = KNRM.load_model(options.model)
    else:
        word_index = a_set.get_word_index()
        knrm = KNRM(int(options.question_length), int(options.answer_length),
                    options.embedding_file, word_index)
    model = Sequential().add(
        TimeDistributed(
            knrm,
            input_shape=(2, int(options.question_length) + int(options.answer_length))))
    model.compile(optimizer=SGD(learningrate=float(options.learning_rate)),
                  loss="rank_hinge")
    for i in range(0, int(options.nb_epoch)):
        model.fit(train_set, batch_size=int(options.batch_size), nb_epoch=1)
        knrm.evaluate_ndcg(validate_set, 3)
        knrm.evaluate_ndcg(validate_set, 5)
        knrm.evaluate_map(validate_set)

    if options.output_path:
        knrm.save_model(options.output_path + "/knrm.model")
        a_set.save_word_index(options.output_path + "/word_index.txt")
        print("Trained model and word dictionary saved")
    sc.stop()
