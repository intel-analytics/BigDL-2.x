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
from pyspark.streaming import StreamingContext

from zoo.common.nncontext import init_nncontext
from zoo.feature.text import DistributedTextSet
from zoo.models.textclassification import TextClassifier


def read_word_index(path):
    word_to_index = dict()
    with open(path, "r") as index_file:
        for line in index_file:
            word, index = line.split(" ")
            word_to_index[word] = int(index)
    return word_to_index


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--index_path", dest="index_path")
    parser.add_option("--partition_num", dest="partition_num", default="4")
    parser.add_option("--token_length", dest="token_length", default="200")
    parser.add_option("--sequence_length", dest="sequence_length", default="500")
    parser.add_option("--max_words_num", dest="max_words_num", default="5000")
    parser.add_option("-b", "--batch_size", dest="batch_size", default="128")
    parser.add_option("-m", "--model", dest="model")

    (options, args) = parser.parse_args(sys.argv)
    sc = init_nncontext("Text Classification Example")
    ssc = StreamingContext(sc, 1)

    lines = ssc.socketTextStream("localhost", 9999)
    model = TextClassifier.load_model(options.model)
    word2index = read_word_index(options.index_path)

    def predict(record):
        if record.getNumPartitions() == 0:
            return
        text_set = DistributedTextSet(record)
        # TODO waiting for Kai's pr
        # text_set.set_word_index(word2index)
        print("Processing text dataset...")
        transformed = text_set.tokenize().normalize()\
            .word2idx(remove_topN=10, max_words_num=int(options.max_words_num))\
            .shape_sequence(len=int(options.sequence_length)).generate_sample()
        predict_set = model.predict(transformed, batch_per_thread=int(options.partition_num))
        # Get the first five prediction probability distributions
        predicts = predict_set.get_predicts().take(5)
        print("Probability distributions of the first five texts in the validation set:")
        for p in predicts:
            print(p)

    lines.foreachRDD(predict)
    # Start the computation
    ssc.start()
    # Wait for the computation to terminate
    ssc.awaitTermination()

