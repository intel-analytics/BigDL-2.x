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


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--host", dest="host",
                      default="localhost")
    parser.add_option("--port", dest="port",
                      default="9999")
    parser.add_option("--index_path", dest="index_path")
    parser.add_option("--partition_num", dest="partition_num",
                      default="4")
    parser.add_option("--sequence_length", dest="sequence_length",
                      default="500")
    parser.add_option("-b", "--batch_size", dest="batch_size",
                      default="128")
    parser.add_option("-m", "--model", dest="model")

    (options, args) = parser.parse_args(sys.argv)

    sc = init_nncontext("Text Classification Example")
    ssc = StreamingContext(sc, 3)
    lines = ssc.socketTextStream(options.host, int(options.port))

    model = TextClassifier.load_model(options.model)

    def predict(record):
        if record.getNumPartitions() == 0:
            return
        text_set = DistributedTextSet(record)
        text_set.load_word_index(options.index_path)
        print("Processing text...")
        transformed = text_set.tokenize().normalize()\
            .word2idx()\
            .shape_sequence(len=int(options.sequence_length)).generate_sample()
        predict_set = model.predict(transformed, int(options.partition_num))
        # Get the first five prediction probability distributions
        predicts = predict_set.get_predicts().take(5)
        print("Probability distributions of top-5 texts:")
        for p in predicts:
            print(p)

    lines.foreachRDD(predict)
    # Start the computation
    ssc.start()
    # Wait for the computation to terminate
    ssc.awaitTermination()
