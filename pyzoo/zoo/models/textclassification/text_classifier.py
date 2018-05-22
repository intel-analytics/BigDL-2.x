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

from zoo.models.common.zoo_model import ZooModel
from bigdl.util.common import callBigDlFunc


if sys.version >= '3':
    long = int
    unicode = str


class TextClassifier(ZooModel):
    """
    The model used for text classification.

    # Arguments
    class_num: The number of text categories to be classified. Positive int.
    token_length: The size of each word vector. Positive int.
    sequence_length: The length of a sequence. Positive int. Default is 500.
    encoder: The encoder for input sequences. String. 'cnn' or 'lstm' or 'gru' are supported.
             Default is 'cnn'.
    encoder_output_dim: The output dimension for the encoder. Positive int. Default is 256.
    """
    def __init__(self, class_num, token_length, sequence_length=500,
                 encoder="cnn", encoder_output_dim=256, bigdl_type="float"):
        super(TextClassifier, self).__init__(None, bigdl_type,
                                             int(class_num),
                                             int(token_length),
                                             int(sequence_length),
                                             encoder,
                                             int(encoder_output_dim))

    @staticmethod
    def load_model(path, weight_path=None, bigdl_type="float"):
        """
        Load an existing TextClassifier model (with weights).

        # Arguments
        path: The path to save the model. Local file system, HDFS and Amazon S3 are supported.
              HDFS path should be like 'hdfs://[host]:[port]/xxx'.
              Amazon S3 path should be like 's3a://bucket/xxx'.
        weight_path: The path for pre-trained weights if any. Default is None.
        """
        jmodel = callBigDlFunc(bigdl_type, "loadTextClassifier", path, weight_path)
        model = ZooModel._do_load(jmodel, bigdl_type)
        model.__class__ = TextClassifier
        return model
