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

from dataset import base
import numpy as np
import json

IMDB_URL = 'https://s3.amazonaws.com/text-datasets/imdb.npz'

def download_imdb(dest_dir):
    """Download pre-processed IMDB movie review data

    :argument
        dest_dir: destination directory to store the data

    :return
        The absolute path of the stored data
    """
    file_name = "imdb.npz"
    file_abs_path = base.maybe_download(file_name,
                                        dest_dir,
                                        'https://s3.amazonaws.com/text-datasets/imdb.npz')
    return file_abs_path

def load_imdb(dest_dir='/tmp/.bigdl/dataset'):
    """Load IMDB dataset.

    :argument
        dest_dir: where to cache the data (relative to `~/.bigdl/dataset`).

    :return
        the train, test separated IMDB dataset.
    """
    path = download_imdb(dest_dir)
    f = np.load(path)
    x_train = f['x_train']
    y_train = f['y_train']
    x_test = f['x_test']
    y_test = f['y_test']
    f.close()

    return (x_train, y_train), (x_test, y_test)

def get_word_index(dest_dir='/tmp/.bigdl/dataset', ):
    """Retrieves the dictionary mapping word indices back to words.

    :argument
        path: where to cache the data (relative to `~/.bigdl/dataset`).

    :return
        The word index dictionary.
    """
    file_name = "imdb_word_index.json"
    path = base.maybe_download(file_name,
                               dest_dir,
                               source_url='https://s3.amazonaws.com/text-datasets/imdb_word_index.json')
    f = open(path)
    data = json.load(f)
    f.close()
    return data

