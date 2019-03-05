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


from bigdl.dataset import base
import numpy as np

from six.moves import cPickle
import sys

def download_imdb(dest_dir):
    """Download pre-processed IMDB movie review data

    :argument
        dest_dir: destination directory to store the data

    :return
        The absolute path of the stored data
    """
    file_name = "imdb_full.pkl"
    file_abs_path = base.maybe_download(file_name,
                                        dest_dir,
                                        'https://s3.amazonaws.com/text-datasets/imdb_full.pkl')
    return file_abs_path


def load_data(dest_dir='/tmp/.bigdl/dataset', nb_words=None, oov_char=2):
    """Load IMDB dataset.

    :argument
        dest_dir: where to cache the data (relative to `~/.bigdl/dataset`).
        nb_words: number of words to keep, the words are already indexed by frequency
                  so that the less frequent words would be abandoned
        oov_char: index to pad the abandoned words, if None, one abandoned word 
                  would be taken place with its next word and total length -= 1

    :return
        the train, test separated IMDB dataset.
    """
    path = download_imdb(dest_dir)
    '''
    f = np.load(path)
    x_train = f['x_train']
    y_train = f['y_train']
    x_test = f['x_test']
    y_test = f['y_test']
    '''
    f = open(path, 'rb')

    (x_train, y_train), (x_test, y_test) = cPickle.load(f)
    # imdb.pkl would return different numbers of variables, not 4

    f.close()

    x = x_train + x_test

    if not nb_words:
        nb_words = max([max(s) for s in x])

    if oov_char is not None:
        x = [[oov_char if word >= nb_words else word for word in s] for s in x]
    else:
        new_x = []
        for s in x:
            new_s = []
            for word in s:
                if word < nb_words:
                    new_s.append(word)
            new_x.append(new_s)
        x = new_x

    x_train = np.array(x[:len(x_train)])
    y_train = np.array(y_train)

    x_test = np.array(x[len(x_test):])
    y_test = np.array(y_test)

    return (x_train, y_train), (x_test, y_test)


def get_word_index(dest_dir='/tmp/.bigdl/dataset', path='imdb_word_index.pkl'):
    """Retrieves the dictionary mapping word indices back to words.

    # Arguments
        path: where to cache the data (relative to `~/.keras/dataset`).

    # Returns
        The word index dictionary.
    """

    path = base.maybe_download(path,
                    work_directory=dest_dir,
                    source_url='https://s3.amazonaws.com/text-datasets/imdb_word_index.pkl',
                    )
    f = open(path, 'rb')

    if sys.version_info < (3,):
        data = cPickle.load(f)
    else:
        data = cPickle.load(f, encoding='latin1')

    f.close()
    return data


if __name__ == "__main__":
    print('Processing text dataset')
    (x_train, y_train), (x_test, y_test) = load_data()
    print('finished processing text')
