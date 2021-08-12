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

import tarfile
from bigdl.dataset import base
import os
import sys

NEWS20_URL = 'http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz'  # noqa
GLOVE_URL = 'http://nlp.stanford.edu/data/glove.6B.zip'  # noqa

CLASS_NUM = 20


def download_news20(dest_dir):
    file_name = "20news-18828.tar.gz"
    file_abs_path = base.maybe_download(file_name, dest_dir, NEWS20_URL)
    tar = tarfile.open(file_abs_path, "r:gz")
    extracted_to = os.path.join(dest_dir, "20news-18828")
    if not os.path.exists(extracted_to):
        print("Extracting %s to %s" % (file_abs_path, extracted_to))
        tar.extractall(dest_dir)
        tar.close()
    return extracted_to


def download_glove_w2v(dest_dir):
    file_name = "glove.6B.zip"
    file_abs_path = base.maybe_download(file_name, dest_dir, GLOVE_URL)
    import zipfile
    zip_ref = zipfile.ZipFile(file_abs_path, 'r')
    extracted_to = os.path.join(dest_dir, "glove.6B")
    if not os.path.exists(extracted_to):
        print("Extracting %s to %s" % (file_abs_path, extracted_to))
        zip_ref.extractall(extracted_to)
        zip_ref.close()
    return extracted_to


def get_news20(source_dir="./data/news20/"):
    """
    Parse or download news20 if source_dir is empty.

    :param source_dir: The directory storing news data.
    :return: A list of (tokens, label)
    """
    news_dir = download_news20(source_dir)
    texts = []  # list of text samples
    label_id = 0
    for name in sorted(os.listdir(news_dir)):
        path = os.path.join(news_dir, name)
        label_id += 1
        if os.path.isdir(path):
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    if sys.version_info < (3,):
                        f = open(fpath)
                    else:
                        f = open(fpath, encoding='latin-1')
                    content = f.read()
                    texts.append((content, label_id))
                    f.close()

    print('Found %s texts.' % len(texts))
    return texts


def get_glove_w2v(source_dir="./data/news20/", dim=100):
    """
    Parse or download the pre-trained glove word2vec if source_dir is empty.

    :param source_dir: The directory storing the pre-trained word2vec
    :param dim: The dimension of a vector
    :return: A dict mapping from word to vector
    """
    w2v_dir = download_glove_w2v(source_dir)
    w2v_path = os.path.join(w2v_dir, "glove.6B.%sd.txt" % dim)
    if sys.version_info < (3,):
        w2v_f = open(w2v_path)
    else:
        w2v_f = open(w2v_path, encoding='latin-1')
    pre_w2v = {}
    for line in w2v_f.readlines():
        items = line.split(" ")
        pre_w2v[items[0]] = [float(i) for i in items[1:]]
    w2v_f.close()
    return pre_w2v


if __name__ == "__main__":
    get_news20("./data/news20/")
    get_glove_w2v("./data/news20/")
