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

import tarfile
import zipfile
from bigdl.dataset import base
from bigdl.util.common import *

NEWS20_URL = 'http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz'
GLOVE_URL = 'http://nlp.stanford.edu/data/glove.6B.zip'


def get_news20(base_dir="./data/20news-18828"):
    """
    Parse 20 Newsgroup dataset and return a list of (tokens, label).
    The dataset will be downloaded automatically if not found in the target base_dir.
    """
    news20_dir = base_dir + "/20news-18828/"
    if not os.path.isdir(news20_dir):
        download_news20(base_dir)
    texts = []
    label_id = 0
    for category in sorted(os.listdir(news20_dir)):
        category_dir = os.path.join(news20_dir, category)
        if os.path.isdir(category_dir):
            for text_file in sorted(os.listdir(category_dir)):
                if text_file.isdigit():
                    text_file_path = os.path.join(category_dir, text_file)
                    if sys.version_info < (3,):
                        f = open(text_file_path)
                    else:
                        f = open(text_file_path, encoding='latin-1')
                    content = f.read()
                    texts.append((content, label_id))
                    f.close()
        label_id += 1
    class_num = label_id
    print('Found %s texts.' % len(texts))
    return texts, class_num


def get_glove(base_dir="./data/glove.6B", dim=100):
    """
    Parse the pre-trained glove6B word2vec and return a dict mapping from word to vector,
    given the dim of a vector.
    The word embeddings will be downloaded automatically if not found in the target base_dir.
    """
    glove_dir = base_dir + "/glove.6B"
    if not os.path.isdir(glove_dir):
        download_glove(base_dir)
    glove_path = os.path.join(glove_dir, "glove.6B.%sd.txt" % dim)
    if sys.version_info < (3,):
        w2v_f = open(glove_path)
    else:
        w2v_f = open(glove_path, encoding='latin-1')
    pre_w2v = {}
    for line in w2v_f.readlines():
        items = line.split(" ")
        pre_w2v[items[0]] = [float(i) for i in items[1:]]
    w2v_f.close()
    return pre_w2v


def download_news20(dest_dir):
    news20 = "20news-18828.tar.gz"
    news20_path = base.maybe_download(news20, dest_dir, NEWS20_URL)
    tar = tarfile.open(news20_path, "r:gz")
    news20_dir = os.path.join(dest_dir, "20news-18828")
    if not os.path.exists(news20_dir):
        print("Extracting %s to %s" % (news20_path, news20_dir))
        tar.extractall(dest_dir)
        tar.close()


def download_glove(dest_dir):
    glove = "glove.6B.zip"
    glove_path = base.maybe_download(glove, dest_dir, GLOVE_URL)
    zip_ref = zipfile.ZipFile(glove_path, 'r')
    glove_dir = os.path.join(dest_dir, "glove.6B")
    if not os.path.exists(glove_dir):
        print("Extracting %s to %s" % (glove_path, glove_dir))
        zip_ref.extractall(glove_dir)
        zip_ref.close()
