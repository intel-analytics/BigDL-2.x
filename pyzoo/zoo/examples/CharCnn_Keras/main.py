import tensorflow as tf
import json
import numpy as np
import os
import sys
import re
import csv
import itertools
import numpy as np
import re

from models.char_cnn_zhang import CharCNNZhang
from models.char_cnn_kim import CharCNNKim
from pyspark import SparkConf, SparkContext

tf.flags.DEFINE_string("model", "char_cnn_zhang",
                       "Specifies which model to use: char_cnn_zhang or char_cnn_kim")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

dicts = {}
alphabet = "abcdefghijklmnopqrstuvwxyz"
for idx, char in enumerate(alphabet):
    dicts[char] = idx + 1


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


def to_vec(token, w2v_bc, embedding_dim):
    if token in w2v_bc:
        return w2v_bc[token]
    else:
        return pad([], 0, embedding_dim)


def to_sample(vectors, embedding_dim):
    flatten_features = list(itertools.chain(*vectors))
    features = np.array(flatten_features, dtype='float').reshape(
        [500, embedding_dim])
    return features


def get_news20(base_dir="./data/news20"):
    news20_dir = base_dir + "/20news-18828/"
    data = []
    label_id = 0
    for category in sorted(os.listdir(news20_dir)):
        category_dir = os.path.join(news20_dir, category)
        label_id += 1
        if os.path.isdir(category_dir):
            for text_file in sorted(os.listdir(category_dir)):
                if text_file.isdigit():
                    text_file_path = os.path.join(category_dir, text_file)
                    if sys.version_info < (3,):
                        f = open(text_file_path)
                    else:
                        f = open(text_file_path, encoding='latin-1')
                    content = f.read()
                    data.append((content, label_id))
                    f.close()
    return data


def get_glove(base_dir="./data/news20", dim=100):
    glove_dir = base_dir + "/glove.6B"
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


def str_to_indexes(s):
    max_length = min(len(s), 1000)
    str2idx = np.zeros(1000, dtype='int64')
    for i in range(0, max_length):
        a = s[i]
        if a in dict:
            str2idx[i - 1] = dicts[a]
    return str2idx


def get_all_data(data):
    data_size = len(data)
    start_index = 0
    end_index = data_size
    batch_texts = data[start_index:end_index]
    batch_indices = []
    for s in batch_texts:
        batch_indices.append(str_to_indexes(s))
    return np.asarray(batch_indices, dtype='int64')


if __name__ == "__main__":
    # Load configurations
    config = json.load(open("config.json"))

    Conf = SparkConf().setMaster("local").setAppName("My App")
    sc = SparkContext(conf=Conf)

    texts = get_news20(base_dir="/tmp/text_data")
    text_data_rdd = sc.parallelize(texts, 4)

    word_meta = analyze_texts(text_data_rdd)
    word_meta = dict(word_meta[10: 5000])
    word_mata_broadcast = sc.broadcast(word_meta)

    word2vec = get_glove(base_dir="/tmp/text_data", dim=200)
    filtered_word2vec = dict((w, v) for w, v in word2vec.items() if w in word_meta)
    filtered_word2vec_broadcast = sc.broadcast(filtered_word2vec)

    tokens_rdd = text_data_rdd.map(lambda text_label:
                                   ([w for w in text_to_words(text_label[0]) if
                                     w in word_mata_broadcast.value], text_label[1]))
    padded_tokens_rdd = tokens_rdd.map(lambda tokens_label:
                                       (pad(tokens_label[0], "##", 500), tokens_label[1]))
    train_rdd_pre, val_rdd_pre = padded_tokens_rdd.randomSplit([0.80, 0.20])

    one_hot = np.eye(20, dtype='int64')

    # train word
    train_vector_rdd = train_rdd_pre.map(lambda tokens_label:
                                         ([to_vec(w, filtered_word2vec_broadcast.value, 200)
                                           for w in tokens_label[0]]))
    training_word = train_vector_rdd.map(lambda vectors_label: to_sample(vectors_label, 200)).collect()

    # train label
    train_label = train_rdd_pre.map(lambda p: p[1]).collect()
    classes_1 = []
    for c in train_label:
        c = int(c) - 1
        classes_1.append(one_hot[c])
    training_label = np.array(classes_1)

    # val word
    val_vector_rdd = val_rdd_pre.map(lambda tokens_label:
                                     ([to_vec(w, filtered_word2vec_broadcast.value, 200)
                                       for w in tokens_label[0]]))
    validation_word = val_vector_rdd.map(lambda vectors_label: to_sample(vectors_label, 200)).collect()

    # val label
    val_label = val_rdd_pre.map(lambda p: p[1]).collect()
    classes_2 = []
    for d in val_label:
        d = int(d) - 1
        classes_2.append(one_hot[d])
    validation_label = np.array(classes_2)

    # train char
    train_char = train_rdd_pre.map(lambda p: ''.join(p[0])).collect()
    training_char = get_all_data(train_char)

    # val char
    val_char = val_rdd_pre.map(lambda p: ''.join(p[0])).collect()
    validation_char = get_all_data(val_char)

    # Load model configurations and build model
    if FLAGS.model == "kim":
        model = CharCNNKim(input_size=config["data"]["input_size"],
                           alphabet_size=config["data"]["alphabet_size"],
                           embedding_size=config["model"]["embedding_size"],
                           conv_layers=config["model"]["conv_layers"],
                           fully_connected_layers=config["model"]["fully_connected_layers"],
                           num_of_classes=config["data"]["num_of_classes"],
                           dropout_p=config["model"]["dropout_p"],
                           optimizer=config["model"]["optimizer"],
                           loss=config["model"]["loss"])
    else:
        model = CharCNNZhang(input_size=config["data"]["input_size"],
                             alphabet_size=config["data"]["alphabet_size"],
                             embedding_size=config["char_cnn_zhang"]["embedding_size"],
                             conv_layers=config["char_cnn_zhang"]["conv_layers"],
                             fully_connected_layers=config["char_cnn_zhang"]["fully_connected_layers"],
                             num_of_classes=config["data"]["num_of_classes"],
                             threshold=config["char_cnn_zhang"]["threshold"],
                             dropout_p=config["char_cnn_zhang"]["dropout_p"],
                             optimizer=config["char_cnn_zhang"]["optimizer"],
                             loss=config["char_cnn_zhang"]["loss"])
    # Train model
    model.train(training_inputs=[training_char, training_word],
                training_labels=training_label,
                validation_inputs=[validation_char, validation_word],
                validation_labels=validation_label,
                epochs=config["training"]["epochs"],
                batch_size=config["training"]["batch_size"],
                checkpoint_every=config["training"]["checkpoint_every"])

    # model.test(testing_inputs=validation_inputs, testing_labels=validation_labels,batch_size=64)

