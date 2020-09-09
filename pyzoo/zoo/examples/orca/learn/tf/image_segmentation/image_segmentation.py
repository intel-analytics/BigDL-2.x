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
import argparse
import os
import zipfile
import pandas as pd

import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import numpy as np

from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca.learn.tf.estimator import Estimator


def load_data_from_zip(file_path, file):
    with zipfile.ZipFile(os.path.join(file_path, file), "r") as zip_ref:
        unzipped_file = zip_ref.namelist()[0]
        zip_ref.extractall(file_path)


def load_data(file_path):
    load_data_from_zip(file_path, 'train.zip')
    load_data_from_zip(file_path, 'train_masks.zip')
    load_data_from_zip(file_path, 'train_masks.csv.zip')


def main(cluster_mode, max_epoch, file_path):
    if args.cluster_mode == "local":
        sc = init_orca_context(cluster_mode="local", cores=4)
    elif args.cluster_mode == "yarn":
        init_orca_context(cluster_mode="yarn-client", num_nodes=2, cores=2, driver_memory="6g")

    load_data(file_path)
    img_dir = os.path.join(file_path, "train")
    label_dir = os.path.join(file_path, "train_masks")

    # Here we only take the first 1000 files for simplicity
    df_train = pd.read_csv(os.path.join(file_path, 'train_masks.csv'))
    ids_train = df_train['img'].map(lambda s: s.split('.')[0])
    ids_train = ids_train[:1000]

    x_train_filenames = []
    y_train_filenames = []
    for img_id in ids_train:
        x_train_filenames.append(os.path.join(img_dir, "{}.jpg".format(img_id)))
        y_train_filenames.append(os.path.join(label_dir, "{}_mask.gif".format(img_id)))

    x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = \
        train_test_split(x_train_filenames, y_train_filenames, test_size=0.2, random_state=42)

    from scipy import misc

    def load_and_process_image(file_path):
        array = mpimg.imread(file_path)
        result = np.array(array)
        result = misc.imresize(result, size=(128, 128))
        result = result.astype(float)
        result /= 255.0
        return result

    def load_and_process_image_label(file_path):
        array = mpimg.imread(file_path)
        result = np.array(array)
        result = misc.imresize(result, size=(128, 128))
        result = np.expand_dims(result[:, :, 1], axis=-1)
        result = result.astype(float)
        result /= 255.0
        return result

    train_images = sc.parallelize(x_train_filenames).map(
        lambda filepath: load_and_process_image(filepath))
    train_label_images = sc.parallelize(y_train_filenames).map(
        lambda filepath: load_and_process_image_label(filepath))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The mode for the Spark cluster. local or yarn.')
    parser.add_argument('--file_path', type=str, default="/tmp/carvana/",
                        help="The path to carvana train.zip, train_mask.zip and train_mask.csv.zip")

    args = parser.parse_args()
    main(args.cluster_mode, 5, args.file_path)
    stop_orca_context()
