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
import shutil
import tempfile
import zipfile

import numpy as np
import os
import json

IDENTIFIER_LEN = 27


class NumpyEncoder(json.JSONEncoder):
    """
    convert numpy array to list for JSON serialize
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_config(file_path, config, replace=False):
    """
    :param file_path: the file path of config to be saved.
    :param config: dict. The config to be saved
    :param replace: whether to replace if the config file already existed.
    :return:
    """
    if os.path.isfile(file_path) and not replace:
        with open(file_path, "r") as input_file:
            old_config = json.load(input_file)
        old_config.update(config)
        config = old_config.copy()

    file_dirname = os.path.dirname(os.path.abspath(file_path))
    if file_dirname and not os.path.exists(file_dirname):
        os.makedirs(file_dirname)

    with open(file_path, "w") as output_file:
        json.dump(config, output_file, cls=NumpyEncoder)


def load_config(file_path):
    with open(file_path, "r") as input_file:
        data = json.load(input_file)
    return data


def save(file_path, feature_transformers=None, model=None, config=None):
    if not os.path.isdir(file_path):
        os.mkdir(file_path)
    config_path = os.path.join(file_path, "config.json")
    model_path = os.path.join(file_path, "weights_tune.h5")
    if feature_transformers is not None:
        feature_transformers.save(config_path, replace=True)
    if model is not None:
        model.save(model_path, config_path)
    if config is not None:
        save_config(config_path, config)


def process(cmd):
    import subprocess
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    # process.wait()
    outs, errors = proc.communicate()
    # if outs:
    #     print("hdfs std out:", outs)
    if errors:
        print("hdfs errors:", errors)
    return outs, errors


def get_remote_list(dir_in):
    args = "hdfs dfs -ls " + dir_in + " | awk '{print $8}'"
    s_output, _ = process(args)

    all_dart_dirs = s_output.split()
    names = []
    for filename in all_dart_dirs:
        filename = filename.decode()
        name_list = filename.split('/')
        names.append(name_list[-1])
    # print(names)
    return names


def put_ckpt_hdfs(remote_dir, ckpt_name):
    """
    Upload checkpoint file with name of ckpt_name to the hdfs directory
    {remote_dir}/{ray_checkpoint_dir}[-IDENTIFIER_LEN:].

    Note that ray_checkpoint_dir is like train_func_0_{config}_{time}_{tmp},
    with a max identifier length of 130. However, if there is a list("[]") in config and is
    truncated with half "[" remained, then the path name can't be identified by hadoop command.
    Therefore we use the last IDENTIFIER_LEN=27 of ray_checkpoint_dir as remote_ckpt_basename,
    with a format of {time}_{tmp}, in order to avoid misinterpretation.
    """

    local_ckpt_dir = os.path.abspath(".")
    remote_ckpt_basename = os.path.basename(local_ckpt_dir)[-IDENTIFIER_LEN:]
    remote_ckpt_dir = os.path.join(remote_dir, remote_ckpt_basename)
    if remote_ckpt_basename not in get_remote_list(remote_dir):
        cmd = f"hadoop fs -mkdir {remote_ckpt_dir};" \
              f" hadoop fs -put -f {ckpt_name} {remote_ckpt_dir}"
    else:
        cmd = f"hadoop fs -put -f {ckpt_name} {remote_ckpt_dir}"
    process(cmd)


def get_ckpt_hdfs(remote_dir, local_ckpt):
    """
    Get checkpoint file from hdfs as local_ckpt
    Remote checkpoint dir is {remote_dir}/{ray_checkpoint_dir}[-IDENTIFIER_LEN:].
    """
    ckpt_name = os.path.basename(local_ckpt)
    local_ckpt_dir = os.path.dirname(local_ckpt)
    remote_ckpt_basename = os.path.basename(local_ckpt_dir)[-IDENTIFIER_LEN:]
    remote_ckpt = os.path.join(remote_dir, remote_ckpt_basename, ckpt_name)

    cmd = "hadoop fs -get {} {}".format(remote_ckpt, local_ckpt_dir)
    process(cmd)


def restore(file, feature_transformers=None, model=None, config=None):
    model_path = os.path.join(file, "weights_tune.h5")
    config_path = os.path.join(file, "config.json")
    local_config = load_config(config_path)
    if config is not None:
        all_config = config.copy()
        all_config.update(local_config)
    else:
        all_config = local_config
    if model:
        model.restore(model_path, **all_config)
    if feature_transformers:
        feature_transformers.restore(**all_config)
    return all_config


def restore_hdfs(model_path, remote_dir, feature_transformers=None, model=None, config=None):
    model_name = os.path.basename(model_path)
    local_best_dirname = os.path.basename(os.path.dirname(model_path))
    remote_model = os.path.join(remote_dir, local_best_dirname[-IDENTIFIER_LEN:], model_name)
    tmp_dir = tempfile.mkdtemp(prefix="automl_save_")
    try:
        cmd = "hadoop fs -get {} {}".format(remote_model, tmp_dir)
        process(cmd)
        with zipfile.ZipFile(os.path.join(tmp_dir, model_name)) as zf:
            zf.extractall(tmp_dir)

        all_config = restore(tmp_dir, feature_transformers, model, config)
    finally:
        shutil.rmtree(tmp_dir)
    return all_config


def convert_bayes_configs(config):
    selected_features = []
    new_config = {}
    for config_name, config_value in config.items():
        if config_name.startswith('bayes_feature'):
            # print(config_name, config_value)
            if config_value >= 0.5:
                feature_name = config_name.replace('bayes_feature_', '')
                selected_features.append(feature_name)
        elif config_name == 'batch_size_log':
            batch_size = int(2 ** config_value)
            new_config['batch_size'] = batch_size
        elif config_name.endswith('float'):
            int_config_name = config_name.replace('_float', '')
            int_config_value = int(config_value)
            new_config[int_config_name] = int_config_value
        else:
            new_config[config_name] = config_value
    if selected_features:
        new_config['selected_features'] = json.dumps(selected_features)
    # print("config after bayes conversion is ", new_config)
    return new_config
