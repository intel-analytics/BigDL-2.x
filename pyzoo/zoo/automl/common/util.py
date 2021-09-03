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

import os
import json

from zoo.chronos.feature.utils import restore

IDENTIFIER_LEN = 27


def process(command, fail_fast=False, timeout=120):
    import subprocess
    pro = subprocess.Popen(
        command,
        shell=True,
        cwd=None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid)
    out, err = pro.communicate(timeout=timeout)
    out = out.decode("utf-8")
    err = err.decode("utf-8")
    print(out)
    print(err)
    errorcode = pro.returncode
    if errorcode != 0:
        if fail_fast:
            raise Exception(err)
        print(err)
    else:
        print(out)


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
