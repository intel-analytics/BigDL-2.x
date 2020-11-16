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

from zoo.common.utils import get_file_list


def find_latest_checkpoint(model_dir, model_type="bigdl"):
    import os
    import re
    import datetime
    ckpt_path = None
    latest_version = None
    optim_prefix = None
    optim_regex = None
    if model_type == "bigdl":
        optim_regex = ".*\.([0-9]+)$"
    elif model_type == "pytorch":
        optim_regex = "TorchModel[0-9a-z]{8}\.([0-9]+)$"
    elif model_type == "tf":
        optim_regex = "TFParkTraining\.([0-9]+)$"
    else:
        ValueError("Only bigdl, pytorch and tf are supported for now.")

    file_list = get_file_list(model_dir, recursive=True)
    optim_dict = {}
    pattern_re = re.compile('(.*)(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})(.*)optimMethod-'
                            + optim_regex)
    for file_path in file_list:
        matched = pattern_re.match(file_path)
        if matched is not None:
            try:
                # check if dir name is date time
                timestamp = matched.group(2)
                datetime.datetime.strptime(timestamp, '%Y-%m-%d_%H-%M-%S')
                if timestamp in optim_dict:
                    optim_dict[timestamp].append((matched.group(4),
                                                  os.path.dirname(file_path),
                                                  os.path.basename(file_path).split('.')[0]))
                else:
                    optim_dict[timestamp] = [(matched.group(4),
                                              os.path.dirname(file_path),
                                              os.path.basename(file_path).split('.')[0])]
            except:
                continue
    if optim_dict:
        latest_timestamp = max(optim_dict)
        latest_version, ckpt_path, optim_prefix = max(optim_dict[latest_timestamp],
                                                   key=lambda version_path: version_path[0])

    return ckpt_path, optim_prefix, latest_version
