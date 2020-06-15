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


def check_tf_version():
    import os
    import logging
    do_check = True
    if "ANALYTICS_ZOO_TF_CHECK" in os.environ and os.environ["ANALYTICS_ZOO_TF_CHECK"] == "false":
        do_check = False
    if do_check:
        import tensorflow as tf
        v_str = tf.__version__
        major, minor, patch = v_str.split(".")
        if v_str != "1.15.0":
            if int(major) == 1:
                logging.warning("\n######################### WARNING ##########################\n"
                                "\nAnalytics Zoo TFPark has only been tested on TensorFlow 1.15.0,"
                                " but your current TensorFlow installation is {}.".format(v_str) +
                                "\nYou may encounter some version incompatibility issues. "
                                "\n##############################################################")
            else:
                message = "Analytics Zoo TFPark only supports TensorFlow 1.15.0, " + \
                          "but your current TensorFlow installation is {}".format(v_str) + \
                          "\nYou can export ANALYTICS_ZOO_TF_CHECK=false to disable this check."
                raise RuntimeError(message)

check_tf_version()

from .model import KerasModel
from .estimator import TFEstimator
from .tf_optimizer import TFOptimizer
from .tf_dataset import TFDataset
from .zoo_optimizer import ZooOptimizer
from .tf_predictor import TFPredictor
from .tfnet import TFNet
