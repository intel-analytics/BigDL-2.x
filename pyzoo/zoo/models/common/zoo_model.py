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

from bigdl.nn.layer import Container, Layer
from bigdl.util.common import *
from zoo.pipeline.api.keras.engine.topology import KerasNet
from zoo.common.utils import callZooFunc

if sys.version >= '3':
    long = int
    unicode = str


class ZooModelCreator(JavaValue):
    def jvm_class_constructor(self):
        name = "createZoo" + self.__class__.__name__
        print("creating: " + name)
        return name


class ZooModel(ZooModelCreator, Container):
    """
    The base class for models in Analytics Zoo.
    """

    def predict_classes(self, x, batch_size=32, zero_based_label=True):
        """
        Predict for classes. By default, label predictions start from 0.

        # Arguments
        x: Prediction data. A Numpy array or RDD of Sample.
        batch_size: Number of samples per batch. Default is 32.
        zero_based_label: Boolean. Whether result labels start from 0.
                          Default is True. If False, result labels start from 1.
        """
        if isinstance(x, np.ndarray):
            data_rdd = to_sample_rdd(x, np.zeros([x.shape[0]]))
        elif isinstance(x, RDD):
            data_rdd = x
        else:
            raise TypeError("Unsupported prediction data type: %s" % type(x))
        return callZooFunc(self.bigdl_type, "zooModelPredictClasses",
                           self.value,
                           data_rdd,
                           batch_size,
                           zero_based_label)

    def save_model(self, path, weight_path=None, over_write=False):
        """
        Save the model to the specified path.

        # Arguments
        path: The path to save the model. Local file system, HDFS and Amazon S3 are supported.
              HDFS path should be like 'hdfs://[host]:[port]/xxx'.
              Amazon S3 path should be like 's3a://bucket/xxx'.
        weight_path: The path to save weights. Default is None.
        over_write: Whether to overwrite the file if it already exists. Default is False.
        """
        callZooFunc(self.bigdl_type, "saveZooModel",
                    self.value, path, weight_path, over_write)

    def summary(self):
        """
        Print out the summary of the model.
        """
        callZooFunc(self.bigdl_type, "zooModelSummary",
                    self.value)

    def set_evaluate_status(self):
        """
        Set the model to be in evaluate status, i.e. remove the effect of Dropout, etc.
        """
        callZooFunc(self.bigdl_type, "zooModelSetEvaluateStatus",
                    self.value)
        return self

    @staticmethod
    def _do_load(jmodel, bigdl_type="float"):
        model = Layer(jvalue=jmodel, bigdl_type=bigdl_type)
        model.value = jmodel
        return model


class KerasZooModel(ZooModel):
    """
    The base class for Keras style models in Analytics Zoo.
    """

    # For the following method, please see documentation of KerasNet for details
    def compile(self, optimizer, loss, metrics=None):
        self.model.compile(optimizer, loss, metrics)

    def fit(self, x, y=None, batch_size=32, nb_epoch=10,
            validation_split=0, validation_data=None, distributed=True):
        self.model.fit(x, y, batch_size, nb_epoch, validation_split, validation_data, distributed)

    def set_checkpoint(self, path, over_write=True):
        self.model.set_checkpoint(path, over_write)

    def set_tensorboard(self, log_dir, app_name):
        self.model.set_tensorboard(log_dir, app_name)

    def get_train_summary(self, tag=None):
        return self.model.get_train_summary(tag)

    def get_validation_summary(self, tag=None):
        return self.model.get_validation_summary(tag)

    def clear_gradient_clipping(self):
        self.model.clear_gradient_clipping()

    def set_constant_gradient_clipping(self, min, max):
        self.model.set_constant_gradient_clipping(min, max)

    def set_gradient_clipping_by_l2_norm(self, clip_norm):
        self.model.set_gradient_clipping_by_l2_norm(clip_norm)

    def set_evaluate_status(self):
        return self.model.set_evaluate_status()

    def evaluate(self, x, y=None, batch_size=32):
        return self.model.evaluate(x, y, batch_size)

    def predict(self, x, batch_per_thread=4, distributed=True):
        return self.model.predict(x, batch_per_thread, distributed)

    def predict_classes(self, x, batch_per_thread=4, distributed=True):
        return self.model.predict_classes(x, batch_per_thread, distributed)

    @staticmethod
    def _do_load(jmodel, bigdl_type="float"):
        model = ZooModel._do_load(jmodel, bigdl_type)
        labor_model = callZooFunc(bigdl_type, "getModule", jmodel)
        model.model = KerasNet(labor_model)
        return model
