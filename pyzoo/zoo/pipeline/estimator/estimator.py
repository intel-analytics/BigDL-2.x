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

from bigdl.util.common import JavaValue, callBigDlFunc


class Estimator(JavaValue):
    def __init__(self, model, optim_methods=None, model_dir=None, jvalue=None, bigdl_type="float"):
        self.bigdl_type = bigdl_type
        self.value = jvalue if jvalue else callBigDlFunc(
            bigdl_type, self.jvm_class_constructor(), model, optim_methods, model_dir)

    def clear_gradient_clipping(self):
        callBigDlFunc(self.bigdl_type, "clearGradientClipping")

    def set_constant_gradient_clipping(self, min, max):
        callBigDlFunc(self.bigdl_type, "setConstantGradientClipping", self.value, min, max)

    def set_l2_norm_gradient_clipping(self, clip_norm):
        callBigDlFunc(self.bigdl_type, "setGradientClippingByL2Norm", self.value, clip_norm)

    def train(self, train_set, criterion, end_trigger=None, checkpoint_trigger=None,
              validation_set=None, validation_method=None, batch_size=32):
        callBigDlFunc(self.bigdl_type, "estimatorTrain", self.value, train_set,
                      criterion, end_trigger, checkpoint_trigger, validation_set,
                      validation_method, batch_size)

    def train_imagefeature(self, train_set, criterion, end_trigger=None, checkpoint_trigger=None,
                           validation_set=None, validation_method=None, batch_size=32):
        callBigDlFunc(self.bigdl_type, "estimatorTrainImageFeature", self.value, train_set,
                      criterion, end_trigger, checkpoint_trigger, validation_set,
                      validation_method, batch_size)

    def evaluate(self, validation_set, validation_method, batch_size=32):
        callBigDlFunc(self.bigdl_type, "estimatorEvaluate", self.value,
                      validation_set, validation_method, batch_size)
