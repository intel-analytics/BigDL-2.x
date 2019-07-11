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

from bigdl.util.common import *
from bigdl.optim.optimizer import OptimMethod, Default
from zoo.pipeline.api.keras.base import ZooKerasCreator

if sys.version >= '3':
    long = int
    unicode = str


class Adam(OptimMethod, ZooKerasCreator):
    """
    An implementation of Adam with learning rate schedule.
    >>> adam = Adam()
    creating: createZooKerasAdam
    creating: createDefault
    """
    def __init__(self,
                 lr=1e-3,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-8,
                 decay=0.0,
                 schedule=None,
                 bigdl_type="float"):
        """
        :param lr learning rate
        :param beta_1 first moment coefficient
        :param beta_2 second moment coefficient
        :param epsilon for numerical stability
        :param decay learning rate decay
        :param schedule learning rate schedule, e.g. Warmup or Poly from BigDL
        """

        # explicitly reimplement the constructor since:
        # 1. This class need to be a subclass of OptimMethod
        # 2. The constructor of OptimMethod invokes JavaValue.jvm_class_constructor() directly
        #    and does not take the polymorphism.
        self.value = callBigDlFunc(
            bigdl_type, ZooKerasCreator.jvm_class_constructor(self),
            lr,
            beta_1,
            beta_2,
            epsilon,
            decay,
            schedule if (schedule) else Default()
        )
        self.bigdl_type = bigdl_type


class AdamWeightDecay(OptimMethod, ZooKerasCreator):
    """
    >>> adam = AdamWeightDecay()
    creating: createZooKerasAdamWeightDecay
    """
    def __init__(self,
                 lr=1e-3,
                 warmup_portion=-1.0,
                 total=-1,
                 schedule="linear",
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-6,
                 weight_decay=0.01,
                 bigdl_type="float"):
        """
        :param lr learning rate
        :param warmupPortion portion of total for the warmup, -1 means no warmup. Default: -1
        :param total total number of training steps for the learning
         rate schedule, -1 means constant learning rate. Default: -1
        :param schedule schedule to use for the warmup. Default: 'linear'
        :param beta1 first moment coefficient
        :param beta2 second moment coefficient
        :param epsilon for numerical stability
        :param weightDecay weight decay
        """

        # explicitly reimplement the constructor since:
        # 1. This class need to be a subclass of OptimMethod
        # 2. The constructor of OptimMethod invokes JavaValue.jvm_class_constructor() directly
        #    and does not take the polymorphism.
        self.value = callBigDlFunc(
            bigdl_type, ZooKerasCreator.jvm_class_constructor(self),
            lr,
            warmup_portion,
            total,
            schedule,
            beta1,
            beta2,
            epsilon,
            weight_decay)
        self.bigdl_type = bigdl_type


from bigdl.optim.optimizer import DistriOptimizer as BDistriOptimizer, SGD


class DistriOptimizer(BDistriOptimizer):
    def __init__(self,
                 model,
                 training_rdd,
                 criterion,
                 end_trigger=None,
                 batch_size=32,
                 optim_method=None,
                 bigdl_type="float"):
        """
        Create an optimizer.


        :param model: the neural net model
        :param training_data: the training dataset
        :param criterion: the loss function
        :param optim_method: the algorithm to use for optimization,
           e.g. SGD, Adagrad, etc. If optim_method is None, the default algorithm is SGD.
        :param end_trigger: when to end the optimization
        :param batch_size: training batch size
        """
        if not optim_method:
            optim_methods = {model.name(): SGD()}
        elif isinstance(optim_method, OptimMethod):
            optim_methods = {model.name(): optim_method}
        elif isinstance(optim_method, JavaObject):
            optim_methods = {model.name(): OptimMethod(optim_method, bigdl_type)}
        else:
            optim_methods = optim_method
        if isinstance(training_rdd, RDD):
            self.bigdl_type = bigdl_type
            self.value = callBigDlFunc(self.bigdl_type, "createDistriOptimizerFromRDD",
                                       model.value, training_rdd, criterion,
                                       optim_methods, end_trigger, batch_size)

    def set_validation(self, batch_size, val_rdd, trigger, val_method=None):
        """
        Configure validation settings.


        :param batch_size: validation batch size
        :param val_rdd: validation dataset
        :param trigger: validation interval
        :param val_method: the ValidationMethod to use,e.g. "Top1Accuracy", "Top5Accuracy", "Loss"
        """
        callBigDlFunc(self.bigdl_type, "setValidationWithPaddingStrategy", self.value, batch_size,
                      trigger, val_rdd, to_list(val_method))
