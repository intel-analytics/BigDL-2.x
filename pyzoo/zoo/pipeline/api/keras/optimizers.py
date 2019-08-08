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
from bigdl.optim.optimizer import OptimMethod, Default, Optimizer, MaxEpoch
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


class SparseAdagrad(OptimMethod, ZooKerasCreator):
    """
    >>> adam = SparseAdagrad()
    creating: createZooKerasSparseAdagrad
    """
    def __init__(self,
                 lr=1e-3,
                 lr_decay=0.01,
                 bigdl_type="float"):
        """
        :param lr learning rate
        :param lr_decay weight decay
        """

        # explicitly reimplement the constructor since:
        # 1. This class need to be a subclass of OptimMethod
        # 2. The constructor of OptimMethod invokes JavaValue.jvm_class_constructor() directly
        #    and does not take the polymorphism.
        self.value = callBigDlFunc(
            bigdl_type, ZooKerasCreator.jvm_class_constructor(self),
            lr,
            lr_decay)
        self.bigdl_type = bigdl_type


class IndexedSlicesAdam(OptimMethod):
    """
    An implementation of Adam http://arxiv.org/pdf/1412.6980.pdf
    :param learningrate learning rate
    :param learningrate_decay learning rate decay
    :param beta1 first moment coefficient
    :param beta2 second moment coefficient
    :param epsilon for numerical stability
    >>> adam = IndexedSlicesAdam()
    creating: createAdam
    """

    def __init__(self,
                 learningrate=1e-3,
                 learningrate_decay=0.0,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 bigdl_type="float"):
        super(IndexedSlicesAdam, self).__init__(None, bigdl_type, learningrate, learningrate_decay,
                                   beta1, beta2, epsilon)

# class IndexedSlicesAdam(OptimMethod, ZooKerasCreator):
#     """
#     >>> adam = IndexedSlicesAdam()
#     creating: createZooKerasIndexedSlicesAdam
#     creating: createDefault
#     """
#     def __init__(self,
#                  lr=1e-3,
#                  beta_1=0.9,
#                  beta_2=0.999,
#                  epsilon=1e-8,
#                  decay=0.0,
#                  schedule=None,
#                  bigdl_type="float"):
#         """
#         :param lr learning rate
#         :param beta_1 first moment coefficient
#         :param beta_2 second moment coefficient
#         :param epsilon for numerical stability
#         :param decay learning rate decay
#         :param schedule learning rate schedule, e.g. Warmup or Poly from BigDL
#         """
#
#         self.value = callBigDlFunc(
#             bigdl_type, ZooKerasCreator.jvm_class_constructor(self),
#             lr,
#             beta_1,
#             beta_2,
#             epsilon,
#             decay,
#             schedule if (schedule) else Default())
#         self.bigdl_type = bigdl_type

class ZooOptimizer(Optimizer):
    @staticmethod
    def create(model,
               training_set,
               criterion,
               end_trigger=None,
               batch_size=32,
               optim_method=None,
               cores=None,
               sparse_optim_method=None,
               bigdl_type="float"):
        """
        Create an optimizer.
        Depend on the input type, the returning optimizer can be a local optimizer \
        or a distributed optimizer.

        :param model: the neural net model
        :param training_set: (features, label) for local mode. RDD[Sample] for distributed mode.
        :param criterion: the loss function
        :param optim_method: the algorithm to use for optimization,
           e.g. SGD, Adagrad, etc. If optim_method is None, the default algorithm is SGD.
        :param end_trigger: when to end the optimization. default value is MapEpoch(1)
        :param batch_size: training batch size
        :param cores: This is for local optimizer only and use total physical cores as the default value
        """
        if not end_trigger:
            end_trigger = MaxEpoch(1)
        if not optim_method:
            optim_method = SGD()
        if isinstance(training_set, RDD):
            return InternalDistriOptimizer(model=model,
                                   training_rdd=training_set,
                                   criterion=criterion,
                                   end_trigger=end_trigger,
                                   batch_size=batch_size,
                                   optim_method=optim_method,
                                   sparse_optim_method=sparse_optim_method,
                                   bigdl_type=bigdl_type)
        elif (isinstance(training_set, tuple) and len(training_set) == 2) or isinstance(training_set, DataSet):
            Optimizer.create(model, training_set, criterion, end_trigger, batch_size, optim_method, cores, bigdl_type)
        else:
            raise Exception("Not supported training set: %s" % type(training_set))


class InternalDistriOptimizer(ZooOptimizer):
    def __init__(self,
                 model,
                 training_rdd,
                 criterion,
                 end_trigger,
                 batch_size,
                 optim_method=None,
                 sparse_optim_method=None,
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
            JavaValue.__init__(self, None, bigdl_type, model.value,
                               training_rdd, criterion,
                               optim_methods, end_trigger, batch_size, sparse_optim_method)
