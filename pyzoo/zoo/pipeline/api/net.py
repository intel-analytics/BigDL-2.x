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

import sys
import tempfile
import warnings

import six
import os
import json
import numpy as np
from py4j.protocol import Py4JJavaError
from pyspark import RDD

from bigdl.nn.criterion import Criterion
from bigdl.nn.layer import Model as BModel
from bigdl.nn.layer import Layer
from bigdl.util.common import to_list, callBigDlFunc, \
    JavaValue, get_node_and_core_number
from zoo.common import Sample, JTensor
from zoo.common.nncontext import getOrCreateSparkContext
from zoo.feature.image import ImageSet
from zoo.pipeline.api.keras.engine.topology import ZooKerasLayer, KerasNet, to_bigdl_metric
from bigdl.optim.optimizer import EveryEpoch, MaxEpoch, Optimizer
from zoo.util import nest

if sys.version >= '3':
    long = int
    unicode = str


class GraphNet(BModel):
    def __init__(self, input, output, jvalue=None, bigdl_type="float", **kwargs):
        super(BModel, self).__init__(jvalue,
                                     to_list(input),
                                     to_list(output),
                                     bigdl_type,
                                     **kwargs)

    def flattened_layers(self, include_container=False):
        jlayers = callBigDlFunc(self.bigdl_type, "getFlattenSubModules", self, include_container)
        layers = [Layer.of(jlayer) for jlayer in jlayers]
        return layers

    @property
    def layers(self):
        jlayers = callBigDlFunc(self.bigdl_type, "getSubModules", self)
        layers = [Layer.of(jlayer) for jlayer in jlayers]
        return layers

    @staticmethod
    def from_jvalue(jvalue, bigdl_type="float"):
        """
        Create a Python Model base on the given java value

        :param jvalue: Java object create by Py4j
        :return: A Python Model
        """
        model = GraphNet([], [], jvalue=jvalue, bigdl_type=bigdl_type)
        model.value = jvalue
        return model

    def new_graph(self, outputs):
        """
        Specify a list of nodes as output and return a new graph using the existing nodes

        :param outputs: A list of nodes specified
        :return: A graph model
        """
        value = callBigDlFunc(self.bigdl_type, "newGraph", self.value, outputs)
        return self.from_jvalue(value, self.bigdl_type)

    def freeze_up_to(self, names):
        """
        Freeze the model from the bottom up to the layers specified by names (inclusive).
        This is useful for finetuning a model

        :param names: A list of module names to be Freezed
        :return: current graph model
        """
        callBigDlFunc(self.bigdl_type, "freezeUpTo", self.value, names)

    def unfreeze(self, names=None):
        """
        "unfreeze" module, i.e. make the module parameters(weight/bias, if exists)
        to be trained(updated) in training process.
        If 'names' is a non-empty list, unfreeze layers that match given names

        :param names: list of module names to be unFreezed. Default is None.
        :return: current graph model
        """
        callBigDlFunc(self.bigdl_type, "unFreeze", self.value, names)

    def to_keras(self):
        value = callBigDlFunc(self.bigdl_type, "netToKeras", self.value)
        return ZooKerasLayer.of(value, self.bigdl_type)


class Net:
    @staticmethod
    def load_bigdl(model_path, weight_path=None, bigdl_type="float"):
        """
        Load a pre-trained BigDL model.

        :param model_path: The path to the pre-trained model.
        :param weight_path: The path to the weights of the pre-trained model. Default is None.
        :return: A pre-trained model.
        """
        jmodel = callBigDlFunc(bigdl_type, "netLoadBigDL", model_path, weight_path)
        return GraphNet.from_jvalue(jmodel)

    @staticmethod
    def load(model_path, weight_path=None, bigdl_type="float"):
        """
        Load an existing Analytics Zoo model defined in Keras-style(with weights).

        :param model_path: The path to load the saved model.
                          Local file system, HDFS and Amazon S3 are supported.
                          HDFS path should be like 'hdfs://[host]:[port]/xxx'.
                          Amazon S3 path should be like 's3a://bucket/xxx'.
        :param weight_path: The path for pre-trained weights if any. Default is None.
        :return: An Analytics Zoo model.
        """
        jmodel = callBigDlFunc(bigdl_type, "netLoad", model_path, weight_path)
        return KerasNet.of(jmodel, bigdl_type)

    @staticmethod
    def load_torch(path, bigdl_type="float"):
        """
        Load a pre-trained Torch model.

        :param path: The path containing the pre-trained model.
        :return: A pre-trained model.
        """
        jmodel = callBigDlFunc(bigdl_type, "netLoadTorch", path)
        return GraphNet.from_jvalue(jmodel, bigdl_type)

    @staticmethod
    def load_tf(path, inputs=None, outputs=None, byte_order="little_endian",
                bin_file=None, bigdl_type="float"):
        """
        Load a pre-trained TensorFlow model.
        :param path: The path containing the pre-trained model.
                     OR alternatively, the exported folder path from `export_tf`.
                     In this case, path should contain 'frozen_inference_graph.pb' and
                     'graph_meta.json'. You don't need to specify inputs and outputs.
        :param inputs: The input nodes of this graph.
        :param outputs: The output nodes of this graph.
        :param byte_order: Byte_order of the file, `little_endian` or `big_endian`.
        :param bin_file: Optional bin file produced by bigdl dump_model util function
                         to store the weights. Default is None.
        :return: A pre-trained model.
        """
        if not inputs and not outputs:  # load_tf from exported folder
            if not os.path.isdir(path):
                raise ValueError("load_tf from exported folder requires path to be a folder")
            jmodel = callBigDlFunc(bigdl_type, "netLoadTF", path)
        else:
            jmodel = callBigDlFunc(bigdl_type, "netLoadTF", path, inputs, outputs,
                                   byte_order, bin_file)
        return GraphNet.from_jvalue(jmodel, bigdl_type)

    @staticmethod
    def load_caffe(def_path, model_path, bigdl_type="float"):
        """
        Load a pre-trained Caffe model.

        :param def_path: The path containing the caffe model definition.
        :param model_path: The path containing the pre-trained caffe model.
        :return: A pre-trained model.
        """
        jmodel = callBigDlFunc(bigdl_type, "netLoadCaffe", def_path, model_path)
        return GraphNet.from_jvalue(jmodel, bigdl_type)

    @staticmethod
    def load_keras(json_path=None, hdf5_path=None, by_name=False):
        """
        Load a pre-trained Keras model.

        :param json_path: The json path containing the keras model definition. Default is None.
        :param hdf5_path: The HDF5 path containing the pre-trained keras model weights
                        with or without the model architecture. Default is None.
        :param by_name: by default the architecture should be unchanged.
                        If set as True, only layers with the same name will be loaded.
        :return: A BigDL model.
        """
        return BModel.load_keras(json_path, hdf5_path, by_name)


def to_sample_rdd(x, y, sc, num_slices=None):
    """
    Conver x and y into RDD[Sample]
    :param sc: SparkContext
    :param x: ndarray and the first dimension should be batch
    :param y: ndarray and the first dimension should be batch
    :param numSlices:
    :return:
    """
    x_rdd = sc.parallelize(x, num_slices)
    y_rdd = sc.parallelize(y, num_slices)
    return x_rdd.zip(y_rdd).map(lambda item: Sample.from_ndarray(item[0], item[1]))


class TFNet(Layer):
    def __init__(self, path, input_names=None, output_names=None, bigdl_type="float"):
        if input_names is None and output_names is None:
            super(TFNet, self).__init__(None, bigdl_type,
                                        path)
        else:
            if isinstance(input_names, six.string_types):
                input_names = [input_names]
            if isinstance(output_names, six.string_types):
                output_names = [output_names]
            super(TFNet, self).__init__(None, bigdl_type,
                                        path,
                                        input_names,
                                        output_names)

    @staticmethod
    def check_input(input):
        """
        :param input: ndarray or list of ndarray or JTensor or list of JTensor.
        :return: (list of JTensor, isTable)
        """

        def to_jtensor(i):
            if isinstance(i, np.ndarray):
                return JTensor.from_ndarray(i)
            elif isinstance(i, JTensor):
                return i
            else:
                raise Exception("Error unknown input type %s" % type(i))

        if type(input) is list:
            if len(input) == 0:
                raise Exception('Error when checking: empty input')
            return list(map(lambda i: to_jtensor(i), input)), True
        else:
            return [to_jtensor(input)], False

    def predict(self, x, batch_per_thread=1, distributed=True):
        """
        Use a model to do prediction.
        """
        if isinstance(x, ImageSet):
            results = callBigDlFunc(self.bigdl_type, "zooPredict",
                                    self.value,
                                    x,
                                    batch_per_thread)
            return ImageSet(results)
        if distributed:
            if isinstance(x, np.ndarray):
                data_rdd = to_sample_rdd(x, np.zeros([x.shape[0]]), getOrCreateSparkContext())
            elif isinstance(x, RDD):
                data_rdd = x
            else:
                raise TypeError("Unsupported prediction data type: %s" % type(x))
            results = callBigDlFunc(self.bigdl_type, "zooPredict",
                                    self.value,
                                    data_rdd,
                                    batch_per_thread)
            return results.map(lambda result: Layer.convert_output(result))
        else:
            if isinstance(x, np.ndarray) or isinstance(x, list):
                results = callBigDlFunc(self.bigdl_type, "zooPredict",
                                        self.value,
                                        self._to_jtensors(x),
                                        batch_per_thread)
                return [Layer.convert_output(result) for result in results]
            else:
                raise TypeError("Unsupported prediction data type: %s" % type(x))

    @staticmethod
    def from_export_folder(folder):
        if not os.path.isdir(folder):
            raise ValueError(folder + " does not exist")
        return TFNet(folder)

    @staticmethod
    def from_session(sess, inputs, outputs,
                     generate_backward=False, allow_non_differentiable_input=True):
        from zoo.util.tf import export_tf
        temp = tempfile.mkdtemp()
        try:
            export_tf(sess, temp, inputs, outputs,
                      generate_backward, allow_non_differentiable_input)
            net = TFNet.from_export_folder(temp)
        finally:
            import shutil
            shutil.rmtree(temp)

        return net


def _find_placeholders(grads):
    '''
    find all the tensors that are used for computing grads and has been
    computed during forward
    :param grads:
    :param forward_ops:
    :return:
    '''
    import sys
    is_py2 = sys.version[0] == '2'
    if is_py2:
        import Queue as queue
    else:
        import queue as queue
    queue = queue.Queue()
    for grad in grads:
        queue.put(grad)

    placeholders = set()
    visited = set()
    while not queue.empty():
        tensor = queue.get()
        # this is necessary, because input may not be differentiable
        if tensor is None:
            continue
        else:
            visited.add(tensor.name)
            if tensor.op.type.startswith("Placeholder"):
                placeholders.add(tensor)
                continue
            for input_tensor in tensor.op.inputs:
                # this is necessary because there may be a cycle in the graph such as tf.while_loop
                if input_tensor.name not in visited:
                    queue.put(input_tensor)
    return list(placeholders)


class IdentityCriterion(Criterion):
    def __init__(self):
        super(IdentityCriterion, self).__init__(None, "float")


class TFTrainingHelper(Layer):
    def __init__(self, path, configProto):
        if configProto is not None:
            byte_arr = configProto.SerializeToString()
        else:
            byte_arr = None
        super(TFTrainingHelper, self).__init__(None, "float", path, byte_arr)


class TFValidationMethod(JavaValue):
    def __init__(self, val_method, output_length, target_length):
        JavaValue.__init__(self, None, "float",
                           val_method, output_length, target_length)


class TFOptimizer:
    def __init__(self, loss, optim_method, sess=None, dataset=None, inputs=None,
                 grads=None, variables=None, graph=None,
                 val_outputs=None, val_labels=None, val_method=None, val_split=0.0,
                 tensors_with_value=None, session_config=None):
        '''
        TFOptimizer is used for distributed training of TensorFlow
        on Spark/BigDL.

        :param loss: The loss tensor of the TensorFlow model, should be a scalar
        :param optim_method: the optimization method to be used, such as bigdl.optim.optimizer.Adam
        :param sess: the current TensorFlow Session, if you want to used a pre-trained model, you
        should use the Session to load the pre-trained variables and pass it to TFOptimizer.
        '''

        import tensorflow as tf
        from zoo.util.tf import export_tf

        if dataset is None:
            args = TFOptimizer._get_arguments_from_loss(loss, optim_method, sess,
                                                        val_outputs, val_labels, val_method)
            loss, optim_method, sess, dataset, inputs = args[:5]
            grads, variables, graph, val_outputs, val_labels, val_method = args[5:]

        additional_inputs = []
        additional_values = []
        all_required_inputs = _find_placeholders([loss])
        all_required_inputs_names = [v.name for v in all_required_inputs]
        if tensors_with_value:
            for t, v in tensors_with_value.items():
                if t.name in all_required_inputs_names:
                    additional_inputs.append(t)
                    additional_values.append(v)

        if not isinstance(inputs, list):
            inputs = nest.flatten(inputs)

        self.optim_method = optim_method
        self.sess = sess
        self.dataset = dataset
        self.inputs = inputs + additional_inputs
        self.graph = graph
        self.session_config = session_config

        from zoo.util.tf import process_grad
        grads = [process_grad(grad) for grad in grads]

        if self.dataset.batch_size <= 0:
            raise ValueError("You should set batch_size instead of batch_per_thread for training")

        if val_outputs is not None and val_labels is not None:
            with self.graph.as_default():
                val_labels = [tf.identity(v) for v in val_labels]
            outputs = val_outputs + val_labels + [loss]
        else:
            outputs = [loss]

        self.grads = grads
        self.outputs = outputs

        self.export_dir = tempfile.mkdtemp()
        export_tf(self.sess, self.export_dir,
                  inputs=self.inputs,
                  outputs=self.grads + self.outputs)

        variable_names = [v.name for v in variables]
        grad_names = [g.name for g in grads]
        output_names = [o.name for o in outputs]

        def to_floats(vs):
            return [float(v) for v in vs]

        meta = {
            "input_names": [i.name for i in self.inputs],
            "output_names": output_names,
            "variables": variable_names,
            "grad_variables": grad_names,
            "default_tensor_values": [to_floats(v) for v in additional_values]
        }

        with open(os.path.join(self.export_dir, "training_meta.json"), "w") as f:
            f.write(json.dumps(meta))

        self.variable_placeholders = []
        with self.graph.as_default():
            assigns = []
            for v in variables:
                p = tf.placeholder(dtype=tf.float32, shape=v.shape)
                a = tf.assign(v, p)
                self.variable_placeholders.append(p)
                assigns.append(a)
            assign = tf.group(*assigns)
        self.assign = assign
        try:
            self.training_helper_layer = TFTrainingHelper(self.export_dir, session_config)
        except Py4JJavaError as e:
            if "expects to be colocated with unknown node" in str(e):
                raise Exception("""
If you are using the embedding layer in tf.keras, then this is a
known issue of TensorFlow, see https://github.com/tensorflow/tensorflow/issues/21889.
Please add zoo.util.tf.variable_creator_scope before model construction.
For example:
from zoo.util.tf import variable_creator_scope
with variable_creator_scope():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(1, 1, input_length=1)])
                """)
            else:
                raise e

        data = self.dataset.rdd
        batch_size = self.dataset.batch_size

        def to_sample(t):
            return Sample.from_ndarray(nest.flatten(t), [np.array([0.0])])

        sample_rdd = data.map(to_sample)
        if val_outputs is not None and val_labels is not None:
            if self.dataset.val_rdd is not None:
                val_rdd = self.dataset.val_rdd.map(to_sample)
                val_method = [TFValidationMethod(m, len(val_outputs), len(val_labels))
                              for m in to_list(val_method)]
                training_rdd = sample_rdd

            elif val_split != 0.0:
                training_rdd, val_rdd = sample_rdd.randomSplit([1 - val_split, val_split])
                val_method = [TFValidationMethod(m, len(val_outputs), len(val_labels))
                              for m in to_list(val_method)]
            else:
                raise ValueError("Validation data is not specified. Please set " +
                                 "val rdd in TFDataset, or set val_split larger than zero")

            self.optimizer = Optimizer.create(self.training_helper_layer,
                                              training_rdd,
                                              IdentityCriterion(),
                                              batch_size=batch_size,
                                              optim_method=self.optim_method)
            self.optimizer.set_validation(self.dataset.batch_size,
                                          val_rdd,
                                          EveryEpoch(),
                                          val_method)
        else:
            training_rdd = sample_rdd
            self.optimizer = Optimizer.create(self.training_helper_layer,
                                              training_rdd,
                                              IdentityCriterion(),
                                              batch_size=batch_size,
                                              optim_method=self.optim_method)

    @staticmethod
    def _get_arguments_from_loss(loss, optim_method, session, val_outputs, val_labels, val_method):
        import tensorflow as tf
        if session is None:
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
        else:
            sess = session
        grads_vars = tf.train.GradientDescentOptimizer(0).compute_gradients(loss)
        variables = []
        grads = []
        for (grad, var) in grads_vars:
            variables.append(var)
            grads.append(grad)

        all_required_inputs = _find_placeholders([loss])
        dataset = tf.get_collection(all_required_inputs[0].name)[0]

        inputs = []
        for item in list(dataset._original_tensors):
            if isinstance(item, dict):
                inputs = inputs + list(item.values())
            else:
                inputs.append(item)

        _check_the_same(all_required_inputs, inputs)

        return [loss, optim_method, sess, dataset, inputs,
                grads, variables, loss.graph, val_outputs, val_labels, val_method]

    @classmethod
    def from_loss(cls, loss, optim_method, session=None, val_outputs=None,
                  val_labels=None, val_method=None, val_split=0.0, **kwargs):
        args = TFOptimizer._get_arguments_from_loss(loss, optim_method,
                                                    session, val_outputs,
                                                    val_labels, val_method)

        return cls(*(args + [val_split]), **kwargs)

    @classmethod
    def from_keras(cls, keras_model, dataset, val_spilt=0.0, **kwargs):
        import tensorflow.keras.backend as K
        loss = keras_model.total_loss
        inputs = keras_model.inputs + keras_model.targets

        variables = keras_model._collected_trainable_weights
        keras_optimizer = keras_model.optimizer
        grads = keras_optimizer.get_gradients(loss, variables)

        sess = K.get_session()
        optim_method = TFOptimizer.to_bigdl_optim_method(keras_optimizer)

        if keras_model.metrics and (dataset.val_rdd is not None or val_spilt != 0.0):
            if isinstance(keras_model.metrics, dict):
                raise ValueError(
                    "different metrics for different outputs are not supported right now")

            if dataset.val_rdd is None and val_spilt == 0.0:
                raise ValueError("Validation data is not specified. Please set " +
                                 "val_rdd in TFDataset, or set val_split larger than zero")
            bigdl_val_methods =\
                [to_bigdl_metric(m, keras_model.loss) for m in keras_model.metrics_names]
            val_outputs = keras_model.outputs
            val_labels = keras_model.targets
        else:
            val_outputs = None
            val_labels = None
            bigdl_val_methods = None

        tensor_with_value = {
            K.learning_phase(): [True, False]
        }

        return cls(loss, optim_method, sess, dataset, inputs,
                   grads, variables, loss.graph, val_outputs, val_labels,
                   bigdl_val_methods, val_spilt,
                   tensors_with_value=tensor_with_value, **kwargs)

    @staticmethod
    def to_bigdl_optim_method(koptim_method):
        # koptim_method is always an object
        import tensorflow.keras.backend as K
        import tensorflow.keras.optimizers as koptimizers
        import bigdl.optim.optimizer as boptimizer
        import tensorflow.train as tftrain
        import tensorflow as tf
        from tensorflow.python.keras.optimizers import TFOptimizer

        if isinstance(koptim_method, TFOptimizer):
            koptim_method = koptim_method.optimizer

        if isinstance(koptim_method, koptimizers.Optimizer):
            lr = float(K.eval(koptim_method.lr))
            decay = float(K.eval(koptim_method.decay))
            if isinstance(koptim_method, koptimizers.Adagrad):
                warnings.warn("For Adagrad, we don't support epsilon for now")
                return boptimizer.Adagrad(learningrate=lr,
                                          learningrate_decay=decay)
            elif isinstance(koptim_method, koptimizers.SGD):
                momentum = float(K.eval(koptim_method.momentum))
                return boptimizer.SGD(learningrate=lr,
                                      learningrate_decay=decay,
                                      momentum=momentum,
                                      nesterov=koptim_method.nesterov)
            elif isinstance(koptim_method, koptimizers.Adam):
                beta1 = float(K.eval(koptim_method.beta_1))
                beta2 = float(K.eval(koptim_method.beta_2))
                return boptimizer.Adam(learningrate=lr,
                                       learningrate_decay=decay,
                                       beta1=beta1,
                                       beta2=beta2,
                                       epsilon=koptim_method.epsilon)
            elif isinstance(koptim_method, koptimizers.RMSprop):
                rho = float(K.eval(koptim_method.rho))
                return boptimizer.RMSprop(learningrate=lr,
                                          learningrate_decay=decay,
                                          decayrate=rho,
                                          epsilon=koptim_method.epsilon)
            elif isinstance(koptim_method, koptimizers.Adadelta):
                warnings.warn(
                    "For Adadelta, we don't support learning rate and learning rate decay for now")
                return boptimizer.Adadelta(decayrate=koptim_method.rho,
                                           epsilon=koptim_method.epsilon)
            elif isinstance(koptim_method, koptimizers.Adamax):
                beta1 = float(K.eval(koptim_method.beta_1))
                beta2 = float(K.eval(koptim_method.beta_2))
                warnings.warn("For Adamax, we don't support learning rate decay for now")
                return boptimizer.Adamax(learningrate=lr,
                                         beta1=beta1,
                                         beta2=beta2,
                                         epsilon=koptim_method.epsilon)
        elif isinstance(koptim_method, tftrain.Optimizer):
            def get_value(v):
                if isinstance(v, (tf.Tensor, tf.SparseTensor, tf.Variable)):
                    return float(K.eval(v))
                else:
                    return float(v)
            if isinstance(koptim_method, tftrain.GradientDescentOptimizer):
                lr = get_value(koptim_method._learning_rate)
                return boptimizer.SGD(learningrate=lr)
            elif isinstance(koptim_method, tftrain.MomentumOptimizer):
                lr = get_value(koptim_method._learning_rate)
                momentum = get_value(koptim_method._momentum)
                use_nesterov = koptim_method._use_nesterov
                return boptimizer.SGD(learningrate=lr, momentum=momentum, nesterov=use_nesterov)
            elif isinstance(koptim_method, tftrain.AdagradOptimizer):
                lr = get_value(koptim_method._learning_rate)
                return boptimizer.Adagrad(learningrate=lr)
            elif isinstance(koptim_method, tftrain.AdamOptimizer):
                lr = get_value(koptim_method._lr)
                beta1 = get_value(koptim_method._beta1)
                beta2 = get_value(koptim_method._beta2)
                epsilon = get_value(koptim_method._epsilon)
                return boptimizer.Adam(learningrate=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
            elif isinstance(koptim_method, tftrain.RMSPropOptimizer):
                lr = get_value(koptim_method._learning_rate)
                decay = get_value(koptim_method._decay)
                momentum = get_value(koptim_method._momentum)
                epsilon = get_value(koptim_method._epsilon)
                centered = get_value(koptim_method._centered)
                if momentum != 0.0 or centered:
                    warnings.warn(
                        "For RMSPropOptimizer, we don't support momentum and centered for now")
                return boptimizer.RMSprop(learningrate=lr,
                                          learningrate_decay=decay,
                                          epsilon=epsilon)
            elif isinstance(koptim_method, tftrain.AdadeltaOptimizer):
                lr = get_value(koptim_method._lr)
                rho = get_value(koptim_method._rho)
                epsilon = get_value(koptim_method._epsilon)
                warnings.warn(
                    "For Adadelta, we don't support learning rate for now")
                return boptimizer.Adadelta(decayrate=rho, epsilon=epsilon)

        raise ValueError("We don't support %s for now" % koptim_method)

    def refresh_weights(self):
        from zoo.util.tf import export_tf
        export_tf(self.sess, self.export_dir,
                  inputs=self.inputs,
                  outputs=self.grads + self.outputs)
        self.training_helper_layer = TFTrainingHelper(self.export_dir, self.session_config)

    def set_train_summary(self, summary):
        self.optimizer.set_train_summary(summary)

    def set_val_summary(self, summary):
        self.optimizer.set_val_summary(summary)

    def optimize(self, end_trigger=None):
        if end_trigger is None:
            end_trigger = MaxEpoch(1)

        self.optimizer.set_end_when(end_trigger)

        self.optimizer.optimize()

        variables = self.training_helper_layer.get_weights()

        feed_dict = dict(zip(self.variable_placeholders, variables))
        self.sess.run(self.assign, feed_dict=feed_dict)


class TensorMeta(object):
    def __init__(self, dtype, name=None, shape=None):
        self.dtype = dtype
        self.name = name
        self.shape = shape


class TFDataset:
    def __init__(self, rdd, tensor_structure, batch_size,
                 batch_per_thread, hard_code_batch_size=False, val_rdd=None):
        '''
        TFDatasets represents a distributed collection of elements to be feed into Tensorflow
        graph. TFDatasets can be created using a RDD and each of its records is one or more
        numpy.ndarray of the same nested structure, representing the tensors to be feed into
        TensorFlow graph on each iteration. TFDatasets must be used with TFOptimizer or
        TFPredictor.
        '''

        if batch_size > 0 and batch_per_thread > 0:
            raise ValueError("bath_size and batch_per_thread should not be set simultaneously")

        self.has_batch = True
        node_num, core_num = get_node_and_core_number()
        self.total_core_num = node_num * core_num
        if batch_size > 0:
            if batch_size % self.total_core_num != 0:
                raise ValueError("batch_size should be a multiple " +
                                 "of total core number, but got batch_size: " +
                                 "%s where total core number is %s" % (batch_size,
                                                                       self.total_core_num))
        if batch_size <= 0 and batch_per_thread <= 0:
            batch_per_thread = 1
            batch_size = self.total_core_num
            self.has_batch = False

        self.batch_size = batch_size
        self.batch_per_thread = batch_per_thread
        self.hard_code_batch_size = hard_code_batch_size
        self.tensor_structure = tensor_structure

        self.val_rdd = val_rdd

        if not self.hard_code_batch_size:
            self.output_shapes = nest.pack_sequence_as(
                self.tensor_structure, [[None] + list(t.shape)
                                        if t is not None else None
                                        for t in nest.flatten(self.tensor_structure)])
        else:
            if self.batch_per_thread > 0:
                self.output_shapes = nest.pack_sequence_as(
                    self.tensor_structure, [[self.batch_per_thread] + t.shape
                                            if t is not None else None
                                            for t in nest.flatten(self.tensor_structure)])
            else:
                self.output_shapes = nest.pack_sequence_as(
                    self.tensor_structure, [[self.batch_size // self.total_core_num] + t.shape
                                            if t is not None else None
                                            for t in nest.flatten(self.tensor_structure)])

        self.rdd = rdd
        self.input_names = nest.pack_sequence_as(
            self.tensor_structure, [t.name
                                    if t is not None else None
                                    for t in nest.flatten(self.tensor_structure)])

        self._tensors = None

    def _create_placeholders(self):
        import tensorflow as tf
        if not self.hard_code_batch_size:
            tensors = nest.pack_sequence_as(
                self.tensor_structure, [tf.placeholder(name=t.name,
                                                       dtype=t.dtype,
                                                       shape=[None] + list(t.shape))
                                        for t in nest.flatten(self.tensor_structure)])
        else:
            if self.batch_per_thread > 0:
                tensors = nest.pack_sequence_as(
                    self.tensor_structure,
                    [tf.placeholder(name=t.name,
                                    dtype=t.dtype,
                                    shape=[self.batch_per_thread] + list(t.shape))
                     for t in nest.flatten(self.tensor_structure)])
            else:
                tensors = nest.pack_sequence_as(
                    self.tensor_structure,
                    [tf.placeholder(name=t.name,
                                    dtype=t.dtype,
                                    shape=[self.batch_size // self.total_core_num] +
                                    list(t.shape))
                     for t in nest.flatten(self.tensor_structure)])

        for tensor in nest.flatten(tensors):
            tf.get_default_graph().clear_collection(tensor.name)
            tf.add_to_collection(tensor.name, self)

        self._original_tensors = tensors
        self._tensors = tensors

        if not self.has_batch:
            self._tensors = nest.pack_sequence_as(self.tensor_structure,
                                                  [t[0] for t in nest.flatten(tensors)])

        return tensors

    @property
    def tensors(self):
        '''
        a nested structure of TensorFlow tensor object in TensorFlow graph.
        The elements of this dataset will be fed into these tensors on each iteration.
        :return: the nested structure of TensorFlow tensor object
        '''

        if self._tensors is None:
            self._create_placeholders()

        return self._tensors

    @property
    def feature_tensors(self):

        if self._tensors is None:
            self._create_placeholders()

        if not isinstance(self._tensors, tuple):
            raise ValueError("To use feature_tensors, " +
                             "the element in TFDataset must be a tuple of two components. " +
                             "Please use TFDataset.from_rdd(rdd, features=..., labels=...). ")

        return self._tensors[0]

    @property
    def label_tensors(self):

        if self._tensors is None:
            self._create_placeholders()

        if not isinstance(self._tensors, tuple):
            raise ValueError("To use label_tensors, " +
                             "the element in TFDataset must be a tuple of two components. " +
                             "Please use TFDataset.from_rdd(rdd, features=..., labels=...). ")

        return self._tensors[1]

    @staticmethod
    def from_rdd(rdd, names=None, shapes=None, types=None,
                 batch_size=-1, batch_per_thread=-1,
                 hard_code_batch_size=False, val_rdd=None,
                 features=None, labels=None):
        '''
        Create a TFDataset from a rdd, each element of the rdd must be a list of numpy.ndarray.

        :param rdd: a rdd of list of numpy.ndarray each representing a tensor to feed into
        tensorflow graph on each iteration
        :param names: the names of the resulting tensors, should be a list of str
        :param shapes: the shapes of the resulting tensors, should be a list of list of int
        :param types: the types of the result tensors, should be a list of tf.dtype
        :param batch_size: the batch size, used for training, should be a multiple of
        total core num
        :param batch_per_thread: the batch size for each thread, used for inference
        :param hard_code_batch_size: whether to hard code the batch_size into tensorflow graph,
        if True, the static size of the first dimension of the resulting tensors is
        batch_size/total_core_num (training) or batch_per_thread for inference; if False,
        it is None.
        :param val_rdd: validation data with the same structure of rdd
        :return: a TFDataset
        '''
        import tensorflow as tf

        if features is not None:
            feature_structure = _to_tensor_structure(features)
            if labels is not None:
                label_structure = _to_tensor_structure(labels)
                tensor_structure = (feature_structure, label_structure)

            else:
                tensor_structure = (feature_structure,)

            return TFDataset(rdd, tensor_structure,
                             batch_size, batch_per_thread,
                             hard_code_batch_size, val_rdd)

        if names is not None or shapes is not None or types is not None:
            if not names:
                names = ["features", "labels"]
            if not shapes:
                shapes = [None] * len(names)

            if not types:
                types = [tf.float32] * len(names)
            tensor_structure = []
            for i in range(len(names)):
                tensor_structure.append(TensorMeta(types[i], name=names[i], shape=shapes[i]))
        else:
            tensor_structure = [TensorMeta(dtype=tf.float32), TensorMeta(dtype=tf.float32)]

        return TFDataset(rdd, tensor_structure,
                         batch_size, batch_per_thread,
                         hard_code_batch_size, val_rdd)

    @staticmethod
    def from_ndarrays(tensors, batch_size=-1, batch_per_thread=-1,
                      hard_code_batch_size=False, val_tensors=None):
        sc = getOrCreateSparkContext()
        node_num, core_num = get_node_and_core_number()
        total_core_num = node_num * core_num

        rdd, tensor_structure = _tensors_to_rdd(tensors, sc, total_core_num)

        val_rdd = None
        if val_tensors is not None:
            val_rdd = _tensors_to_rdd(val_tensors, sc, total_core_num)

        return TFDataset(rdd, tensor_structure, batch_size,
                         batch_per_thread, hard_code_batch_size, val_rdd)


def _tensors_to_rdd(tensors, sc, splits):
    import tensorflow as tf
    if isinstance(tensors, np.ndarray):
        tensors = (tensors,)

    if isinstance(tensors, list):
        for i in range(len(tensors)):
            if tensors[i].dtype == np.dtype("float64"):
                tensors[i] = np.float32(tensors[i])

        data_list = _splits(tensors)
        rdd = sc.parallelize(data_list, splits)
        tensor_structure = [TensorMeta(tf.as_dtype(t.dtype),
                                       shape=t.shape[1:],
                                       name="input_%s" % i)
                            for i, t in enumerate(tensors)]
    else:
        flattened = nest.flatten(tensors)
        for i in range(len(flattened)):
            if flattened[i].dtype == np.dtype("float64"):
                flattened[i] = np.float32(flattened[i])
        data_list = _splits(flattened)
        rdd = sc.parallelize(data_list, splits)
        rdd = rdd.map(lambda x: nest.pack_sequence_as(tensors, x))
        tensor_structure = nest.pack_sequence_as(tensors,
                                                 [TensorMeta(tf.as_dtype(t.dtype),
                                                             shape=t.shape[1:],
                                                             name="input_%s" % i)
                                                  for i, t in enumerate(flattened)])
    return rdd, tensor_structure


def _splits(tensors):
    data_list = []
    data_size = tensors[0].shape[0]
    for i in range(data_size):
        sample = []
        for j in range(len(tensors)):
            sample.append(tensors[j][i])
        data_list.append(sample)
    return data_list


def _to_tensor_structure(tensors):
    if isinstance(tensors, tuple):
        tensor_structure = TensorMeta(dtype=tensors[0], shape=tensors[1], name="input0")
    elif isinstance(tensors, list):
        tensor_structure = [TensorMeta(dtype=value[0], shape=value[1], name=idx)
                            for (idx, value) in enumerate(tensors)]
    elif isinstance(tensors, dict):
        tensor_structure = {}
        for key, value in tensors.items():
            tensor_structure[key] = TensorMeta(dtype=value[0], shape=value[1], name=key)
    else:
        raise ValueError("In TFDataset.from_rdd, features and labels should be a tuple, "
                         "a list of tuples or a dict of tuples")
    return tensor_structure


def _check_the_same(all_required_inputs, inputs_in_datasets):
    inputs_not_in_dataset = [i for i in all_required_inputs if i not in inputs_in_datasets]
    if inputs_not_in_dataset:
        raise ValueError("You should not use any placeholder that are not defined in dataset, " +
                         "found %s" % inputs_not_in_dataset)
    if len(inputs_in_datasets) != len(all_required_inputs):
        inputs_not_require_by_loss = [i for i in inputs_in_datasets if i not in all_required_inputs]
        raise ValueError("You should use all the placeholders that are defined in dataset, " +
                         "%s are not used" % inputs_not_require_by_loss)


class TFPredictor:
    def __init__(self, sess, outputs, inputs=None, dataset=None):
        '''
        TFPredictor takes a list of TensorFlow tensors as the model outputs and
        feed all the elements in TFDatasets to produce those outputs and returns
        a Spark RDD with each of its elements representing the model prediction
        for the corresponding input elements.

        :param sess: the current TensorFlow Session, you should first use this session
        to load the trained variables then pass into TFPredictor
        :param outputs: the output tensors of the TensorFlow model
        '''
        if inputs is None:
            dataset, inputs = TFPredictor._get_datasets_and_inputs(outputs)

        self.sess = sess
        self.dataset = dataset
        self.inputs = inputs
        self.tfnet = TFNet.from_session(sess, self.inputs, outputs)
        if self.dataset.batch_per_thread <= 0:
            raise ValueError("You should set batch_per_thread on TFDataset" +
                             "instead of batch_size for prediction")

    @staticmethod
    def _get_datasets_and_inputs(outputs):
        import tensorflow as tf
        all_required_inputs = _find_placeholders(outputs)
        dataset = tf.get_collection(all_required_inputs[0].name)[0]
        inputs = dataset.tensors
        _check_the_same(all_required_inputs, inputs)
        return dataset, inputs

    @classmethod
    def from_outputs(cls, sess, outputs):
        dataset, inputs = TFPredictor._get_datasets_and_inputs(outputs)
        return cls(sess, outputs, inputs, dataset)

    @classmethod
    def from_keras(cls, keras_model, dataset):
        import tensorflow.keras.backend as K
        sess = K.get_session()

        outputs = keras_model.outputs
        inputs = keras_model.inputs
        return cls(sess, outputs, inputs, dataset)

    def predict(self):
        rdd = self.dataset.rdd
        sample_rdd = rdd.map(lambda x: Sample.from_ndarray(x, np.array([0.0])))

        return self.tfnet.predict(sample_rdd, self.dataset.batch_per_thread)
