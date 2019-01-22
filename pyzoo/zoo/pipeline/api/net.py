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
from bigdl.util.common import to_list, callBigDlFunc, get_spark_context, \
    JavaValue, get_node_and_core_number
from zoo.common import Sample, JTensor
from zoo.feature.image import ImageSet
from zoo.pipeline.api.keras.engine.topology import ZooKerasLayer, KerasNet, to_bigdl_metric
from bigdl.optim.optimizer import EveryEpoch, MaxEpoch, Optimizer, Adam

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
                data_rdd = to_sample_rdd(x, np.zeros([x.shape[0]]), get_spark_context())
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
        import tensorflow as tf
        from zoo.util.tf import export_tf
        '''
        TFOptimizer is used for distributed training of tensorflow
        on Spark/BigDL.

        :param loss: The loss tensor of the tensorflow model, should be a scalar
        :param optim_method: the optimization method to be used, such as bigdl.optim.optimizer.Adam
        :param sess: the current tensorflow Session, if you want to used a pre-trained model, you
        should use the Session to load the pre-trained variables and pass it to TFOptimizer.
        '''

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

        self.optim_method = optim_method
        self.sess = sess
        self.dataset = dataset
        self.inputs = inputs + additional_inputs
        self.graph = graph

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

        self.export_dir = tempfile.mkdtemp()
        export_tf(self.sess, self.export_dir,
                  inputs=self.inputs,
                  outputs=grads + outputs)

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
known issue of tensorflow, see https://github.com/tensorflow/tensorflow/issues/21889.
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
        sample_rdd = data.map(lambda t: Sample.from_ndarray(
            t, [np.array([0.0])]))
        if val_outputs is not None and val_labels is not None:
            if self.dataset.val_rdd is not None:
                val_rdd = self.dataset.val_rdd \
                    .map(lambda t: Sample.from_ndarray(t,
                                                       [np.array([0.0])]))
                val_method = [TFValidationMethod(m, len(val_outputs), len(val_labels))
                              for m in to_list(val_method)]
                training_rdd = sample_rdd

            elif val_split != 0.0:
                training_rdd, val_rdd = sample_rdd.randomSplit([1-val_split, val_split])
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

        inputs = dataset.tensors

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
        with sess.as_default():
            optim_method = TFOptimizer.to_bigdl_optim_method(keras_optimizer)

        if keras_model.metrics:
            if isinstance(keras_model.metrics, dict):
                raise ValueError(
                    "different metrics for different outputs are not supported right now")

            if dataset.val_rdd is None and val_spilt == 0.0:
                raise ValueError("Validation data is not specified. Please set " +
                                 "val rdd in TFDataset, or set val_split larger than zero")
            bigdl_val_methods = [to_bigdl_metric(m) for m in keras_model.metrics_names]
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
        else:
            raise Exception("We don't support %s for now" % koptim_method)

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


class TFDataset:

    def __init__(self, rdd, names, shapes, types, batch_size,
                 batch_per_thread, hard_code_batch_size=False, val_rdd=None):
        import tensorflow as tf
        '''
        TFDatasets represents a distributed collection of elements to be feed into
        Tensorflow graph. TFDatasets can be created using a RDD and each of its records
        is a list of numpy.ndarray representing the tensors to be feed into tensorflow
        graph on each iteration. TFDatasets must be used with TFOptimizer or TFPredictor.

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
        '''
        if batch_size > 0 and batch_per_thread > 0:
            raise ValueError("bath_size and batch_per_thread should not be set simultaneously")

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
        self.batch_size = batch_size
        self.batch_per_thread = batch_per_thread

        if not hard_code_batch_size:
            self.tensors = [tf.placeholder(name=names[i],
                                           dtype=types[i],
                                           shape=[None] + shapes[i])
                            for i in range(len(names))]
        else:
            if batch_per_thread > 0:
                self.tensors = [tf.placeholder(name=names[i],
                                               dtype=types[i],
                                               shape=[batch_per_thread] + shapes[i])
                                for i in range(len(names))]
            else:
                self.tensors = [tf.placeholder(name=names[i],
                                               dtype=types[i],
                                               shape=[batch_size / self.total_core_num] + shapes[i])
                                for i in range(len(names))]

        self.val_rdd = val_rdd

        self.rdd = rdd.map(lambda arr: arr[:len(names)])
        self.input_names = names
        for i in range(len(self.tensors)):
            tf.add_to_collection(self.tensors[i].name, self)

    @staticmethod
    def from_rdd(rdd, names=None, shapes=None, types=None,
                 batch_size=-1, batch_per_thread=-1,
                 hard_code_batch_size=False, val_rdd=None):
        import tensorflow as tf
        if not names:
            names = ["features", "labels"]
        if not shapes:
            shapes = [None] * len(names)

        if not types:
            types = [tf.float32] * len(names)
        return TFDataset(rdd, names, shapes, types,
                         batch_size, batch_per_thread,
                         hard_code_batch_size, val_rdd)


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
        TFPredictor takes a list of tensorflow tensors as the model outputs and
        feed all the elements in TFDatasets to produce those outputs and returns
        a Spark RDD with each of its elements representing the model prediction
        for the corresponding input elements.

        :param sess: the current tensorflow Session, you should first use this session
        to load the trained variables then pass into TFPredictor
        :param outputs: the output tensors of the tensorflow model
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
