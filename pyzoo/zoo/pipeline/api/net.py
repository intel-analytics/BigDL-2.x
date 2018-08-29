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

import six
import os
import json
import tensorflow as tf
import numpy as np

from bigdl.nn.layer import Model as BModel
from bigdl.nn.layer import Layer
from bigdl.util.common import callBigDlFunc, to_list
from zoo.pipeline.api.keras.engine.topology import ZooKerasLayer, KerasNet
from zoo.util.tf import export_tf
from bigdl.optim.optimizer import *

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
    def from_export_folder(folder):
        if not os.path.isdir(folder):
            raise ValueError(folder + " does not exist")
        return TFNet(folder)

    @staticmethod
    def from_session(sess, inputs, outputs,
                     generate_backward=False, allow_non_differentiable_input=True):
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


class TFOptimizer:
    def __init__(self, loss, optim_method, sess=None):
        self.optim_method = optim_method
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        grads_vars = tf.train.GradientDescentOptimizer(0).compute_gradients(loss)
        variables = []
        grads = []
        for (grad, var) in grads_vars:
            variables.append(var)
            grads.append(grad)
        self.export_dir = tempfile.mkdtemp()
        all_required_inputs = _find_placeholders([loss])
        self.dataset = tf.get_collection(all_required_inputs[0].name)[0]
        self.inputs = self.dataset.inputs

        inputs_not_in_dataset = [i for i in all_required_inputs if i not in self.inputs]
        if inputs_not_in_dataset:
            raise ValueError("You should not use any placeholder that are not defined in dataset, "
                             "found %s" % inputs_not_in_dataset)
        if len(self.inputs) != len(all_required_inputs):
            inputs_not_require_by_loss = [i for i in self.inputs if i not in all_required_inputs]
            raise ValueError("You should use all the placeholders that are defined in dataset, "
                             "%s are not used" % inputs_not_require_by_loss)

        export_tf(self.sess, self.export_dir, inputs=self.inputs, outputs=grads + [loss])

        variable_names = [v.name for v in variables]
        grad_names = [g.name for g in grads]

        meta = {
            "input_names": [i.name for i in self.inputs],
            "output_names": [loss.name],
            "variables": variable_names,
            "grad_variables": grad_names
        }

        with open(os.path.join(self.export_dir, "training_meta.json"), "w") as f:
            f.write(json.dumps(meta))

        self.variable_placeholders = []
        assigns = []
        for v in variables:
            p = tf.placeholder(dtype=tf.float32, shape=v.shape)
            a = tf.assign(v, p)
            self.variable_placeholders.append(p)
            assigns.append(a)
        self.assign = tf.group(assigns)

    def optimize(self, end_trigger=MaxEpoch(1), batch_size=32):
        data = self.dataset.rdd

        sample_rdd = data.map(lambda t: Sample.from_ndarray(t, [np.array([0.0])]))
        variables = Layer.convert_output(callBigDlFunc("float", "trainTFNet",
                                                       self.export_dir, self.optim_method,
                                                       sample_rdd, batch_size, end_trigger))

        feed_dict = dict(zip(self.variable_placeholders, variables))
        self.sess.run(self.assign, feed_dict=feed_dict)


class TFDataset:

    def __init__(self, rdd, names, shapes, types):
        self.rdd = rdd.map(lambda arr: arr[:len(names)])
        self.input_names = names
        self.inputs = [tf.placeholder(name=names[i],
                                      dtype=types[i],
                                      shape=shapes[i]) for i in range(len(names))]
        for i in range(len(self.inputs)):
            tf.add_to_collection(self.inputs[i].name, self)

    @staticmethod
    def from_dataframe(dataframe):
        input_names = dataframe.schema.names

        def _get_data(row, tensor_names):
            _names = [n.split(":")[0] for n in tensor_names]
            _data = [np.array(row[n]) for n in _names]
            return _data
        data = dataframe.rdd\
            .map(lambda r: _get_data(r, input_names))\
            .map(lambda t: Sample.from_ndarray(t, [np.array([0.0])]))
        return TFDataset(data, input_names, [None]*len(input_names), [tf.float32]*len(input_names))

    @staticmethod
    def from_rdd(rdd, names=None, shapes=None, types=None):
        if not names:
            names = ["features", "labels"]
        if not shapes:
            shapes = [None] * len(names)

        if not types:
            types = [tf.float32] * len(names)
        return TFDataset(rdd, names, shapes, types)
