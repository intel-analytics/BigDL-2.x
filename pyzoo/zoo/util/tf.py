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

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
import tensorflow as tf
import os
import json
import copy


def export_tf(sess, folder, inputs, outputs,
              generate_backward=False, allow_non_differentiable_input=True):
    """
    Export the frozen tensorflow graph as well as the inputs/outputs information
    to the folder for inference.

    This function will
    1. freeze the graph (replace all variables with constants)
    2. strip all unused node as specified by inputs and outputs
    3. add placeholder nodes as needed
    4. write the frozen graph and inputs/outputs names to the folder

    Note: There should not be any queuing operation between inputs and outputs

    :param sess: tensorflow session holding the variables to be saved
    :param folder: the folder where graph file and inputs/outputs information are saved
    :param inputs: a list of tensorflow tensors that will be fed during inference
    :param outputs: a list of tensorflow tensors that will be fetched during inference
    :return:
    """

    output_node_names = list({t.op.name for t in outputs})

    graph_def = sess.graph_def
    graph = sess.graph

    # clear device specifications
    for node in graph_def.node:
        node.device = ""

    non_placeholder_input_names = []
    type_enums = []
    for input_tensor in inputs:
        if input_tensor.op.type != "Placeholder":
            non_placeholder_input_names.append(input_tensor.name)
            type_enums.append(input_tensor.dtype.as_datatype_enum)

    output_names = list(map(lambda o: o.name, outputs))

    all_variables = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    # freeze graph
    frozen_graph_def = graph_util.convert_variables_to_constants(
        sess,
        graph_def,
        output_node_names
    )

    optimized_graph_def, old_names2new = strip_unused(frozen_graph_def,
                                                      non_placeholder_input_names,
                                                      output_names,
                                                      type_enums)

    new_input_names = []
    for t in inputs:
        if t.name in old_names2new:
            new_input_names.append(old_names2new[t.name])
        else:
            new_input_names.append(t.name)

    # check all placeholder in the graph are listed in the new_input_names:
    new_input_nodes = {name.split(":")[0] for name in new_input_names}
    for node in optimized_graph_def.node:
        if node.op == "Placeholder" and node.name not in new_input_nodes:
            raise ValueError(
                "Node %s is a Placeholder but not listed in inputs, inputs are %s"
                % (node.name, inputs))

    temp_tensors = None
    used_variables = []
    grad_variables = []
    grad_inputs = []
    if generate_backward:
        nodes = set(map(lambda n: n.name, optimized_graph_def.node))
        for v in all_variables:
            if v.op.name in nodes:
                used_variables.append(v.name)

        with tf.Graph().as_default() as g:
            tf.import_graph_def(optimized_graph_def, name='')
            output_tensors = map(lambda x: g.get_tensor_by_name(x), output_names)
            grad_output_placeholders =\
                map(lambda x:
                    tf.placeholder(dtype=x.dtype,
                                   name=x.name.split(":")[0] + "_grad",
                                   shape=x.shape),
                    output_tensors)

            variables = map(lambda x: g.get_tensor_by_name(x), used_variables)

            inputs = map(lambda x: g.get_tensor_by_name(x), new_input_names)
            grads = tf.gradients(output_tensors, variables + inputs,
                                 grad_ys=grad_output_placeholders)

            def process_grad(g):
                if g is not None:
                    g = ops.convert_to_tensor_or_indexed_slices(g)
                    if isinstance(g, ops.IndexedSlices):
                        # In IndexedSlices is not supported in java api, we have to convert it to
                        # a dense tensor. This operation is potentially expensive, but there seems
                        # no work around
                        g = tf.unsorted_segment_sum(g.values, g.indices, g.dense_shape[0])
                return g

            grads = list(map(lambda g: process_grad(g), grads))

            temp_tensors = _find_temp_tensors(grads, nodes)

            grad_variables = list(map(lambda x: x.name, grads[0:len(variables)]))

            grad_inputs = []
            for i in range(len(variables), len(grads)):
                grad = grads[i]
                if grad is not None:
                    grad_inputs.append(grad.name)
                else:
                    # if input is not differentiable, we just return zero
                    input_tensor = inputs[i - len(variables)]
                    if allow_non_differentiable_input:
                        zero_grad = tf.zeros(shape=tf.shape(input_tensor))
                        grad_inputs.append(zero_grad.name)
                    else:
                        raise ValueError(
                            "input tensor: %s is not differentiable" % input_tensor.name)

            optimized_graph_def = g.as_graph_def()

    if not os.path.isdir(folder):
        os.makedirs(folder)

    with gfile.GFile(os.path.join(folder, "frozen_inference_graph.pb"), "wb") as f:
        f.write(optimized_graph_def.SerializeToString())

    meta = {
        "input_names": new_input_names,
        "output_names": output_names
    }

    if generate_backward:
        meta["temp_tensors"] = list(temp_tensors)
        meta["variables"] = used_variables
        meta["grad_variables"] = grad_variables
        meta["grad_inputs"] = grad_inputs

    with open(os.path.join(folder, "graph_meta.json"), "w") as f:
        f.write(json.dumps(meta))


def _insert_identity_nodes(graph_def, temp_tensors):
    new_temp_tensors = []
    new_graph_def = graph_pb2.GraphDef()
    added_nodes = set()
    name2node = {}
    for node in graph_def.node:
        name2node[node.name] = node

    for node in graph_def.node:
        for i in range(len(node.input)):
            input_name = _append_port(node.input[i])
            if input_name in temp_tensors:
                new_node_name = input_name.replace(":", "_") + "_identity"
                input_node_name = input_name.split(":")[0]
                input_node_port = int(input_name.split(":")[1])
                input_node = name2node[input_node_name]
                if new_node_name not in added_nodes:
                    added_nodes.add(new_node_name)
                    identity_node = node_def_pb2.NodeDef()
                    identity_node.op = "Identity"
                    identity_node.name = new_node_name
                    identity_node.attr['T'].type = input_node.attr['T'].type
                    identity_node.input.append(node.input[i])
                    node.input[i] = identity_node.name + ":0"
                    new_temp_tensors.append(node.input[i])
                    new_graph_def.node.extend([identity_node])
        new_graph_def.node.extend([copy.deepcopy(node)])
    return new_graph_def, new_temp_tensors


def _find_temp_tensors(grads, forward_ops):
    import sys
    is_py2 = sys.version[0] == '2'
    if is_py2:
        import Queue as queue
    else:
        import queue as queue
    queue = queue.Queue()
    for grad in grads:
        queue.put(grad)

    temp_tensors = set()
    visited = set()
    while not queue.empty():
        tensor = queue.get()
        # this is necessary, because input may not be differentiable
        if tensor is None:
            continue
        else:
            visited.add(tensor.name)
            if tensor.op.type == "Placeholder":
                continue
            if tensor.op.name in forward_ops:
                temp_tensors.add(tensor.name)
                continue
            for input_tensor in tensor.op.inputs:
                # this is necessary because there may be a cycle in the graph such as tf.while_loop
                if input_tensor.name not in visited:
                    queue.put(input_tensor)
    return temp_tensors


def strip_unused(input_graph_def, input_tensor_names, output_tensor_names,
                 placeholder_type_enum):
    """Removes unused nodes from a GraphDef.

  Args:
    input_graph_def: A graph with nodes we want to prune.
    input_tensor_names: A list of the nodes we use as inputs.
    output_tensor_names: A list of the output nodes.
    placeholder_type_enum: The AttrValue enum for the placeholder data type, or
        a list that specifies one value per input node name.

  Returns:
    A `GraphDef` with all unnecessary ops removed. and a map containing the old input
    names to the new input names

  Raises:
    ValueError: If any element in `input_node_names` refers to a tensor instead
      of an operation.
    KeyError: If any element in `input_node_names` is not found in the graph.
  """
    for name in input_tensor_names:
        if ":" not in name:
            raise ValueError("Input '%s' appears to refer to a Operation, "
                             "not a Tensor." % name)

    old2new = {}

    # Here we replace the nodes we're going to override as inputs with
    # placeholders so that any unused nodes that are inputs to them are
    # automatically stripped out by extract_sub_graph().
    not_found = {name for name in input_tensor_names}
    input_node_names = {name.split(":")[0] for name in input_tensor_names}
    output_node_names = list({name.split(":")[0] for name in output_tensor_names})
    inputs_replaced_graph_def = graph_pb2.GraphDef()
    for node in input_graph_def.node:
        if node.name not in input_node_names:
            for i in range(len(node.input)):
                if _append_port(node.input[i]) in input_tensor_names:
                    old_name = _append_port(node.input[i])
                    not_found.remove(old_name)
                    new_input_name = node.input[i].replace(":", "_")
                    placeholder_node = node_def_pb2.NodeDef()
                    placeholder_node.op = "Placeholder"
                    placeholder_node.name = new_input_name
                    if isinstance(placeholder_type_enum, list):
                        input_node_index = input_tensor_names.index(old_name)
                        placeholder_node.attr["dtype"].CopyFrom(
                            attr_value_pb2.AttrValue(type=placeholder_type_enum[
                                input_node_index]))
                    else:
                        placeholder_node.attr["dtype"].CopyFrom(
                            attr_value_pb2.AttrValue(type=placeholder_type_enum))
                    if "_output_shapes" in node.attr:
                        placeholder_node.attr["_output_shapes"].CopyFrom(
                            node.attr["_output_shapes"])
                    node.input[i] = new_input_name
                    old2new[old_name] = new_input_name + ":0"
                    inputs_replaced_graph_def.node.extend([placeholder_node])
            inputs_replaced_graph_def.node.extend([copy.deepcopy(node)])

    if not_found:
        raise KeyError("The following input nodes were not found: %s\n" % not_found)

    output_graph_def = graph_util.extract_sub_graph(inputs_replaced_graph_def,
                                                    output_node_names)
    return output_graph_def, old2new


def _append_port(input_name):
    if input_name.find(":") == -1:
        return input_name + ":0"
    else:
        return input_name
