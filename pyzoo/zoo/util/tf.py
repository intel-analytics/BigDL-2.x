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
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
import tensorflow as tf
import os
import json
import copy

import zoo.util.tf_graph_util as graph_util


def process_grad(grad):
    if grad is not None:
        grad = ops.convert_to_tensor_or_indexed_slices(grad)
        if isinstance(grad, ops.IndexedSlices):
            # In IndexedSlices is not supported in java api, we have to convert it to
            # a dense tensor. This operation is potentially expensive, but there seems
            # no work around
            grad = tf.unsorted_segment_sum(grad.values, grad.indices,
                                           grad.dense_shape[0])
    return grad


def _find_placeholders(grads):
    '''
    find all the tensors that are used for computing grads and has been
    computed during forward
    :param grads:
    :param forward_ops:
    :return:
    '''
    from collections import deque
    queue = deque([])
    for grad in grads:
        queue.append(grad)

    placeholders = set()
    visited = set()
    while len(queue) > 0:
        tensor = queue.popleft()
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
                    queue.append(input_tensor)
    return list(placeholders)


def _to_operation_name(name):
    return name.split(":")[0]


def export_tf_for_train(sess,
                        folder,
                        inputs,
                        loss,
                        variables,
                        grad_variables,
                        val_outputs,
                        val_labels,
                        tensors_with_value,
                        updates=None):

    graph = sess.graph

    additional_inputs = []
    additional_values = []
    all_required_inputs = _find_placeholders([loss])
    all_required_inputs_names = [v.name for v in all_required_inputs]
    if tensors_with_value:
        for t, v in tensors_with_value.items():
            if t.name in all_required_inputs_names:
                additional_inputs.append(t)
                additional_values.append(v)

    all_trainable_variables = variables

    all_variables = graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    update_ops = graph.get_collection(tf.GraphKeys.UPDATE_OPS)

    if updates is not None:
        update_ops += updates

    name2idx = dict([(v.name, idx) for idx, v in enumerate(all_trainable_variables)])

    with graph.as_default():

        if val_outputs is not None and val_labels is not None:
            val_labels = [tf.to_float(tf.identity(v)) for v in val_labels]
            outputs = val_outputs + val_labels + [loss]
        else:
            outputs = [loss]

        grads = [process_grad(grad) for grad in grad_variables]

        trainable_variables = []
        trainable_assigns = []
        trainable_variable_placeholders = []
        extra_variables = []
        extra_variable_assigns = []
        extra_variable_assign_placeholders = []
        for v in all_variables:
            p = tf.placeholder(dtype=v.dtype, shape=v.shape)
            a = tf.assign(v, p)

            # special treatment for ResourceVariable
            if v.op.type == "VarHandleOp":
                v_float_value = tf.to_float(v.read_value())
            else:
                v_float_value = tf.to_float(v)

            if v.name in name2idx:
                trainable_variables.append(v_float_value)
                trainable_assigns.append(a)
                trainable_variable_placeholders.append(p)
            else:
                extra_variables.append(v_float_value)
                extra_variable_assigns.append(a)
                extra_variable_assign_placeholders.append(p)

        extra_variable_assign = tf.group(*extra_variable_assigns)
        trainable_assign = tf.group(*trainable_assigns)
        update_op = tf.group(update_ops)

        saver = tf.train.Saver()
        if not os.path.isdir(folder):
            os.makedirs(folder)
        saver.save(sess, os.path.join(folder, "model"), write_meta_graph=False)
    inputs = inputs + additional_inputs
    output_names = [o.name for o in outputs]
    input_names = [i.name for i in inputs]

    meta = {
        "inputs": input_names,
        "input_types": [i.dtype.as_datatype_enum for i in inputs],
        "outputs": output_names,
        "variables": [v.name for v in trainable_variables],
        "variable_types": [v.dtype.as_datatype_enum for v in trainable_variable_placeholders],
        "variable_assign_placeholders": [v.name for v in trainable_variable_placeholders],
        "assign_variable_op": trainable_assign.name,
        "extra_variables": [v.name for v in extra_variables],
        "extra_variable_types": [v.dtype.as_datatype_enum for v in extra_variable_assign_placeholders],
        "extra_variable_assign_placeholders": [p.name for p in extra_variable_assign_placeholders],
        "assign_extra_variable_op": extra_variable_assign.name,
        "grad_variables": [g.name for g in grads],
        "update_op": update_op.name,
        "restore_op": saver.saver_def.restore_op_name,
        "restore_path_placeholder": saver.saver_def.filename_tensor_name,
        "save_op": _to_operation_name(saver.saver_def.save_tensor_name),  # to work around a bug in TF Java
        "save_path_placeholder": saver.saver_def.filename_tensor_name,
        "default_tensor_value": [_to_floats(v) for v in additional_values]
    }

    with open(os.path.join(folder, "training_meta.json"), "w") as f:
        f.write(json.dumps(meta))

    with gfile.GFile(os.path.join(folder, "model.meta"), "wb") as f:
            f.write(graph.as_graph_def().SerializeToString())

    return meta, saver


def _to_floats(vs):
    return [float(v) for v in vs]


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
        if input_tensor.op.type not in ["Placeholder", "PlaceholderWithDefault"]:
            non_placeholder_input_names.append(input_tensor.name)
            type_enums.append(input_tensor.dtype.as_datatype_enum)

    output_names = [o.name for o in outputs]

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

    nodes_of_graph = []
    for node in optimized_graph_def.node:
        nodes_of_graph.append(node.name + ":0")
    nodes_of_graph_set = set(nodes_of_graph)

    new_input_names = []
    error_input_nodes = []
    for t in inputs:
        if t.name in old_names2new:
            if old_names2new[t.name] not in nodes_of_graph_set:
                error_input_nodes.append("\"" + (t.name)[0:-2] + "\"")
            new_input_names.append(old_names2new[t.name])
        else:
            if t.name not in nodes_of_graph_set:
                error_input_nodes.append("\"" + (t.name)[0:-2] + "\"")
            new_input_names.append(t.name)

    if error_input_nodes:
        error_nodes_name = " and ".join(error_input_nodes)
        raise ValueError("Node %s doesn't exist in the graph" % str(error_nodes_name))

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
        nodes = set([n.name for n in optimized_graph_def.node])
        for v in all_variables:
            if v.op.name in nodes:
                used_variables.append(v.name)

        with tf.Graph().as_default() as g:
            tf.import_graph_def(optimized_graph_def, name='')
            output_tensors = [g.get_tensor_by_name(x) for x in output_names]
            grad_output_placeholders = [tf.placeholder(dtype=x.dtype,
                                                       name=x.name.split(":")[0] + "_grad",
                                                       shape=x.shape) for x in output_tensors]

            variables = [g.get_tensor_by_name(x) for x in used_variables]

            inputs = [g.get_tensor_by_name(x) for x in new_input_names]
            grads = tf.gradients(output_tensors, variables + inputs,
                                 grad_ys=grad_output_placeholders)

            grads = [process_grad(grad) for grad in grads]

            temp_tensors = _find_temp_tensors(grads, nodes)

            grad_variables = [x.name for x in grads[0:len(variables)]]

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


def _find_temp_tensors(grads, forward_ops):
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
