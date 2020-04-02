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

import tensorflow as tf

from zoo.util.tf import export_tf
from optparse import OptionParser


def ckpt_to_frozen_graph(options):
    with tf.gfile.GFile(options.pbPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        var_list_name = [node.name + ":0" for node in graph_def.node if node.op in ["Variable", "VariableV2", "VarHandleOp"]]

    # now build the graph in the memory and visualize it
    with tf.Session() as sess:
        graph = tf.get_default_graph()
        tf.import_graph_def(graph_def, name="")

        var_list = [graph.get_tensor_by_name(name) for name in var_list_name]

        for v in var_list:
            tf.add_to_collection(tf.GraphKeys.TRAINABLE_VARIABLES, v)

        saver = tf.train.Saver(var_list)
        saver.restore(sess, options.ckptPath)

        export_tf(sess, options.outputDir, inputs=[graph.get_tensor_by_name(options.inputsName)],
                  outputs=[graph.get_tensor_by_name(options.outputsName)])


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--pbPath", dest="pbPath", default="")
    parser.add_option("--ckptPath", dest="ckptPath", default="")
    parser.add_option("--inputsName", dest="inputsName", default="")
    parser.add_option("--outputsName", dest="outputsName", default="")
    parser.add_option("-o", "--outputDir", dest="outputDir", default=".")
    import sys
    (options, args) = parser.parse_args(sys.argv)
    ckpt_to_frozen_graph(options)
