#!/usr/bin/env python

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

import os
import re
from os.path import isfile, join

scala_layers_dirs = ["./zoo/src/main/scala/com/intel/analytics/zoo/pipeline/api/keras/layers",
                     "./zoo/src/main/scala/com/intel/analytics/zoo/pipeline/api/keras/models",
                     "./zoo/src/main/scala/com/intel/analytics/zoo/pipeline/api/autograd"]
python_layers_dirs = ["./pyzoo/zoo/pipeline/api/keras/layers",
                      "./pyzoo/zoo/pipeline/api/keras/engine",
                      "./pyzoo/zoo/pipeline/api/keras/",
                      "./pyzoo/zoo/pipeline/api/"]

scala_to_python = {"CustomLossWithVariable": "CustomLoss"}
scala_class_to_path = {}


def extract_scala_class(class_path):
    exclude_key_words = {"KerasLayerWrapper", "LambdaTorch", "CustomLossWithFunc", "abstract",
                         "InternalRecurrent", "InternalTimeDistributed", "InternalMM", "Recurrent",
                         "InternalLocalOptimizer", "InternalDistriOptimizer",
                         "EmbeddingMatrixHolder", "InternalParameter", "KerasParameter",
                         "InternalCAddTable", "InternalGetShape",
                         "EmbeddingMatrixHolder", "Pooling2D", "InternalSplitTensor",
                         "SplitTensor", "Expand", "InternalMax", "InternalConvLSTM3D",
                         "InternalConvLSTM2D", "InternalCMulTable", "SoftMax", "TransformerLayer",
                         "InternalConvLSTM2D", "InternalCMulTable", "SoftMax",
                         "KerasConstant", "InternalConstant"}
    content = "\n".join([line for line in open(class_path).readlines()
                         if all([key not in line for key in exclude_key_words])])
    match = re.findall(r"class ([\w]+)[^{]+", content)
    return match


def get_all_scala_layers(scala_dirs):
    results = set()
    raw_result = []
    for scala_dir in scala_dirs:
        for name in os.listdir(scala_dir):
            if isfile(join(scala_dir, name)):
                res = extract_scala_class(join(scala_dir, name))
                raw_result += res
                for item in res:
                    scala_class_to_path[item] = join(scala_dir, name)
        results.update(set(class_name for class_name in raw_result if class_name is not None))
    return results


def get_python_classes(python_dirs):
    exclude_classes = {"InputLayer", "ZooKerasLayer", "ZooKerasCreator",
                       "KerasNet", "Net"}
    raw_classes = []
    results = []
    for python_dir in python_dirs:
        python_files = [join(python_dir, name) for name in os.listdir(python_dir)
                        if isfile(join(python_dir, name)) and name.endswith('py')
                        and "__" not in name]
        results += python_files
    for p in results:
        with open(p) as f:
            raw_classes.extend([line for line in f.readlines() if line.startswith("class")])
    classes = [name.split()[1].split("(")[0]for name in raw_classes]
    return set([name for name in classes if name not in exclude_classes])


scala_layers = get_all_scala_layers(scala_layers_dirs)
python_layers = get_python_classes(python_layers_dirs)

# print("Layers in Scala: {0}, {1}".format(len(scala_layers), scala_layers))
# print("")
# print("Layers in Python: {0}, {1}".format(len(python_layers), python_layers))

print("Layers in Scala but not in Python: "),
diff_count = 0
for name in scala_layers:
    if name not in python_layers:
        if name not in scala_to_python or \
                (name in scala_to_python and scala_to_python[name] not in python_layers):
            print("{} : {}".format(name, scala_class_to_path[name]))
            diff_count += 1

if diff_count > 0:
    raise Exception("There exist layers in Scala but not wrapped in Python")
