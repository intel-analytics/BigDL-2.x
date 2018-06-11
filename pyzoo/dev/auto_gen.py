#!/usr/bin/env bash

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

# NB: This is just a helper script and might be removed shortly.
import re


def get_class_name(src_code):
    import re
    match = re.search(r"object(.*)\{", src_code)
    return match.group(1).strip()


def get_parameters(src_code):
    match = re.search(r"def apply.*ClassTag\]\s*\((.*)\)\s*\(implicit", src_code, re.DOTALL)
    params = [p.strip() for p in match.group(1).strip().split("\n") if len(p.strip()) > 0]
    names = []
    values = []
    types = []
    for p in params:
        if "=" in p:  # with default value
            match = re.search(r"(.*)\:(.*)=([^\,]*)", p)
            param_value = match.group(3).strip()
        else:
            match = re.search(r"(.*)\:([^\,]*)", p)
            param_value = None
        param_name = match.group(1).strip()
        param_type = match.group(2).strip()
        names.append(param_name)
        values.append(param_value)
        types.append(param_type)
    return zip(names, values, types)


# wRegularizer
def to_py_name(scala_name):
    name_mapping = {"w_regularizer": "W_regularizer"}
    result = []
    previous_is_lower = False
    for c in scala_name:
        if c.isupper() and previous_is_lower:
            result.append("_")
            previous_is_lower = False
        else:
            previous_is_lower = True
        result.append(c.lower())
    tmp_result = "".join(result)
    return name_mapping.get(tmp_result, tmp_result)


# print(to_py_name("wRegularizer"))

def to_py_value(scala_value):
    name_mapping = {"true": "True", "false": "False", "null": "None",
                    "RandomUniform": "\"glorot_uniform\""}
    return name_mapping.get(scala_value, scala_value)


#       wRegularizer: Regularizer[T] = null,
def to_py_param(scala_param):
    param_name, param_value, param_type = scala_param
    if param_value:
        return to_py_name(param_name) + "=" + to_py_value(param_value)
    else:
        return to_py_name(param_name)


def to_py_params(scala_params):
    result = []
    for param in scala_params:
        result.append(to_py_param(param))
    return result


def append_semi(result_list):
    result = []
    for index, r in enumerate(result_list):
        if (index < len(result_list) - 1):
            result.append(r + ",")
        else:
            result.append(r)
    return "\n".join(result)


def format_py_params(py_params):
    return append_semi([8 * " " + param for param in py_params])


def format_py_params_for_value(py_params):
    mapping = {"init": "to_bigdl_init(init)",
               "activation": "get_activation_by_name(activation) if activation else None",
               "W_regularizer": "to_bigdl_reg(W_regularizer)",
               "b_regularizer": "to_bigdl_reg(b_regularizer)",
               "input_shape": "list(input_shape) if input_shape else None"}
    py_param_names = [p.split("=")[0].strip() for p in py_params]
    return append_semi([12 * " " + mapping.get(name, name) for name in py_param_names])


def to_py_constructor(scala_src):
    class_name = get_class_name(scala_src)
    scala_params = get_parameters(scala_src)
    py_params = to_py_params(scala_params)

    print("")
    doc_test = "  "
    init_content = """
        super(%s, self).__init__(None, bigdl_type, \n%s)
    """ % (class_name, format_py_params_for_value(py_params))
    result = []
    result.append("class %s(Layer):" % class_name)
    result.append(4 * " " + "\'''%s\'''" % doc_test)
    result.append(
        4 * " " + """def __init__(self, \n%s,bigdl_type="float"):""" % format_py_params(py_params))
    result.append(init_content)
    return "\n".join(result)


def to_java_creator(scala_src):
    class_name = get_class_name(scala_src)
    scala_params = get_parameters(scala_src)

    def format_creator_param_list(scala_params):
        mapping = {"inputShape": 8 * " " + "inputShape: JList[Int] = null"}
        result = []
        for name, value, t in scala_params:
            if name in mapping:
                result.append(mapping[name])
            else:
                result.append(8 * " " + """%s: %s%s""" % (name, t, " = " + value if value else ""))
        return append_semi(result)

    def format_init_list(scala_params):
        mapping = {"inputShape": """toScalaShape(inputShape)"""}
        result = []
        for name, value, t in scala_params:
            if name in mapping:
                result.append(mapping[name])
            else:
                result.append(name)
        return append_semi([12 * " " + i for i in result])

    result = []
    formated_creator_param_list = format_creator_param_list(scala_params)
    formated_init_list = format_init_list(scala_params)
    result.append("""
    def create%s(\n%s): %s[T] = {
    %s(\n%s)
    }
    """ % (class_name, formated_creator_param_list, class_name, 4 * " " + class_name,
           formated_init_list))
    return "\n".join(result)


if __name__ == "__main__":
    from optparse import OptionParser
    import sys
    import os

    cur_path = os.path.dirname(os.path.realpath(__file__))

    keras_dir = os.path.join(
        cur_path,
        "../../zoo/src/main/scala/com/intel/analytics/zoo/pipeline/api/keras2/layers/")
    parser = OptionParser()
    parser.add_option("-l", "--layer", dest="layer", default="Dense")
    (options, args) = parser.parse_args(sys.argv)
    src_path = os.path.join(keras_dir, options.layer + ".scala")
    src_code = "\n".join(open(src_path).readlines())
    # print(dense)
    print("--------->Python code<-----------------")
    print(to_py_constructor(src_code))
    print("--------->Java code<-----------------")
    print(to_java_creator(src_code))
