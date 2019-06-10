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

import re


def to_list(input):
    if isinstance(input, (list, tuple)):
        return list(input)
    else:
        return [input]


def resourceToBytes(resource_str):
    matched = re.compile("([0-9]+)([a-z]+)?").match(resource_str.lower())
    fraction_matched = re.compile("([0-9]+\\.[0-9]+)([a-z]+)?").match(resource_str.lower())
    if fraction_matched:
        raise Exception(
            "Fractional values are not supported. Input was: {}".format(resource_str))
    try:
        value = int(matched.group(1))
        postfix = matched.group(2)
        if postfix == 'b':
            value = value
        elif postfix == 'k':
            value = value * 1000
        elif postfix == "m":
            value = value * 1000 * 1000
        elif postfix == 'g':
            value = value * 1000 * 1000 * 1000
        else:
            raise Exception("Not supported type: {}".format(resource_str))
        return value
    except Exception:
        raise Exception("Size must be specified as bytes(b),"
                        "kilobytes(k), megabytes(m), gigabytes(g). "
                        "E.g. 50b, 100k, 250m, 30g")
