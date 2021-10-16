#!/usr/bin/env python

#
# Copyright 2016 The BigDL Authors.
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
import fnmatch
from setuptools import setup
import urllib.request
import os
import stat
import sys
import re
from html.parser import HTMLParser


def setup_package():

    metadata = dict(
        name='bigdl-tf',
        version='0.14.0.dev0',
        description='',
        author='',
        author_email='',
        packages=["bigdl.share.tflibs"],
        url='https://github.com/intel-analytics/analytics-zoo/tree/bigdl-2.0',
        package_data={"bigdl.share.tflibs": ["libtensorflow_framework.so",
                                             "libtensorflow_jni.so"]},
        package_dir={'': 'src'},
    )

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
