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
import sys
from shutil import copyfile, copytree, rmtree
import fnmatch
from setuptools import setup

bigdl_home = os.path.abspath(__file__ + "/../../../..")
exclude_patterns = ["*__pycache__*", "*ipynb_checkpoints*"]

VERSION = open(os.path.join(bigdl_home, 'python/version.txt'), 'r').read().strip()


def get_bigdl_packages():
    bigdl_python_home = os.path.abspath(__file__ + "/..")
    bigdl_packages = []
    source_dir = os.path.join(bigdl_python_home, "bigdl")
    for dirpath, dirs, files in os.walk(source_dir):
        package = dirpath.split(bigdl_python_home)[1].replace('/', '.')
        if any(fnmatch.fnmatchcase(package, pat=pattern)
                for pattern in exclude_patterns):
            print("excluding", package)
        else:
            bigdl_packages.append(package)
            print("including", package)
    return bigdl_packages


def setup_package():
    metadata = dict(
        name='bigdl-chronos',
        version=VERSION,
        description='Seamlessly scale out TensorFlow and PyTorch for Big Data (using Spark & Ray)',
        author='BigDL Authors',
        author_email='bigdl-user-group@googlegroups.com',
        license='Apache License, Version 2.0',
        url='https://github.com/intel-analytics/analytics-zoo',
        packages=get_bigdl_packages(),
        install_requires=['pyspark==2.4.6', 'conda-pack==0.3.1',
                          'packaging', 'filelock'],
        extras_require={'ray': ['ray==1.2.0', 'psutil', 'aiohttp==3.7.0', 'aioredis==1.1.0',
                                'setproctitle', 'hiredis==1.1.0', 'async-timeout==3.0.1'],
                        'automl': ['tensorflow>=1.15.0,<2.0.0', 'h5py==2.10.0',
                                   'ray[tune]==1.2.0', 'psutil', 'aiohttp==3.7.0',
                                   'aioredis==1.1.0', 'setproctitle', 'hiredis==1.1.0',
                                   'pandas==1.0.3', 'scikit-learn>=0.20.0,<=0.22.0', 'requests',
                                   'scipy==1.5', 'protobuf==3.12.0', 'torch==1.8.1',
                                   'tsfresh==0.17.0']},
        dependency_links=['https://d3kbcqa49mib13.cloudfront.net/spark-2.0.0-bin-hadoop2.7.tgz'],
        include_package_data=True,
        classifiers=[
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: Implementation :: CPython'],
        platforms=['mac', 'linux']
    )

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
