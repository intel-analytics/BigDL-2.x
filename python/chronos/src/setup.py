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

long_description = '''
BigDL Chronos is an application framework for building a fast, accurate
 and scalable time series analysis application.

See [here](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/chronos.html)
 for more information.
'''

bigdl_home = os.path.abspath(__file__ + "/../../../..")
exclude_patterns = ["*__pycache__*", "*ipynb_checkpoints*"]

VERSION = open(os.path.join(bigdl_home, 'python/version.txt'), 'r').read().strip()


def get_bigdl_packages():
    bigdl_python_home = os.path.abspath(__file__ + "/..")
    bigdl_packages = []
    source_dir = os.path.join(bigdl_python_home, "bigdl")
    for dirpath, dirs, files in os.walk(source_dir):
        package = dirpath.split(bigdl_python_home)[1].replace(os.sep, '.')
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
        description='Scalable time series analysis using AutoML',
        long_description=long_description,
        long_description_content_type="text/markdown",
        author='BigDL Authors',
        author_email='bigdl-user-group@googlegroups.com',
        license='Apache License, Version 2.0',
        url='https://github.com/intel-analytics/BigDL',
        packages=get_bigdl_packages(),
        install_requires=['pandas>=1.0.5, <1.3.0', 'scikit-learn',
                          'bigdl-nano==' + VERSION, 'numpy<=1.23.5'],
        extras_require={'pytorch': ['bigdl-nano[pytorch]==' + VERSION],
                        'tensorflow': ['bigdl-nano[tensorflow_27]=='+VERSION],
                        'automl': ['optuna<=2.10.1', 'configspace<=0.5.0', 'SQLAlchemy<=1.4.27'],
                        'distributed:platform_system!="Windows"': ['bigdl-orca[automl]=='+VERSION],
                        'inference': ['bigdl-nano[inference]==' + VERSION],
                        'all': ['bigdl-nano[pytorch]==' + VERSION,
                                'bigdl-nano[tensorflow_27]=='+VERSION,
                                'optuna<=2.10.1', 'configspace<=0.5.0',
                                'bigdl-orca[automl]==' + VERSION + ';platform_system!="Windows"',
                                'pmdarima==1.8.5',
                                'prophet==1.1.0',
                                'tsfresh==0.17.0',
                                'pyarrow==6.0.1',
                                'matplotlib',
                                'bigdl-nano[inference]==' + VERSION]},
        dependency_links=['https://d3kbcqa49mib13.cloudfront.net/spark-2.0.0-bin-hadoop2.7.tgz'],
        include_package_data=True,
        entry_points={
            "console_scripts": [
                "benchmark-chronos=bigdl.chronos.benchmark.benchmark_chronos:main",
            ]
        },
        classifiers=[
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: Implementation :: CPython'],
        platforms=['mac', 'linux', 'windows']
    )

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
