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
import sys
from shutil import copyfile, copytree, rmtree

from setuptools import setup

TEMP_PATH = "zoo/share"
analytics_zoo_home = os.path.abspath(__file__ + "/../../")

try:
    exec(open('zoo/version.py').read())
except IOError:
    print("Failed to load Analytics Zoo version file for packaging. \
      You must be in Analytics Zoo's pyzoo dir.")
    sys.exit(-1)

VERSION = __version__

building_error_msg = """
If you are packing python API from zoo source, you must build Analytics Zoo first
and run sdist.
    To build Analytics Zoo with maven you can run:
      cd $ZOO_HOME
      ./make-dist.sh
    Building the source dist is done in the Python directory:
      cd pyzoo
      python setup.py sdist
      pip install dist/*.tar.gz"""


def build_from_source():
    code_path = analytics_zoo_home + "/pyzoo/zoo/common/nncontext.py"
    print("Checking: %s to see if build from source" % code_path)
    if os.path.exists(code_path):
        return True
    return False


def init_env():
    if build_from_source():
        print("Start to build distributed package")
        print("HOME OF ANALYTICS ZOO: " + analytics_zoo_home)
        dist_source = analytics_zoo_home + "/dist"
        if not os.path.exists(dist_source):
            print(building_error_msg)
            sys.exit(-1)
        if os.path.exists(TEMP_PATH):
            rmtree(TEMP_PATH)
        copytree(dist_source, TEMP_PATH)
        copyfile(analytics_zoo_home + "/pyzoo/zoo/models/__init__.py", TEMP_PATH + "/__init__.py")
    else:
        print("Do nothing for release installation")


def setup_package():
    metadata = dict(
        name='analyticszoo',
        version=VERSION,
        description='An analytics + AI platform for Apache Spark and BigDL',
        author='zoo Authors',
        author_email='zoo-user-group@googlegroups.com',
        license='Apache License, Version 2.0',
        url='https://github.com/intel-analytics/zoo',
        packages=['zoo',
                  'zoo.common',
                  'zoo.examples',
                  'zoo.examples.objectdetection',
                  'zoo.examples.textclassification',
                  'zoo.feature',
                  'zoo.feature.image',
                  'zoo.models',
                  'zoo.models.common',
                  'zoo.models.image',
                  'zoo.models.image.common',
                  'zoo.models.image.objectdetection',
                  'zoo.models.recommendation',
                  'zoo.models.textclassification',
                  'zoo.pipeline',
                  'zoo.pipeline.api',
                  'zoo.pipeline.api.keras',
                  'zoo.pipeline.api.keras.engine',
                  'zoo.pipeline.api.keras.layers',
                  'zoo.pipeline.api.keras.metrics',
                  'zoo.pipeline.nnframes',
                  'zoo.util',
                  'zoo.share',
                  'bigdl',
                  'bigdl.dataset',
                  'bigdl.nn',
                  'bigdl.nn.keras',
                  'bigdl.transform',
                  'bigdl.transform.vision',
                  'bigdl.keras',
                  'bigdl.examples',
                  'bigdl.examples.keras',
                  'bigdl.models',
                  'bigdl.models.lenet',
                  'bigdl.models.local_lenet',
                  'bigdl.models.ml_pipeline',
                  'bigdl.models.rnn',
                  'bigdl.models.textclassifier',
                  'bigdl.optim',
                  'bigdl.util'],
        install_requires=['numpy>=1.7', 'pyspark>=2.2', 'six>=1.10.0'],
        dependency_links=['https://d3kbcqa49mib13.cloudfront.net/spark-2.0.0-bin-hadoop2.7.tgz'],
        include_package_data=True,
        package_dir={"bigdl": '../backend/bigdl/pyspark/bigdl', "zoo.share": TEMP_PATH},
        package_data={"zoo.share": ['lib/analytics-zoo*with-dependencies.jar', 'conf/*', 'bin/*']},
        classifiers=[
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: Implementation :: CPython'],
        platforms=['mac', 'linux']
    )

    setup(**metadata)


if __name__ == '__main__':
    try:
        init_env()
        setup_package()
    except Exception as e:
        raise e
    finally:
        if build_from_source() and os.path.exists(TEMP_PATH):
            rmtree(TEMP_PATH)
