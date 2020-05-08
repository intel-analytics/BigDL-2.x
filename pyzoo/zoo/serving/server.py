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
import subprocess

import shutil
import glob
import os
import urllib.request
from zoo.util.engine import get_analytics_zoo_classpath


class ClusterServing:

    def __init__(self):
        self.name = 'cluster-serving'
        self.proc = None
        self.conf_path = os.path.abspath(
            __file__ + "/../../share/bin/cluster-serving/config.yaml")
        self.zoo_jar = 'zoo.jar'
        self.bigdl_jar = 'bigdl.jar'
        # self.spark_redis_jar = 'spark-redis-2.4.0-jar-with-dependencies.jar'

        # self.download_spark_redis_jar()
        self.copy_config()

        self.copy_zoo_jar()

        if not os.path.exists('model'):
            os.mkdir('model')

    def try_copy_bigdl_jar(self):
        try:
            from bigdl.util.engine import get_bigdl_classpath
            shutil.copyfile(get_bigdl_classpath(), self.bigdl_jar)

        except Exception:
            print("WARNING: if you are running Cluster Serving using pip, you have misconfig"
                  "with bigdl python package, otherwise, ignore this WARNING.")

    def copy_zoo_jar(self):
        jar_path = get_analytics_zoo_classpath()
        if jar_path:
            self.try_copy_bigdl_jar()
        else:
            """
            not install by pip, so run prepare_env here
            """
            build_jar_paths = glob.glob(os.path.abspath(

                __file__ + "/../../../../dist/lib/*.jar"))
            prebuilt_jar_paths = glob.glob(os.path.abspath(
                __file__ + "/../../../../*.jar"))
            jar_paths = build_jar_paths + prebuilt_jar_paths

            assert len(jar_paths) > 0, "No zoo jar is found"
            assert len(jar_paths) == 1, "Expecting one jar: %s" % len(jar_paths)
            jar_path = jar_paths[0]
        shutil.copyfile(jar_path, self.zoo_jar)

    def download_spark_redis_jar(self):
        if not os.path.exists(self.spark_redis_jar):
            print("Downloading spark-redis dependency...")
            urllib.request.urlretrieve('https://oss.sonatype.org/content/repositories/'
                                       'public/com/redislabs/spark-redis/2.4.0/'
                                       + self.spark_redis_jar,
                                       self.spark_redis_jar)
        else:
            print("spark-redis jar already exist.")

    def copy_config(self):
        print("Trying to find config file in ", self.conf_path)
        if not os.path.exists(self.conf_path):
            print('WARNING: Config file does not exist in your pip directory,'
                  'are you sure that you install serving by pip?')
            build_conf_path = glob.glob(os.path.abspath(
                __file__ + "/../../../../scripts/cluster-serving/config.yaml"))
            prebuilt_conf_path = glob.glob(os.path.abspath(
                __file__ + "/../../../../../bin/cluster-serving/config.yaml"))
            conf_paths = build_conf_path + prebuilt_conf_path

            assert len(conf_paths) > 0, "No config file is found"
            self.conf_path = conf_paths[0]
            print("config path is found at ", self.conf_path)

            if not os.path.exists(self.conf_path):
                raise EOFError("Can not find your config file.")
        try:
            shutil.copyfile(self.conf_path, 'config.yaml')
        except Exception as e:
            print(e)
            print("WARNING: An initialized config file already exists.")

        subprocess.Popen(['chmod', 'a+x', self.conf_path])
