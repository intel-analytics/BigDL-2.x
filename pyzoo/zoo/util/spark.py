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

from pyspark import SparkContext
from zoo import init_nncontext, init_spark_conf
from zoo.util.utils import detect_python_location, pack_penv
from zoo.util.utils import get_executor_conda_zoo_classpath, get_zoo_bigdl_classpath_on_driver


class SparkRunner:
    standalone_env = None

    def __init__(self,
                 spark_log_level="WARN",
                 redirect_spark_log=True):
        self.spark_log_level = spark_log_level
        self.redirect_spark_log = redirect_spark_log
        with SparkContext._lock:
            if SparkContext._active_spark_context:
                raise Exception("There's existing SparkContext. Please close it first.")
        import pyspark
        print("Current pyspark location is : {}".format(pyspark.__file__))

    def create_sc(self, submit_args, spark_conf):
        submit_args = submit_args + " pyspark-shell"
        os.environ["PYSPARK_SUBMIT_ARGS"] = submit_args
        sc = init_nncontext(conf=spark_conf, spark_log_level=self.spark_log_level,
                            redirect_spark_log=self.redirect_spark_log)
        return sc

    def init_spark_on_local(self, cores, conf=None, python_location=None):
        print("Start to getOrCreate SparkContext")
        if "PYSPARK_PYTHON" not in os.environ:
            os.environ["PYSPARK_PYTHON"] = \
                python_location if python_location else detect_python_location()
        master = "local[{}]".format(cores)
        zoo_conf = init_spark_conf(conf).setMaster(master)
        sc = init_nncontext(conf=zoo_conf, spark_log_level=self.spark_log_level,
                            redirect_spark_log=self.redirect_spark_log)
        print("Successfully got a SparkContext")
        return sc

    def init_spark_on_yarn(self,
                           hadoop_conf,
                           conda_name,
                           num_executors,
                           executor_cores,
                           executor_memory="2g",
                           driver_cores=4,
                           driver_memory="1g",
                           extra_executor_memory_for_ray=None,
                           extra_python_lib=None,
                           penv_archive=None,
                           additional_archive=None,
                           hadoop_user_name="root",
                           spark_yarn_archive=None,
                           conf=None,
                           jars=None):
        print("Initializing SparkContext for yarn-client mode")
        executor_python_env = "python_env"
        os.environ["HADOOP_CONF_DIR"] = hadoop_conf
        os.environ["HADOOP_USER_NAME"] = hadoop_user_name
        os.environ["PYSPARK_PYTHON"] = "{}/bin/python".format(executor_python_env)

        pack_env = False
        assert penv_archive or conda_name, \
            "You should either specify penv_archive or conda_name explicitly"
        try:

            if not penv_archive:
                penv_archive = pack_penv(conda_name, executor_python_env)
                pack_env = True

            archive = "{}#{}".format(penv_archive, executor_python_env)
            if additional_archive:
                archive = archive + "," + additional_archive
            submit_args = "--master yarn --deploy-mode client"
            submit_args = submit_args + " --archives {}".format(archive)
            submit_args = submit_args + gen_submit_args(
                driver_cores, driver_memory, num_executors, executor_cores,
                executor_memory, extra_python_lib, jars)

            spark_conf = create_spark_conf(conf, driver_cores, driver_memory, num_executors, executor_cores,
                                           executor_memory, extra_executor_memory_for_ray)
            spark_conf.set("spark.scheduler.minRegisteredResourcesRatio", "1.0")\
                .set("spark.executorEnv.PYTHONHOME", executor_python_env)
            if spark_yarn_archive:
                spark_conf.set("spark.yarn.archive", spark_yarn_archive)
            zoo_bigdl_path_on_executor = ":".join(list(get_executor_conda_zoo_classpath(executor_python_env)))
            if spark_conf.contains("spark.executor.extraClassPath"):
                spark_conf.set("spark.executor.extraClassPath", "{}:{}".format(
                    zoo_bigdl_path_on_executor, spark_conf.get("spark.executor.extraClassPath")))
            else:
                spark_conf.set("spark.executor.extraClassPath", zoo_bigdl_path_on_executor)

            sc = self.create_sc(submit_args, spark_conf)
        finally:
            if conda_name and penv_archive and pack_env:
                os.remove(penv_archive)
        return sc

    def init_spark_standalone(self,
                              num_executors,
                              executor_cores,
                              executor_memory="2g",
                              driver_cores=4,
                              driver_memory="1g",
                              master=None,
                              extra_executor_memory_for_ray=None,
                              extra_python_lib=None,
                              conf=None,
                              jars=None):
        import subprocess
        import pyspark
        from zoo.util.utils import get_node_ip

        if "PYSPARK_PYTHON" not in os.environ:
            os.environ["PYSPARK_PYTHON"] = detect_python_location()
        if not master:
            pyspark_home = os.path.abspath(pyspark.__file__ + "/../")
            zoo_standalone_home = os.path.abspath(__file__ + "/../../share/bin/standalone")
            node_ip = get_node_ip()
            SparkRunner.standalone_env = {
                "SPARK_HOME": pyspark_home,
                "ZOO_STANDALONE_HOME": zoo_standalone_home,
                # If not set this, by default master is hostname but not ip,
                "SPARK_MASTER_HOST": node_ip}
            # The scripts installed from pip don't have execution permission
            # and need to first give them permission.
            pro = subprocess.Popen(["chmod", "-R", "+x", "{}/sbin".format(zoo_standalone_home)])
            os.waitpid(pro.pid, 0)
            # Start master
            start_master_pro = subprocess.Popen(
                "{}/sbin/start-master.sh".format(zoo_standalone_home),
                shell=True, env=SparkRunner.standalone_env)
            os.waitpid(start_master_pro.pid, 0)
            master = "spark://{}:7077".format(node_ip)  # 7077 is the default port
            # Start worker
            start_worker_pro = subprocess.Popen(
                "{}/sbin/start-worker.sh {}".format(zoo_standalone_home, master),
                shell=True, env=SparkRunner.standalone_env)
            os.waitpid(start_worker_pro.pid, 0)
        else:  # A Spark standalone cluster has already been started by the user.
            assert master.startswith("spark://"), \
                "Please input a valid master address for your Spark standalone cluster: " \
                "spark://master:port"

        # Start pyspark-shell
        submit_args = "--master " + master
        submit_args = submit_args + gen_submit_args(
            driver_cores, driver_memory, num_executors, executor_cores,
            executor_memory, extra_python_lib, jars)

        spark_conf = create_spark_conf(conf, driver_cores, driver_memory, num_executors, executor_cores,
                                       executor_memory, extra_executor_memory_for_ray)
        spark_conf.set("spark.cores.max", num_executors * executor_cores)\
            .set("spark.executorEnv.PYTHONHOME", "/".join(detect_python_location().split("/")[:-2]))
        zoo_bigdl_jar_path = ":".join(list(get_zoo_bigdl_classpath_on_driver()))
        if spark_conf.contains("spark.executor.extraClassPath"):
            spark_conf.set("spark.executor.extraClassPath", "{}:{}".format(
                zoo_bigdl_jar_path, spark_conf.get("spark.executor.extraClassPath")))
        else:
            spark_conf.set("spark.executor.extraClassPath", zoo_bigdl_jar_path)

        sc = self.create_sc(submit_args, spark_conf)
        return sc

    @staticmethod
    def stop_spark_standalone():
        import subprocess
        env = SparkRunner.standalone_env
        if not env:
            import pyspark
            pyspark_home = os.path.abspath(pyspark.__file__ + "/../")
            zoo_standalone_home = os.path.abspath(__file__ + "/../../share/bin/standalone")
            pro = subprocess.Popen(["chmod", "-R", "+x", "{}/sbin".format(zoo_standalone_home)])
            os.waitpid(pro.pid, 0)
            env = {"SPARK_HOME": pyspark_home,
                   "ZOO_STANDALONE_HOME": zoo_standalone_home}
        stop_worker_pro = subprocess.Popen(
            "{}/sbin/stop-worker.sh".format(env["ZOO_STANDALONE_HOME"]), shell=True, env=env)
        os.waitpid(stop_worker_pro.pid, 0)
        stop_master_pro = subprocess.Popen(
            "{}/sbin/stop-master.sh".format(env["ZOO_STANDALONE_HOME"]), shell=True, env=env)
        os.waitpid(stop_master_pro.pid, 0)

    def init_spark_on_k8s(self,
                          master,
                          container_image,
                          num_executors,
                          executor_cores,
                          executor_memory="2g",
                          driver_memory="1g",
                          driver_cores=4,
                          extra_executor_memory_for_ray=None,
                          extra_python_lib=None,
                          conf=None,
                          jars=None,
                          python_location=None):
        if python_location:
            os.environ["PYSPARK_PYTHON"] = python_location

        submit_args = "--master " + master + " --deploy-mode client"
        submit_args = submit_args + gen_submit_args(
            driver_cores, driver_memory, num_executors, executor_cores,
            executor_memory, extra_python_lib, jars)

        spark_conf = create_spark_conf(conf, driver_cores, driver_memory, num_executors, executor_cores,
                                       executor_memory, extra_executor_memory_for_ray)
        spark_conf.set("spark.cores.max", num_executors * executor_cores) \
            .set("spark.kubernetes.container.image", container_image)
        # TODO: driver and executor python environments may vary? Thus may not be able to find
        # zoo and bigdl jar on executor directly.
        # Instead of specifying executor extraClassPath, submit the jars on driver via --jars.
        zoo_bigdl_jar_path = ":".join(list(get_zoo_bigdl_classpath_on_driver()))
        if spark_conf.contains("spark.executor.extraClassPath"):
            spark_conf.set("spark.executor.extraClassPath", "{}:{}".format(
                zoo_bigdl_jar_path, conf.get("spark.executor.extraClassPath")))
        else:
            spark_conf.set("spark.executor.extraClassPath", zoo_bigdl_jar_path)

        sc = self.create_sc(submit_args, spark_conf)
        return sc


def gen_submit_args(driver_cores, driver_memory, num_executors, executor_cores, executor_memory,
                    extra_python_lib=None, jars=None):
    submit_args = " --driver-cores {} --driver-memory {} --num-executors {}" \
                  " --executor-cores {} --executor-memory {}" \
        .format(driver_cores, driver_memory, num_executors, executor_cores, executor_memory)
    if extra_python_lib:
        submit_args = submit_args + " --py-files {}".format(extra_python_lib)
    if jars:
        submit_args = submit_args + " --jars {}".format(jars)
    return submit_args


def create_spark_conf(conf, driver_cores, driver_memory, num_executors, executor_cores, executor_memory,
                      extra_executor_memory_for_ray=None):
    spark_conf = init_spark_conf(conf) \
        .set("spark.driver.cores", driver_cores) \
        .set("spark.driver.memory", driver_memory) \
        .set("spark.executor.instances", num_executors) \
        .set("spark.executor.cores", executor_cores) \
        .set("spark.executor.memory", executor_memory)
    if extra_executor_memory_for_ray:
        spark_conf.set("spark.executor.memoryOverhead", extra_executor_memory_for_ray)
    return spark_conf
