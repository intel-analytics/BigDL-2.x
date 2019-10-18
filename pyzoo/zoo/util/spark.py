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


class SparkRunner(object):
    def __init__(self,
                 spark_log_level="WARN",
                 redirect_spark_log=True):
        self.spark_log_level = spark_log_level
        self.redirect_spark_log = redirect_spark_log
        self.PYTHON_ENV = "python_env"
        with SparkContext._lock:
            if SparkContext._active_spark_context:
                raise Exception("There's existing SparkContext. Please close it first.")
        import pyspark
        print("Current pyspark location is : {}".format(pyspark.__file__))

    def pack_penv(self, conda_name):
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        tmp_path = "{}/{}.tar.gz".format(tmp_dir, self.PYTHON_ENV)
        print("Start to pack current python env")
        pack_conda_env(["--output", tmp_path, "--n-threads", "8", "--name", conda_name])
        print("Packing has been completed: {}".format(tmp_path))
        return tmp_path

    def _create_sc(self, submit_args, conf):
        print("pyspark_submit_args is: {}".format(submit_args))
        os.environ['PYSPARK_SUBMIT_ARGS'] = submit_args
        zoo_conf = init_spark_conf()
        for key, value in conf.items():
            zoo_conf.set(key=key, value=value)
        sc = init_nncontext(conf=zoo_conf, redirect_spark_log=self.redirect_spark_log)
        sc.setLogLevel(self.spark_log_level)

        return sc

    def init_spark_on_local(self, cores, conf=None, python_location=None):
        print("Start to getOrCreate SparkContext")
        os.environ['PYSPARK_PYTHON'] = \
            python_location if python_location else detect_python_location()
        master = "local[{}]".format(cores)
        zoo_conf = init_spark_conf().setMaster(master)
        if conf:
            zoo_conf.setAll(conf.items())
        sc = init_nncontext(conf=zoo_conf, redirect_spark_log=self.redirect_spark_log)
        sc.setLogLevel(self.spark_log_level)
        print("Successfully got a SparkContext")
        return sc

    def init_spark_on_yarn(self,
                           hadoop_conf,
                           conda_name,
                           num_executor,
                           executor_cores,
                           executor_memory="10g",
                           driver_memory="1g",
                           driver_cores=4,
                           extra_executor_memory_for_ray=None,
                           extra_python_lib=None,
                           penv_archive=None,
                           hadoop_user_name="root",
                           spark_yarn_archive=None,
                           spark_conf=None,
                           jars=None):
        os.environ["HADOOP_CONF_DIR"] = hadoop_conf
        os.environ['HADOOP_USER_NAME'] = hadoop_user_name
        os.environ['PYSPARK_PYTHON'] = "{}/bin/python".format(self.PYTHON_ENV)

        def _yarn_opt():
            command = " --archives {}#{} --num-executors {} " \
                      " --executor-cores {} --executor-memory {}". \
                format(penv_archive, self.PYTHON_ENV, num_executor, executor_cores, executor_memory)

            if extra_python_lib:
                command = command + " --py-files {} ".format(extra_python_lib)
            if jars:
                command = command + " --jars {}".format(jars)
            return command

        def _submit_opt():
            conf = {
                "spark.driver.memory": driver_memory,
                "spark.driver.cores": driver_cores,
                "spark.scheduler.minRegisteredResourcesRatio": "1.0"}
            if extra_executor_memory_for_ray:
                conf["spark.executor.memoryOverhead"] = extra_executor_memory_for_ray
            if spark_yarn_archive:
                conf["spark.yarn.archive"] = spark_yarn_archive
            return " --master yarn --deploy-mode client" + _yarn_opt() + ' pyspark-shell ', conf

        pack_env = False
        assert penv_archive or conda_name, \
            "You should either specify penv_archive or conda_name explicitly"
        try:
            if not penv_archive:
                penv_archive = self.pack_penv(conda_name)
                pack_env = True

            submit_args, conf = _submit_opt()

            if not spark_conf:
                spark_conf = {}
            zoo_bigdl_path_on_executor = ":".join(prepare_zoo_classpath_for_executor())

            if "spark.executor.extraClassPath" in spark_conf:
                spark_conf["spark.executor.extraClassPath"] = "{}:{}".format(
                    zoo_bigdl_path_on_executor, spark_conf["spark.executor.extraClassPath"])
            else:
                spark_conf["spark.executor.extraClassPath"] = zoo_bigdl_path_on_executor

            for item in spark_conf.items():
                conf[str(item[0])] = str(item[1])
            sc = self._create_sc(submit_args, conf)
        finally:
            if conda_name and penv_archive and pack_env:
                os.remove(penv_archive)
        return sc

    def init_spark_on_standalone(self,
                                 master,
                                 executor_cores,
                                 total_executor_cores,
                                 executor_memory="10g",
                                 driver_memory="1g",
                                 driver_cores=4,
                                 extra_executor_memory_for_ray=None,
                                 python_location=None,
                                 extra_python_lib=None,
                                 spark_conf=None,
                                 jars=None):
        os.environ['PYSPARK_PYTHON'] = \
            python_location if python_location else detect_python_location()

        def _standalone_opt():
            command = " --total-executor-cores {} " \
                      " --executor-cores {} --executor-memory {}". \
                format(total_executor_cores, executor_cores, executor_memory)

            if extra_python_lib:
                command = command + " --py-files {} ".format(extra_python_lib)
            if jars:
                command = command + " --jars {}".format(jars)
            return command

        def _submit_opt():
            conf = {
                "spark.driver.memory": driver_memory,
                "spark.driver.cores": driver_cores,
                "spark.scheduler.minRegisteredResourcesRatio": "1.0"}
            if extra_executor_memory_for_ray:
                conf["spark.executor.memoryOverhead"] = extra_executor_memory_for_ray
            return " --master " + master + _standalone_opt() + ' pyspark-shell ', conf

        submit_args, conf = _submit_opt()

        if not spark_conf:
            spark_conf = {}
        zoo_bigdl_path_on_executor = ":".join(prepare_zoo_classpath_for_executor())

        if "spark.executor.extraClassPath" in spark_conf:
            spark_conf["spark.executor.extraClassPath"] = "{}:{}".format(
                zoo_bigdl_path_on_executor, spark_conf["spark.executor.extraClassPath"])
        else:
            spark_conf["spark.executor.extraClassPath"] = zoo_bigdl_path_on_executor

        for item in spark_conf.items():
            conf[str(item[0])] = str(item[1])
        sc = self._create_sc(submit_args, conf)
        return sc


def prepare_zoo_classpath_for_executor():
    from bigdl.util.engine import get_bigdl_classpath
    bigdl_classpath = get_bigdl_classpath()
    assert bigdl_classpath, "Cannot find bigdl classpath. Please check your bigdl installation"
    from zoo.util.engine import get_analytics_zoo_classpath
    analytics_zoo_classpath = get_analytics_zoo_classpath()
    assert analytics_zoo_classpath,\
        "Cannot find analytics-zoo classpath. Please check your analytics-zoo installation"
    return [analytics_zoo_classpath, bigdl_classpath]


# This is adopted from conda-pack.
def pack_conda_env(args):
    import sys
    import traceback
    from conda_pack.cli import fail, PARSER, context
    import conda_pack
    from conda_pack import pack, CondaPackException
    args = PARSER.parse_args(args=args)
    # Manually handle version printing to output to stdout in python < 3.4
    if args.version:
        print('conda-pack %s' % conda_pack.__version__)
        sys.exit(0)

    try:
        with context.set_cli():
            pack(name=args.name,
                 prefix=args.prefix,
                 output=args.output,
                 format=args.format,
                 force=args.force,
                 compress_level=args.compress_level,
                 n_threads=args.n_threads,
                 zip_symlinks=args.zip_symlinks,
                 zip_64=not args.no_zip_64,
                 arcroot=args.arcroot,
                 dest_prefix=args.dest_prefix,
                 verbose=not args.quiet,
                 filters=args.filters)
    except CondaPackException as e:
        fail("CondaPackError: %s" % e)
    except KeyboardInterrupt:
        fail("Interrupted")
    except Exception:
        fail(traceback.format_exc())


def detect_python_location():
    # TODO: add python_location for standalone
    import subprocess
    pro = subprocess.Popen(
        "command -v python",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    out, err = pro.communicate()
    out = out.decode("utf-8")
    err = err.decode("utf-8")
    error_code = pro.returncode
    if 0 != error_code:
        raise Exception(err +
                        "Cannot detect current python location."
                        "Please set it manually by python_location")
    return out.strip()
