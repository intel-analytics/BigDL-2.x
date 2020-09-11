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
# Some portions of this file Copyright (c) 2017, Maxpoint
# and licensed under the BSD license.
#

from bigdl.util.common import *
from zoo.common.utils import callZooFunc
import warnings
import multiprocessing
import os

import threading
import sys


def init_spark_on_local(cores=2, conf=None, python_location=None, spark_log_level="WARN",
                        redirect_spark_log=True):
    """
    Create a SparkContext with Zoo configuration in local machine.
    :param cores: The number of cores for Spark local. Default to be 2. You can also set it to "*"
           to use all the available cores. i.e `init_spark_on_local(cores="*")`
    :param conf: You can append extra conf for Spark in key-value format.
           i.e conf={"spark.executor.extraJavaOptions": "-XX:+PrintGCDetails"}.
           Default to be None.
    :param python_location: The path to your running Python executable. If not specified, the
           default Python interpreter in effect would be used.
    :param spark_log_level: The log level for Spark. Default to be 'WARN'.
    :param redirect_spark_log: Whether to redirect the Spark log to local file. Default to be True.
    :return: An instance of SparkContext.
    """
    from zoo.util.spark import SparkRunner
    runner = SparkRunner(spark_log_level=spark_log_level,
                         redirect_spark_log=redirect_spark_log)
    return runner.init_spark_on_local(cores=cores, conf=conf,
                                      python_location=python_location)


def init_spark_on_yarn(hadoop_conf,
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
                       spark_log_level="WARN",
                       redirect_spark_log=True,
                       jars=None,
                       conf=None):
    """
    Create a SparkContext with Analytics Zoo configurations on Yarn cluster for yarn-client mode.
    You only need to create a conda environment and install the python dependencies in that
    environment beforehand on the driver machine. These dependencies would be automatically
    packaged and distributed to the whole Yarn cluster.

    :param hadoop_conf: The path to the yarn configuration folder.
    :param conda_name: The name of the conda environment.
    :param num_executors: The number of Spark executors.
    :param executor_cores: The number of cores for each executor.
    :param executor_memory: The memory for each executor. Default to be '2g'.
    :param driver_cores: The number of cores for the Spark driver. Default to be 4.
    :param driver_memory: The memory for the Spark driver. Default to be '1g'.
    :param extra_executor_memory_for_ray: The extra memory for Ray services. Default to be None.
    :param extra_python_lib: Extra python files or packages needed for distribution.
           Default to be None.
    :param penv_archive: Ideally, the program would auto-pack the conda environment specified by
           'conda_name', but you can also pass the path to a packed file in "tar.gz" format here.
           Default to be None.
    :param additional_archive: Comma-separated list of additional archives to be uploaded and
           unpacked on executors. Default to be None.
    :param hadoop_user_name: The user name for running the yarn cluster. Default to be 'root'.
    :param spark_yarn_archive: Conf value for setting spark.yarn.archive. Default to be None.
    :param spark_log_level: The log level for Spark. Default to be 'WARN'.
    :param redirect_spark_log: Whether to redirect the Spark log to local file. Default to be True.
    :param jars: Comma-separated list of jars to be included on driver and executor's classpath.
           Default to be None.
    :param conf: You can append extra conf for Spark in key-value format.
           i.e conf={"spark.executor.extraJavaOptions": "-XX:+PrintGCDetails"}.
           Default to be None.

    :return: An instance of SparkContext.
    """
    from zoo.util.spark import SparkRunner
    runner = SparkRunner(spark_log_level=spark_log_level,
                         redirect_spark_log=redirect_spark_log)
    sc = runner.init_spark_on_yarn(
        hadoop_conf=hadoop_conf,
        conda_name=conda_name,
        num_executors=num_executors,
        executor_cores=executor_cores,
        executor_memory=executor_memory,
        driver_cores=driver_cores,
        driver_memory=driver_memory,
        extra_executor_memory_for_ray=extra_executor_memory_for_ray,
        extra_python_lib=extra_python_lib,
        penv_archive=penv_archive,
        additional_archive=additional_archive,
        hadoop_user_name=hadoop_user_name,
        spark_yarn_archive=spark_yarn_archive,
        jars=jars,
        conf=conf)
    return sc


def init_spark_standalone(num_executors,
                          executor_cores,
                          executor_memory="2g",
                          driver_cores=4,
                          driver_memory="1g",
                          master=None,
                          extra_executor_memory_for_ray=None,
                          extra_python_lib=None,
                          spark_log_level="WARN",
                          redirect_spark_log=True,
                          conf=None,
                          jars=None,
                          python_location=None,
                          enable_numa_binding=False):
    """
    Create a SparkContext with Analytics Zoo configurations on Spark standalone cluster of
    a single node.
    By default, a new Spark standalone cluster would be started first and the SparkContext
    would use its master address. You need to call `stop_spark_standalone` after your program
    finishes to shutdown the cluster.
    You can also specify spark_master if you have already started a standalone cluster.

    :param num_executors: The number of Spark executors.
    :param executor_cores: The number of cores for each executor.
    :param executor_memory: The memory for each executor. Default to be '2g'.
    :param driver_cores: The number of cores for the Spark driver. Default to be 4.
    :param driver_memory: The memory for the Spark driver. Default to be '1g'.
    :param master: The master URL of an existing Spark standalone cluster: 'spark://master:port'.
    You only need to specify this if you have already started a standalone cluster.
    Default to be None and a new standalone cluster would be started in this case.
    :param extra_executor_memory_for_ray: The extra memory for Ray services. Default to be None.
    :param extra_python_lib: Extra python files or packages needed for distribution.
           Default to be None.
    :param spark_log_level: The log level for Spark. Default to be 'WARN'.
    :param redirect_spark_log: Whether to redirect the Spark log to local file. Default to be True.
    :param jars: Comma-separated list of jars to be included on driver and executor's classpath.
           Default to be None.
    :param conf: You can append extra conf for Spark in key-value format.
           i.e conf={"spark.executor.extraJavaOptions": "-XX:+PrintGCDetails"}.
           Default to be None.
    :param python_location: The path to your running Python executable. If not specified, the
           default Python interpreter in effect would be used.
    :param enable_numa_binding: Whether to use numactl to start spark worker in order to bind
           different worker processes to different cpus and memory areas. This is may lead to
           better performance on a multi-sockets machine. Defaults to False.

    :return: An instance of SparkContext.
    """
    from zoo.util.spark import SparkRunner
    runner = SparkRunner(spark_log_level=spark_log_level,
                         redirect_spark_log=redirect_spark_log)
    sc = runner.init_spark_standalone(
        num_executors=num_executors,
        executor_cores=executor_cores,
        executor_memory=executor_memory,
        driver_cores=driver_cores,
        driver_memory=driver_memory,
        master=master,
        extra_executor_memory_for_ray=extra_executor_memory_for_ray,
        extra_python_lib=extra_python_lib,
        conf=conf,
        jars=jars,
        python_location=python_location,
        enable_numa_binding=enable_numa_binding)
    return sc


def init_spark_on_k8s(master,
                      container_image,
                      num_executors,
                      executor_cores,
                      executor_memory="2g",
                      driver_memory="1g",
                      driver_cores=4,
                      extra_executor_memory_for_ray=None,
                      extra_python_lib=None,
                      spark_log_level="WARN",
                      redirect_spark_log=True,
                      jars=None,
                      conf=None,
                      python_location=None):
    """
    Create a SparkContext with Analytics Zoo configurations on Kubernetes cluster for k8s client
    mode. You are recommended to use the Docker image intelanalytics/hyperzoo:latest.
    You can refer to https://github.com/intel-analytics/analytics-zoo/tree/master/docker/hyperzoo
    to build your own Docker image.

    :param master: The master address of your k8s cluster.
    :param container_image: The name of the docker container image for Spark executors.
           For example, intelanalytics/hyperzoo:latest
    :param num_executors: The number of Spark executors.
    :param executor_cores: The number of cores for each executor.
    :param executor_memory: The memory for each executor. Default to be '2g'.
    :param driver_cores: The number of cores for the Spark driver. Default to be 4.
    :param driver_memory: The memory for the Spark driver. Default to be '1g'.
    :param extra_executor_memory_for_ray: The extra memory for Ray services. Default to be None.
    :param extra_python_lib: Extra python files or packages needed for distribution.
           Default to be None.
    :param spark_log_level: The log level for Spark. Default to be 'WARN'.
    :param redirect_spark_log: Whether to redirect the Spark log to local file. Default to be True.
    :param jars: Comma-separated list of jars to be included on driver and executor's classpath.
           Default to be None.
    :param conf: You can append extra conf for Spark in key-value format.
           i.e conf={"spark.executor.extraJavaOptions": "-XX:+PrintGCDetails"}.
           Default to be None.
    :param python_location: The path to your running Python executable. If not specified, the
           default Python interpreter in effect would be used.

    :return: An instance of SparkContext.
    """
    from zoo.util.spark import SparkRunner
    runner = SparkRunner(spark_log_level=spark_log_level,
                         redirect_spark_log=redirect_spark_log)
    sc = runner.init_spark_on_k8s(
        master=master,
        container_image=container_image,
        num_executors=num_executors,
        executor_cores=executor_cores,
        executor_memory=executor_memory,
        driver_memory=driver_memory,
        driver_cores=driver_cores,
        extra_executor_memory_for_ray=extra_executor_memory_for_ray,
        extra_python_lib=extra_python_lib,
        jars=jars,
        conf=conf,
        python_location=python_location)
    return sc


def stop_spark_standalone():
    """
    Stop the Spark standalone cluster created from init_spark_standalone (master not specified).
    """
    from zoo.util.spark import SparkRunner
    SparkRunner.stop_spark_standalone()


class ZooContextMeta(type):

    _log_output = False

    @property
    def log_output(cls):
        """
        Whether to redirect Spark driver JVM's stdout and stderr to the current
        python process. This is useful when running Analytics Zoo in jupyter notebook.
        Default to be False. Needs to be set before initializing SparkContext.
        """
        return cls._log_output

    @log_output.setter
    def log_output(cls, value):
        if SparkContext._active_spark_context is not None:
            raise AttributeError("log_output cannot be set after SparkContext is created."
                                 " Please set it before init_nncontext, init_spark_on_local"
                                 "or init_spark_on_yarn")
        assert isinstance(value, bool), "log_output should either be True or False"
        cls._log_output = value


class ZooContext(metaclass=ZooContextMeta):
    pass


# The following function copied from
# https://github.com/Valassis-Digital-Media/spylon-kernel/blob/master/
# spylon_kernel/scala_interpreter.py
def _read_stream(fd, fn):
    """Reads bytes from a file descriptor, utf-8 decodes them, and passes them
    to the provided callback function on the next IOLoop tick.
    Assumes fd.read will block and should be used in a thread.
    Parameters
    ----------
    fd : file
        File descriptor to read
    fn : callable(str) -> None
        Callback function that handles chunks of text
    """
    while True:
        # Specify a max read size so the read doesn't block indefinitely
        # Using a value less than the typical default max pipe size
        # and greater than a single system page.
        buff = fd.read(8192)
        if buff:
            fn(buff.decode('utf-8'))


def init_nncontext(conf=None, spark_log_level="WARN", redirect_spark_log=True):
    """
    Creates or gets a SparkContext with optimized configurations for BigDL performance.
    This method will also initialize the BigDL engine.

    Note: If you use spark-shell or Jupyter notebook, as the SparkContext is created
    before your code, you have to set the Spark configurations through command line options
    or the properties file before calling this method. In this case, you are recommended
    to use the launching scripts under `analytics-zoo/scripts`.

    :param conf: An instance of SparkConf. If not specified, a new SparkConf with
           Analytics Zoo and BigDL configurations would be created and used.
           You can also input a string here to indicate the name of the application.
    :param spark_log_level: The log level for Spark. Default to be 'WARN'.
    :param redirect_spark_log: Whether to redirect the Spark log to local file. Default to be True.

    :return: An instance of SparkContext.
    """
    # The following code copied and modified from
    # https://github.com/Valassis-Digital-Media/spylon-kernel/blob/master/
    # spylon_kernel/scala_interpreter.py
    if ZooContext.log_output:
        import subprocess
        import pyspark.java_gateway
        spark_jvm_proc = None

        def Popen(*args, **kwargs):
            """Wraps subprocess.Popen to force stdout and stderr from the child process
            to pipe to this process without buffering.
            """
            nonlocal spark_jvm_proc
            # Override these in kwargs to avoid duplicate value errors
            # Set streams to unbuffered so that we read whatever bytes are available
            # when ready, https://docs.python.org/3.6/library/subprocess.html#popen-constructor
            kwargs['bufsize'] = 0
            # Capture everything from stdout for display in the notebook
            kwargs['stdout'] = subprocess.PIPE
            # Optionally capture stderr, otherwise it'll go to the kernel log
            kwargs['stderr'] = subprocess.PIPE
            spark_jvm_proc = subprocess.Popen(*args, **kwargs)
            return spark_jvm_proc

        pyspark.java_gateway.Popen = Popen

    if isinstance(conf, six.string_types):
        sc = getOrCreateSparkContext(conf=None, appName=conf)
    else:
        sc = getOrCreateSparkContext(conf=conf)
    sc.setLogLevel(spark_log_level)

    if ZooContext.log_output:
        if spark_jvm_proc.stdout is not None:
            stdout_reader = threading.Thread(target=_read_stream,
                                             daemon=True,
                                             kwargs=dict(
                                                 fd=spark_jvm_proc.stdout,
                                                 fn=sys.stdout.write))
            stdout_reader.start()
        if spark_jvm_proc.stderr is not None:
            stderr_reader = threading.Thread(target=_read_stream,
                                             daemon=True,
                                             kwargs=dict(
                                                 fd=spark_jvm_proc.stderr,
                                                 fn=sys.stderr.write))
            stderr_reader.start()
    check_version()
    if redirect_spark_log:
        redire_spark_logs()
        show_bigdl_info_logs()
    init_engine()
    return sc


def getOrCreateSparkContext(conf=None, appName=None):
    """
    Get the current active SparkContext or create a new SparkContext.
    :param conf: An instance of SparkConf. If not specified, a new SparkConf with
           Analytics Zoo and BigDL configurations would be created and used.
    :param appName: The name of the application if any.

    :return: An instance of SparkContext.
    """
    with SparkContext._lock:
        if SparkContext._active_spark_context is None:
            spark_conf = init_spark_conf() if conf is None else conf
            if appName:
                spark_conf.setAppName(appName)
            return SparkContext.getOrCreate(spark_conf)
        else:
            return SparkContext.getOrCreate()


def get_analytics_zoo_conf():
    zoo_conf_file = "spark-analytics-zoo.conf"
    zoo_python_wrapper = "python-api.zip"

    for p in sys.path:
        if zoo_conf_file in p and os.path.isfile(p):
            with open(p) if sys.version_info < (3,) else open(p, encoding='latin-1') as conf_file:
                return load_conf(conf_file.read())
        if zoo_python_wrapper in p and os.path.isfile(p):
            import zipfile
            with zipfile.ZipFile(p, 'r') as zip_conf:
                if zoo_conf_file in zip_conf.namelist():
                    content = zip_conf.read(zoo_conf_file)
                    if sys.version_info >= (3,):
                        content = str(content, 'latin-1')
                    return load_conf(content)
    return {}


def init_env(conf):
    # Default env
    kmp_affinity = "granularity=fine,compact,1,0"
    kmp_settings = "1"
    omp_num_threads = "1"
    kmp_blocktime = "0"

    # Check env and override if necessary
    # Currently, focused on ZOO_NUM_MKLTHREADS,
    # OMP_NUM_THREADS, KMP_BLOCKTIME, KMP_AFFINITY
    # and KMP_SETTINGS
    if "KMP_AFFINITY" in os.environ:
        kmp_affinity = os.environ["KMP_AFFINITY"]
    if "KMP_SETTINGS" in os.environ:
        kmp_settings = os.environ["KMP_SETTINGS"]
    if "ZOO_NUM_MKLTHREADS" in os.environ:
        if os.environ["ZOO_NUM_MKLTHREADS"].lower() == "all":
            omp_num_threads = conf.get('spark.executor.cores', str(multiprocessing.cpu_count()))
        else:
            omp_num_threads = os.environ["ZOO_NUM_MKLTHREADS"]
    elif "OMP_NUM_THREADS" in os.environ:
        omp_num_threads = os.environ["OMP_NUM_THREADS"]
    if "KMP_BLOCKTIME" in os.environ:
        kmp_blocktime = os.environ["KMP_BLOCKTIME"]

    # Set env
    conf.set("spark.executorEnv.KMP_AFFINITY", kmp_affinity)
    conf.set("spark.executorEnv.KMP_SETTINGS", kmp_settings)
    conf.set("spark.executorEnv.KMP_BLOCKTIME", kmp_blocktime)
    conf.set("spark.executorEnv.OMP_NUM_THREADS", omp_num_threads)
    os.environ["KMP_AFFINITY"] = kmp_affinity
    os.environ["KMP_SETTINGS"] = kmp_settings
    os.environ["OMP_NUM_THREADS"] = omp_num_threads
    os.environ["KMP_BLOCKTIME"] = kmp_blocktime


def init_spark_conf(conf=None):
    spark_conf = SparkConf()
    if conf:
        spark_conf.setAll(conf.items())
    init_env(spark_conf)
    zoo_conf = get_analytics_zoo_conf()
    # Set bigDL and TF conf
    spark_conf.setAll(zoo_conf.items())

    if os.environ.get("BIGDL_JARS", None) and not is_spark_below_2_2():
        if 'PYSPARK_SUBMIT_ARGS' in os.environ:
            submit_args = os.environ['PYSPARK_SUBMIT_ARGS']
            start = submit_args.find('pyspark-shell')
            submit_args = '{}--driver-class-path {} {}'.format(submit_args[:start],
                                                               os.environ["BIGDL_JARS"],
                                                               submit_args[start:])
        else:
            submit_args = " --driver-class-path " + os.environ["BIGDL_JARS"] + " pyspark-shell "
        print("pyspark_submit_args is: {}".format(submit_args))
        os.environ['PYSPARK_SUBMIT_ARGS'] = submit_args

    # add content in PYSPARK_FILES in spark.submit.pyFiles
    # This is a workaround for current Spark on k8s
    python_lib = os.environ.get('PYSPARK_FILES', None)
    if python_lib:
        existing_py_files = spark_conf.get("spark.submit.pyFiles")
        if existing_py_files:
            spark_conf.set(key="spark.submit.pyFiles",
                           value="%s,%s" % (python_lib, existing_py_files))
        else:
            spark_conf.set(key="spark.submit.pyFiles", value=python_lib)

    return spark_conf


def check_version():
    sc = getOrCreateSparkContext()
    conf = sc._conf
    if conf.get("spark.analytics.zoo.versionCheck", "False").lower() == "true":
        report_warn = conf.get(
            "spark.analytics.zoo.versionCheck.warning", "False").lower() == "true"
        _check_spark_version(sc, report_warn)


def _split_full_version(version):
    parts = version.split(".")
    major = parts[0]
    feature = parts[1]
    maintenance = parts[2]
    return (major, feature, maintenance)


def _check_spark_version(sc, report_warn):
    version_info = _get_bigdl_verion_conf()
    (c_major, c_feature, c_maintenance) = _split_full_version(version_info['spark_version'])
    (r_major, r_feature, r_maintenance) = _split_full_version(sc.version)
    error_message = \
        """
        The compile time spark version is not compatible with the spark runtime version.
        Compile time version is %s, runtime version is %s. If you want bypass this check,
        please set spark.analytics.zoo.versionCheck to false, and if you want to only report
        an warning message, please set spark.analytics.zoo.versionCheck.warning to true.
        """ % (version_info['spark_version'], sc.version)
    if c_major != r_major:
        if not report_warn:
            print("***************************Usage Error*****************************")
            print(error_message)
            raise RuntimeError(error_message)
        else:
            warnings.warn(error_message)
    elif not (c_maintenance == r_maintenance and c_feature == r_feature):
        warnings.warn("The compile time spark version may not compatible with " +
                      "the Spark runtime version. " +
                      "Compile time version is %s, " % version_info['spark_version'] +
                      "runtime version is %s" % sc.version)


def _get_bigdl_verion_conf():
    bigdl_build_file = "zoo-version-info.properties"
    bigdl_python_wrapper = "python-api.zip"

    for p in sys.path:
        if bigdl_build_file in p and os.path.isfile(p):
            with open(p) if sys.version_info < (3,) else open(p, encoding='latin-1') as conf_file:
                return load_conf(conf_file.read(), "=")
        if bigdl_python_wrapper in p and os.path.isfile(p):
            import zipfile
            with zipfile.ZipFile(p, 'r') as zip_conf:
                if bigdl_build_file in zip_conf.namelist():
                    content = zip_conf.read(bigdl_build_file)
                    if sys.version_info >= (3,):
                        content = str(content, 'latin-1')
                    return load_conf(content, "=")
    raise RuntimeError("Error while locating file zoo-version-info.properties, " +
                       "please make sure the mvn generate-resources phase" +
                       " is executed and a zoo-version-info.properties file" +
                       " is located in zoo/target/extra-resources")


def load_conf(conf_str, split_char=None):
    return dict(line.split(split_char) for line in conf_str.split("\n") if
                "#" not in line and line.strip())


def set_optimizer_version(optimizerVersion, bigdl_type="float"):
    """
    Set DistriOptimizer version.
    param optimizerVersion: should be "OptimizerV1" or "OptimizerV2".
    """
    callZooFunc(bigdl_type, "setOptimizerVersion", optimizerVersion)


def get_optimizer_version(bigdl_type="float"):
    """
    Get DistriOptimizer version.
    return optimizerVersion
    """
    return callZooFunc(bigdl_type, "getOptimizerVersion")
