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
from zoo import ZooContext, init_nncontext
from zoo import init_spark_on_local, init_spark_on_yarn, init_spark_standalone, stop_spark_standalone
from zoo.ray import RayContext
from zoo.util.utils import detect_python_location


class OrcaContextMeta(type):

    _pandas_read_backend = "spark"
    __eager_mode = True
    _serialize_data_creation = False

    @property
    def log_output(cls):
        """
        Whether to redirect Spark driver JVM's stdout and stderr to the current
        python process. This is useful when running Analytics Zoo in jupyter notebook.
        Default to be False. Needs to be set before initializing SparkContext.
        """
        return ZooContext.log_output

    @log_output.setter
    def log_output(cls, value):
        ZooContext.log_output = value

    @property
    def pandas_read_backend(cls):
        """
        The backend for reading csv/json files. Either "spark" or "pandas".
        spark backend would call spark.read and pandas backend would call pandas.read.
        Default to be "spark".
        """
        return cls._pandas_read_backend

    @pandas_read_backend.setter
    def pandas_read_backend(cls, value):
        value = value.lower()
        assert value == "spark" or value == "pandas", \
            "pandas_read_backend must be either spark or pandas"
        cls._pandas_read_backend = value

    @property
    def _eager_mode(cls):
        """
        Whether to compute eagerly for SparkXShards.
        Default to be True.
        """
        return cls.__eager_mode

    @_eager_mode.setter
    def _eager_mode(cls, value):
        assert isinstance(value, bool), "_eager_mode should either be True or False"
        cls.__eager_mode = value

    @property
    def serialize_data_creation(cls):
        """
        Whether add a file lock to the data loading process for PyTorch Horovod training.
        This would be useful when you run multiple workers on a single node to download data
        to the same destination.
        Default to be False.
        """
        return cls._serialize_data_creation

    @serialize_data_creation.setter
    def serialize_data_creation(cls, value):
        assert isinstance(value, bool), "serialize_data_creation should either be True or False"
        cls._serialize_data_creation = value


class OrcaContext(metaclass=OrcaContextMeta):
    pass


def init_orca_context(cluster_mode="", cores=2, memory='2g', num_nodes=1, init_ray_on_spark=False, **kwargs):
    cluster_mode = cluster_mode.lower()
    spark_args = {}
    for key in ["conf", "spark_log_level", "redirect_spark_log"]:
        if key in kwargs:
            spark_args[key] = kwargs[key]
    if cluster_mode == "":
        sc = init_nncontext(**spark_args)
    elif cluster_mode == "local":
        assert num_nodes == 1
        os.environ["SPARK_DRIVER_MEMORY"] = memory
        if "python_location" in kwargs:
            spark_args["python_location"] = kwargs["python_location"]
        sc = init_spark_on_local(cores, **spark_args)
    elif cluster_mode.startswith("yarn"):  # yarn or yarn-client
        if cluster_mode == "yarn-cluster":
            raise ValueError("For yarn-cluster mode, please use the default cluster_mode and spark-submit instead")
        hadoop_conf = os.environ.get("HADOOP_CONF_DIR")
        if not hadoop_conf:
            assert "hadoop_conf" in kwargs,\
                "Directory path to hadoop conf not found for yarn-client mode. Please either specify argument " \
                "hadoop_conf or set the environment variable HADOOP_CONF_DIR"
            hadoop_conf = kwargs["hadoop_conf"]
        python_location = detect_python_location()  # /path/to/conda/envs/conda_name/bin/python
        assert "envs" in python_location, "You must use a conda environment for yarn-client mode"
        for key in ["driver_cores", "driver_memory", "extra_executor_memory_for_ray", "extra_python_lib",
                    "penv_archive", "additional_archive", "hadoop_user_name", "spark_yarn_archive", "jars"]:
            if key in kwargs:
                spark_args[key] = kwargs[key]
        sc = init_spark_on_yarn(hadoop_conf=hadoop_conf, conda_name=python_location.split("/")[-3],
                                num_executors=num_nodes, executor_cores=cores, executor_memory=memory,
                                **spark_args)
    elif cluster_mode == "standalone":
        for key in ["driver_cores", "driver_memory", "extra_executor_memory_for_ray", "extra_python_lib",
                    "jars", "master"]:
            if key in kwargs:
                spark_args[key] = kwargs[key]
        sc = init_spark_standalone(num_executors=num_nodes, executor_cores=cores, executor_memory=memory,
                                   **spark_args)
    else:
        raise ValueError("cluster_mode can only be local, yarn-client or standalone, "
                         "but got: %s".format(cluster_mode))
    ray_args = {}
    for key in ["redis_port", "password", "object_store_memory", "verbose", "env",
                "extra_params", "num_ray_nodes", "ray_node_cpu_cores"]:
        if key in kwargs:
            ray_args[key] = kwargs[key]
    ray_ctx = RayContext(sc, **ray_args)
    if init_ray_on_spark:
        driver_cores = 0  # This is the default value.
        if "driver_cores" in kwargs:
            driver_cores = kwargs["driver_cores"]
        ray_ctx.init(driver_cores=driver_cores)
    return sc


def stop_orca_context():
    ray_ctx = RayContext.get(initialize=False)
    if ray_ctx.initialized:
        ray_ctx.stop()
    sc = SparkContext.getOrCreate()
    if sc.getConf().get("spark.master").startswith("spark://"):
        stop_spark_standalone()
    sc.stop()
