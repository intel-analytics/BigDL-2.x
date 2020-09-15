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
import re
import random
import signal
import warnings
import multiprocessing

from zoo.ray.process import session_execute, ProcessMonitor
from zoo.ray.utils import is_local
from zoo.ray.utils import resource_to_bytes


class JVMGuard:
    """
    The registered pids would be put into the killing list of Spark Executor.
    """
    @staticmethod
    def register_pids(pids):
        import traceback
        try:
            from zoo.common.utils import callZooFunc
            import zoo
            callZooFunc("float",
                        "jvmGuardRegisterPids",
                        pids)
        except Exception as err:
            print(traceback.format_exc())
            print("Cannot successfully register pid into JVMGuard")
            for pid in pids:
                os.kill(pid, signal.SIGKILL)
            raise err


def kill_redundant_log_monitors(redis_address):

    """
    Killing redundant log_monitor.py processes.
    If multiple ray nodes are started on the same machine,
    there will be multiple ray log_monitor.py processes
    monitoring the same log dir. As a result, the logs
    will be replicated multiple times and forwarded to driver.
    See issue https://github.com/ray-project/ray/issues/10392
    """

    import psutil
    import subprocess
    log_monitor_processes = []
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            # Avoid throw exception when listing lwsslauncher in macOS
            if proc.name() is None or proc.name() == "lwsslauncher":
                continue
            cmdline = subprocess.list2cmdline(proc.cmdline())
            is_log_monitor = "log_monitor.py" in cmdline
            is_same_redis = "--redis-address={}".format(redis_address)
            if is_log_monitor and is_same_redis in cmdline:
                log_monitor_processes.append(proc)
        except psutil.AccessDenied:
            # psutil may encounter AccessDenied exceptions
            # when it's trying to visit core services
            if psutil.MACOS:
                continue
            else:
                raise

    if len(log_monitor_processes) > 1:
        for proc in log_monitor_processes[1:]:
            proc.kill()


class RayServiceFuncGenerator(object):
    """
    This should be a pickable class.
    """
    def _prepare_env(self):
        modified_env = os.environ.copy()
        if self.python_loc == "python_env/bin/python":
            # In this case the executor is using the conda yarn archive under the current
            # working directory. Need to get the full path.
            executor_python_path = "{}/{}".format(
                os.getcwd(), "/".join(self.python_loc.split("/")[:-1]))
        else:
            executor_python_path = "/".join(self.python_loc.split("/")[:-1])
        if "PATH" in os.environ:
            modified_env["PATH"] = "{}:{}".format(executor_python_path, os.environ["PATH"])
        else:
            modified_env["PATH"] = executor_python_path
        modified_env["LC_ALL"] = "C.UTF-8"
        modified_env["LANG"] = "C.UTF-8"
        modified_env.pop("MALLOC_ARENA_MAX", None)
        modified_env.pop("RAY_BACKEND_LOG_LEVEL", None)
        # Unset all MKL setting as Analytics Zoo would give default values when init env.
        # Running different programs may need different configurations.
        modified_env.pop("intra_op_parallelism_threads", None)
        modified_env.pop("inter_op_parallelism_threads", None)
        modified_env.pop("OMP_NUM_THREADS", None)
        modified_env.pop("KMP_BLOCKTIME", None)
        modified_env.pop("KMP_AFFINITY", None)
        modified_env.pop("KMP_SETTINGS", None)
        if self.env:  # Add in env argument if any MKL setting is needed.
            modified_env.update(self.env)
        if self.verbose:
            print("Executing with these environment settings:")
            for pair in modified_env.items():
                print(pair)
            print("The $PATH is: {}".format(modified_env["PATH"]))
        return modified_env

    def __init__(self, python_loc, redis_port, ray_node_cpu_cores,
                 password, object_store_memory, verbose=False, env=None,
                 extra_params=None):
        """object_store_memory: integer in bytes"""
        self.env = env
        self.python_loc = python_loc
        self.redis_port = redis_port
        self.password = password
        self.ray_node_cpu_cores = ray_node_cpu_cores
        self.ray_exec = self._get_ray_exec()
        self.object_store_memory = object_store_memory
        self.extra_params = extra_params
        self.verbose = verbose
        # _mxnet_worker and _mxnet_server are resource tags for distributed MXNet training only
        # in order to diff worker from server.
        # This is useful to allocate workers and servers in the cluster.
        # Leave some reserved custom resources free to avoid unknown crash due to resources.
        self.labels = \
            """--resources '{"_mxnet_worker": %s, "_mxnet_server": %s, "_reserved": %s}'""" \
            % (1, 1, 2)

    def gen_stop(self):
        def _stop(iter):
            command = "{} stop".format(self.ray_exec)
            print("Start to end the ray services: {}".format(command))
            session_execute(command=command, fail_fast=True)
            return iter

        return _stop

    @staticmethod
    def _enrich_command(command, object_store_memory, extra_params):
        if object_store_memory:
            command = command + " --object-store-memory {}".format(str(object_store_memory))
        if extra_params:
            for pair in extra_params.items():
                command = command + " --{} {}".format(pair[0], pair[1])
        return command

    def _gen_master_command(self):
        command = "{} start --head " \
                  "--include-webui true --redis-port {} " \
                  "--redis-password {} --num-cpus {}". \
            format(self.ray_exec, self.redis_port, self.password,
                   self.ray_node_cpu_cores)
        if self.labels:
            command = command + " " + self.labels
        return RayServiceFuncGenerator._enrich_command(command=command,
                                                       object_store_memory=self.object_store_memory,
                                                       extra_params=self.extra_params)

    @staticmethod
    def _get_raylet_command(redis_address,
                            ray_exec,
                            password,
                            ray_node_cpu_cores,
                            labels="",
                            object_store_memory=None,
                            extra_params=None):
        command = "{} start --address {} --redis-password {} --num-cpus {}".format(
            ray_exec, redis_address, password, ray_node_cpu_cores)
        if labels:
            command = command + " " + labels
        return RayServiceFuncGenerator._enrich_command(command=command,
                                                       object_store_memory=object_store_memory,
                                                       extra_params=extra_params)

    def _start_ray_node(self, command, tag):
        modified_env = self._prepare_env()
        print("Starting {} by running: {}".format(tag, command))
        process_info = session_execute(command=command, env=modified_env, tag=tag)
        JVMGuard.register_pids(process_info.pids)
        import ray.services as rservices
        process_info.node_ip = rservices.get_node_ip_address()
        return process_info

    def _get_ray_exec(self):
        if "envs" in self.python_loc:  # conda environment
            python_bin_dir = "/".join(self.python_loc.split("/")[:-1])
            return "{}/python {}/ray".format(python_bin_dir, python_bin_dir)
        else:  # system environment with ray installed; for example: /usr/local/bin/ray
            return "ray"

    def gen_ray_start(self, master_ip):
        def _start_ray_services(iter):
            from pyspark import BarrierTaskContext
            from zoo.util.utils import get_node_ip
            tc = BarrierTaskContext.get()
            current_ip = get_node_ip()
            print("current address {}".format(current_ip))
            print("master address {}".format(master_ip))
            redis_address = "{}:{}".format(master_ip, self.redis_port)
            process_info = None
            import tempfile
            import filelock
            base_path = tempfile.gettempdir()
            master_flag_path = os.path.join(base_path, "ray_master_initialized")
            if current_ip == master_ip:  # Start the ray master.
                lock_path = os.path.join(base_path, "ray_master_start.lock")
                # It is possible that multiple executors are on one node. In this case,
                # the first executor that gets the lock would be the master and it would
                # create a flag to indicate the master has initialized.
                # The flag file is removed when ray start processes finish so that this
                # won't affect other programs.
                with filelock.FileLock(lock_path):
                    if not os.path.exists(master_flag_path):
                        print("partition id is : {}".format(tc.partitionId()))
                        process_info = self._start_ray_node(command=self._gen_master_command(),
                                                            tag="ray-master")
                        process_info.master_addr = redis_address
                        os.mknod(master_flag_path)

            tc.barrier()
            if not process_info:  # Start raylets.
                lock_path = os.path.join(base_path, "raylet_start.lock")
                with filelock.FileLock(lock_path):
                    print("partition id is : {}".format(tc.partitionId()))
                    process_info = self._start_ray_node(
                        command=RayServiceFuncGenerator._get_raylet_command(
                            redis_address=redis_address,
                            ray_exec=self.ray_exec,
                            password=self.password,
                            ray_node_cpu_cores=self.ray_node_cpu_cores,
                            labels=self.labels,
                            object_store_memory=self.object_store_memory,
                            extra_params=self.extra_params),
                        tag="raylet")
                    kill_redundant_log_monitors(redis_address=redis_address)
            if os.path.exists(master_flag_path):
                os.remove(master_flag_path)

            yield process_info
        return _start_ray_services


class RayContext(object):
    _active_ray_context = None

    def __init__(self, sc, redis_port=None, password="123456", object_store_memory=None,
                 verbose=False, env=None, extra_params=None,
                 num_ray_nodes=None, ray_node_cpu_cores=None):
        """
        The RayContext would initiate a ray cluster on top of the configuration of SparkContext.
        After creating RayContext, call the init method to set up the cluster.

        - For Spark local mode: The total available cores for Ray is equal to the number of
        Spark local cores.
        - For Spark cluster mode: The number of raylets to be created is equal to the number of
        Spark executors. The number of cores allocated for each raylet is equal to the number of
        cores for each Spark executor.
        You are allowed to specify num_ray_nodes and ray_node_cpu_cores for configurations
        to start raylets.

        :param sc: An instance of SparkContext.
        :param redis_port: redis port for the "head" node.
        The value would be randomly picked if not specified.
        :param password: Password for the redis. Default to be "123456" if not specified.
        :param object_store_memory: The memory size for ray object_store in string.
        This can be specified in bytes(b), kilobytes(k), megabytes(m) or gigabytes(g).
        For example, 50b, 100k, 250m, 30g.
        :param verbose: True for more logs when starting ray. Default is False.
        :param env: The environment variable dict for running ray processes. Default is None.
        :param extra_params: The key value dict for extra options to launch ray.
        For example, extra_params={"temp-dir": "/tmp/ray/"}
        :param num_ray_nodes: The number of raylets to start across the cluster.
        For Spark local mode, you don't need to specify this value.
        For Spark cluster mode, it is default to be the number of Spark executors. If
        spark.executor.instances can't be detected in your SparkContext, you need to explicitly
        specify this. It is recommended that num_ray_nodes is not larger than the number of
        Spark executors to make sure there are enough resources in your cluster.
        :param ray_node_cpu_cores: The number of available cores for each raylet.
        For Spark local mode, it is default to be the number of Spark local cores.
        For Spark cluster mode, it is default to be the number of cores for each Spark executor. If
        spark.executor.cores or spark.cores.max can't be detected in your SparkContext, you need to
        explicitly specify this. It is recommended that ray_node_cpu_cores is not larger than the
        number of cores for each Spark executor to make sure there are enough resources in your
        cluster.
        """
        assert sc is not None, "sc cannot be None, please create a SparkContext first"
        self.sc = sc
        self.initialized = False
        self.is_local = is_local(sc)
        self.verbose = verbose
        self.redis_password = password
        self.object_store_memory = resource_to_bytes(object_store_memory)
        self.ray_processesMonitor = None
        self.env = env
        self.extra_params = extra_params
        self._address_info = None
        if self.is_local:
            self.num_ray_nodes = 1
            spark_cores = self._get_spark_local_cores()
            if ray_node_cpu_cores:
                ray_node_cpu_cores = int(ray_node_cpu_cores)
                if ray_node_cpu_cores > spark_cores:
                    warnings.warn("ray_node_cpu_cores is larger than available Spark cores, "
                                  "make sure there are enough resources on your machine")
                self.ray_node_cpu_cores = ray_node_cpu_cores
            else:
                self.ray_node_cpu_cores = spark_cores
        # For Spark local mode, directly call ray.init() and ray.shutdown().
        # ray.shutdown() would clear up all the ray related processes.
        # Ray Manager is only needed for Spark cluster mode to monitor ray processes.
        else:
            if self.sc.getConf().contains("spark.executor.cores"):
                executor_cores = int(self.sc.getConf().get("spark.executor.cores"))
            else:
                executor_cores = None
            if ray_node_cpu_cores:
                ray_node_cpu_cores = int(ray_node_cpu_cores)
                if executor_cores and ray_node_cpu_cores > executor_cores:
                    warnings.warn("ray_node_cpu_cores is larger than Spark executor cores, "
                                  "make sure there are enough resources on your cluster")
                self.ray_node_cpu_cores = ray_node_cpu_cores
            elif executor_cores:
                self.ray_node_cpu_cores = executor_cores
            else:
                raise Exception("spark.executor.cores not detected in the SparkContext, "
                                "you need to manually specify num_ray_nodes and ray_node_cpu_cores "
                                "for RayContext to start ray services")
            if self.sc.getConf().contains("spark.executor.instances"):
                num_executors = int(self.sc.getConf().get("spark.executor.instances"))
            elif self.sc.getConf().contains("spark.cores.max"):
                import math
                num_executors = math.floor(
                    int(self.sc.getConf().get("spark.cores.max")) / self.ray_node_cpu_cores)
            else:
                num_executors = None
            if num_ray_nodes:
                num_ray_nodes = int(num_ray_nodes)
                if num_executors and num_ray_nodes > num_executors:
                    warnings.warn("num_ray_nodes is larger than the number of Spark executors, "
                                  "make sure there are enough resources on your cluster")
                self.num_ray_nodes = num_ray_nodes
            elif num_executors:
                self.num_ray_nodes = num_executors
            else:
                raise Exception("spark.executor.cores not detected in the SparkContext, "
                                "you need to manually specify num_ray_nodes and ray_node_cpu_cores "
                                "for RayContext to start ray services")

            from zoo.util.utils import detect_python_location
            self.python_loc = os.environ.get("PYSPARK_PYTHON", detect_python_location())
            self.redis_port = random.randint(10000, 65535) if not redis_port else int(redis_port)
            self.ray_service = RayServiceFuncGenerator(
                python_loc=self.python_loc,
                redis_port=self.redis_port,
                ray_node_cpu_cores=self.ray_node_cpu_cores,
                password=self.redis_password,
                object_store_memory=self.object_store_memory,
                verbose=self.verbose,
                env=self.env,
                extra_params=self.extra_params)
        RayContext._active_ray_context = self

    @classmethod
    def get(cls, initialize=True):
        if RayContext._active_ray_context:
            ray_ctx = RayContext._active_ray_context
            if initialize and not ray_ctx.initialized:
                ray_ctx.init()
            return ray_ctx
        else:
            raise Exception("No active RayContext. Please create a RayContext and init it first")

    def _gather_cluster_ips(self):
        """
        Get the ips of all Spark executors in the cluster. The first ip returned would be the
        ray master.
        """
        def info_fn(iter):
            from zoo.util.utils import get_node_ip
            yield get_node_ip()

        ips = self.sc.range(0, self.num_ray_nodes,
                            numSlices=self.num_ray_nodes).barrier().mapPartitions(info_fn).collect()
        return ips

    def stop(self):
        if not self.initialized:
            print("The Ray cluster has not been launched.")
            return
        import ray
        ray.shutdown()
        if not self.is_local:
            if not self.ray_processesMonitor:
                print("Please start the runner first before closing it")
            else:
                self.ray_processesMonitor.clean_fn()
        self.initialized = False

    def purge(self):
        """
        Invoke ray stop to clean ray processes.
        """
        if not self.initialized:
            print("The Ray cluster has not been launched.")
            return
        if self.is_local:
            import ray
            ray.shutdown()
        else:
            self.sc.range(0,
                          self.num_ray_nodes,
                          numSlices=self.num_ray_nodes).barrier().mapPartitions(
                self.ray_service.gen_stop()).collect()
        self.initialized = False

    def _get_spark_local_cores(self):
        local_symbol = re.match(r"local\[(.*)\]", self.sc.master).group(1)
        if local_symbol == "*":
            return multiprocessing.cpu_count()
        else:
            return int(local_symbol)

    def init(self, driver_cores=0):
        """
        Initiate the ray cluster.

        :param driver_cores: The number of cores for the raylet on driver for Spark cluster mode.
        Default is 0 and in this case the local driver wouldn't have any ray workload.

        :return The dictionary of address information about the ray cluster.
        Information contains node_ip_address, redis_address, object_store_address,
        raylet_socket_name, webui_url and session_dir.
        """
        if self.initialized:
            print("The Ray cluster has been launched.")
        else:
            if self.is_local:
                if self.env:
                    os.environ.update(self.env)
                import ray
                self._address_info = ray.init(num_cpus=self.ray_node_cpu_cores,
                                              object_store_memory=self.object_store_memory,
                                              resources=self.extra_params)
            else:
                self.cluster_ips = self._gather_cluster_ips()
                from bigdl.util.common import init_executor_gateway
                init_executor_gateway(self.sc)
                print("JavaGatewayServer has been successfully launched on executors")
                self._start_cluster()
                self._address_info = self._start_driver(num_cores=driver_cores)

            print(self._address_info)
            kill_redundant_log_monitors(self._address_info["redis_address"])
            self.initialized = True
        return self._address_info

    @property
    def address_info(self):
        if self._address_info:
            return self._address_info
        else:
            raise Exception("The Ray cluster has not been launched yet. Please call init first")

    def _start_cluster(self):
        print("Start to launch ray on cluster")
        ray_rdd = self.sc.range(0, self.num_ray_nodes,
                                numSlices=self.num_ray_nodes)
        # The first ip would be used to launch ray master.
        process_infos = ray_rdd.barrier().mapPartitions(
            self.ray_service.gen_ray_start(self.cluster_ips[0])).collect()

        self.ray_processesMonitor = ProcessMonitor(process_infos, self.sc, ray_rdd, self,
                                                   verbose=self.verbose)
        self.redis_address = self.ray_processesMonitor.master.master_addr
        return self

    def _start_restricted_worker(self, num_cores, node_ip_address):
        extra_param = {"node-ip-address": node_ip_address}
        if self.extra_params is not None:
            extra_param.update(self.extra_params)
        command = RayServiceFuncGenerator._get_raylet_command(
            redis_address=self.redis_address,
            ray_exec="ray",
            password=self.redis_password,
            ray_node_cpu_cores=num_cores,
            object_store_memory=self.object_store_memory,
            extra_params=extra_param)
        modified_env = self.ray_service._prepare_env()
        print("Executing command: {}".format(command))
        process_info = session_execute(command=command, env=modified_env,
                                       tag="raylet", fail_fast=True)
        ProcessMonitor.register_shutdown_hook(pgid=process_info.pgid)

    def _start_driver(self, num_cores=0):
        print("Start to launch ray driver on local")
        import ray.services
        node_ip = ray.services.get_node_ip_address(self.redis_address)
        self._start_restricted_worker(num_cores=num_cores,
                                      node_ip_address=node_ip)
        ray.shutdown()
        return ray.init(address=self.redis_address,
                        redis_password=self.ray_service.password,
                        node_ip_address=node_ip)
