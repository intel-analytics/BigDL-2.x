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
import time
import signal

from pyspark import BarrierTaskContext
from zoo.ray.util.process import session_execute, ProcessMonitor
from zoo.ray.util.utils import resourceToBytes
import ray.services as rservices


class JVMGuard():
    """
    The registered pids would be put into the killing list of Spark Executor.
    """

    @staticmethod
    def registerPids(pids):
        import logging
        import traceback
        try:
            from bigdl.util.common import callBigDlFunc
            import zoo
            callBigDlFunc("float",
                          "jvmGuardRegisterPids",
                          pids)
        except Exception as err:
            logging.err(traceback.format_exc())
            logging.err("Cannot sucessfully register pid into JVMGuard")
            for pid in pids:
                os.kill(pid, signal.SIGKILL)
            raise err


class RayServiceFuncGenerator(object):
    """
    This should be a pickable class.
    """

    def _get_MKL_config(self, cores):
        return {"intra_op_parallelism_threads": str(cores),
                "inter_op_parallelism_threads": str(cores),
                "OMP_NUM_THREADS": str(cores),
                "KMP_BLOCKTIME": "0",
                "KMP_AFFINITY": "granularity = fine, verbose, compact, 1, 0",
                "KMP_SETTINGS": "0"}

    def _prepare_env(self, cores=None):
        modified_env = os.environ.copy()
        if self.env:
            modified_env.update(self.env)
        cwd = os.getcwd()
        modified_env["PATH"] = "{}/{}:{}".format(cwd, "/".join(self.python_loc.split("/")[:-1]),
                                                 os.environ["PATH"])
        modified_env.pop("MALLOC_ARENA_MAX", None)
        modified_env.pop("RAY_BACKEND_LOG_LEVEL", None)
        # unset all MKL setting
        modified_env.pop("intra_op_parallelism_threads", None)
        modified_env.pop("inter_op_parallelism_threads", None)
        modified_env.pop("OMP_NUM_THREADS", None)
        modified_env.pop("KMP_BLOCKTIME", None)
        modified_env.pop("KMP_AFFINITY", None)
        modified_env.pop("KMP_SETTINGS", None)
        if cores:
            cores = int(cores)
            print("MKL cores is {}".format(cores))
            modified_env.update(self._get_MKL_config(cores))
        if self.verbose:
            print("Executing with these environment setting:")
            for pair in modified_env.items():
                print(pair)
            print("The $PATH is: {}".format(modified_env["PATH"]))
        return modified_env

    def __init__(self, python_loc, redis_port, ray_node_cpu_cores, mkl_cores,
                 password, object_store_memory, waitting_time=10, verbose=False, env=None):
        self.env = env
        self.python_loc = python_loc
        self.redis_port = redis_port
        self.password = password
        self.ray_node_cpu_cores = ray_node_cpu_cores
        self.mkl_cores = mkl_cores
        self.ray_exec = self._get_ray_exec()
        self.object_store_memory = object_store_memory
        self.WAITING_TIME_SEC = waitting_time
        self.verbose = verbose
        self.labels = """--resources='{"trainer": %s, "ps": %s }' """ % (1, 1)

    def gen_stop(self):
        def _stop(iter):
            command = "{} stop".format(self.ray_exec)
            print("Start to end the ray services: {}".format(command))
            session_execute(command=command, fail_fast=True)
            return iter

        return _stop

    def _start_master(self):
        """
        Start the Master for Ray
        :return:
        """
        modified_env = self._prepare_env(self.mkl_cores)

        command = "{} start --head " \
                  "--include-webui --redis-port {} \
                  --redis-password {} --num-cpus {} ". \
            format(self.ray_exec, self.redis_port, self.password, self.ray_node_cpu_cores)
        if self.object_store_memory:
            command = command + "--object-store-memory {} ".format(str(self.object_store_memory))
        print("Starting ray master by running: {}".format(command))
        process_info = session_execute(command=command, env=modified_env, tag="ray_master")
        JVMGuard.registerPids(process_info.pids)
        process_info.node_ip = rservices.get_node_ip_address()
        time.sleep(self.WAITING_TIME_SEC)
        return process_info

    def _start_raylet(self, redis_address):
        """
        Start the Slave for Ray
        :return:
        """
        command = "{} start --redis-address {} --redis-password  {} --num-cpus {} {}  ".format(
            self.ray_exec, redis_address, self.password, self.ray_node_cpu_cores, self.labels)

        if self.object_store_memory:
            command = command + "--object-store-memory {} ".format(str(self.object_store_memory))

        print("Starting raylet by running: {}".format(command))

        modified_env = self._prepare_env(self.mkl_cores)
        time.sleep(self.WAITING_TIME_SEC)
        process_info = session_execute(command=command, env=modified_env, tag="raylet")
        JVMGuard.registerPids(process_info.pids)
        process_info.node_ip = rservices.get_node_ip_address()
        return process_info

    def _get_ray_exec(self):
        python_bin_dir = "/".join(self.python_loc.split("/")[:-1])
        return "{}/python {}/ray".format(python_bin_dir, python_bin_dir)

    def gen_ray_booter(self):
        def _start_ray_services(iter):
            tc = BarrierTaskContext.get()
            # The address is sorted by partitionId according to the comments
            # Partition 0 is the Master
            task_addrs = [taskInfo.address for taskInfo in tc.getTaskInfos()]
            print(task_addrs)
            master_ip = task_addrs[0].split(":")[0]
            print("current address {}".format(task_addrs[tc.partitionId()]))
            print("master address {}".format(master_ip))
            redis_address = "{}:{}".format(master_ip, self.redis_port)
            if tc.partitionId() == 0:
                print("partition id is : {}".format(tc.partitionId()))
                process_info = self._start_master()
                process_info.master_addr = redis_address
                yield process_info
            else:
                print("partition id is : {}".format(tc.partitionId()))
                process_info = self._start_raylet(redis_address=redis_address)
                yield process_info
            tc.barrier()

        return _start_ray_services


class RayRunner(object):
    # TODO: redis_port should be retrieved by random searched
    def __init__(self, sc, redis_port="5346", password="123456", object_store_memory=None,
                 force_purge=False, verbose=False, env=None):
        self.sc = sc
        self.ray_node_cpu_cores = self._get_ray_node_cpu_cores()
        self.num_ray_nodes = self._get_num_ray_nodes()
        self.python_loc = os.environ['PYSPARK_PYTHON']
        self.ray_processesMonitor = None
        self.ray_context = RayServiceFuncGenerator(
            python_loc=self.python_loc,
            redis_port=redis_port,
            ray_node_cpu_cores=self.ray_node_cpu_cores,
            mkl_cores=self._get_mkl_cores(),
            password=password,
            object_store_memory=resourceToBytes(
                str(object_store_memory)) if object_store_memory else None,
            verbose=verbose, env=env)
        self._gather_cluster_ips()
        if force_purge:
            self.stop()
        from bigdl.util.common import init_executor_gateway
        print("Start to launch the JVM guarding process")
        init_executor_gateway(sc)
        print("JVM guarding process has been successfully launched")

    def _gather_cluster_ips(self):
        total_cores = int(self._get_num_ray_nodes()) * int(self._get_ray_node_cpu_cores())

        def info_fn(iter):
            tc = BarrierTaskContext.get()
            task_addrs = [taskInfo.address.split(":")[0] for taskInfo in tc.getTaskInfos()]
            yield task_addrs
            tc.barrier()

        ips = self.sc.range(0, total_cores,
                            numSlices=total_cores).barrier().mapPartitions(info_fn).collect()
        return ips[0]

    @classmethod
    def from_spark(cls, hadoop_conf,
                   slave_num, slave_cores, slave_memory,
                   driver_cores, driver_memory, conda_name,
                   extra_pmodule_zip=None, penv_archive=None, jars=None, env=None,
                   object_store_memory_ratio=0.5,
                   force_purge=False,
                   verbose=False,
                   spark_log_level="WARN",
                   redirect_spark_log=True):
        from zoo.ray.util.spark import SparkRunner
        spark_runner = SparkRunner(spark_log_level=spark_log_level,
                                   redirect_spark_log=redirect_spark_log)
        sc = spark_runner.init_spark_on_yarn(
            hadoop_conf=hadoop_conf,
            penv_archive=penv_archive,
            conda_name=conda_name,
            extra_pmodule_zip=extra_pmodule_zip,
            num_executor=slave_num,
            executor_cores=slave_cores,
            executor_memory="{}b".format(
                int(resourceToBytes(slave_memory) * (1 - object_store_memory_ratio))),
            driver_memory=driver_memory,
            driver_cores=driver_cores,
            extra_executor_memory_for_ray="{}b".format(
                int(resourceToBytes(slave_memory) * (object_store_memory_ratio))),
            jars=jars)
        return cls(sc=sc, force_purge=force_purge, verbose=verbose, env=env,
                   object_store_memory="{}b".format(
                       int(resourceToBytes(slave_memory) * object_store_memory_ratio)))

    def stop(self):
        import ray
        ray.shutdown()
        if not self.ray_processesMonitor:
            print("Please start the runner first before closing it")
        else:
            self.ray_processesMonitor.clean_fn()
        # self.sc.range(0,
        #               self.num_ray_nodes,
        #               numSlices=self.num_ray_nodes).barrier().mapPartitions(
        #     self.ray_context.gen_stop()).collect()

    def _get_mkl_cores(self):
        if "local" in self.sc.master:
            return 1
        else:
            return int(self.sc._conf.get("spark.executor.cores"))

    def _get_ray_node_cpu_cores(self):
        if "local" in self.sc.master:
            return 1
        else:
            return self.sc._conf.get("spark.executor.cores")

    def _get_ray_driver_memory(self):
        if "local" in self.sc.master:
            return "1g"
        else:
            return self.sc._conf.get("spark.driver.memory")

    def _get_num_ray_nodes(self):
        if "local" in self.sc.master:
            return int(re.match(r"local\[(.*)\]", self.sc.master).group(1))
        else:
            return int(self.sc._conf.get("spark.executor.instances"))

    def start(self):
        self._start_cluster()
        self._start_driver(object_store_memory=self._get_ray_driver_memory())

    def _start_cluster(self):
        ray_rdd = self.sc.range(0, self.num_ray_nodes,
                                numSlices=self.num_ray_nodes)
        process_infos = ray_rdd.barrier().mapPartitions(
            self.ray_context.gen_ray_booter()).collect()

        self.ray_processesMonitor = ProcessMonitor(process_infos, self.sc, ray_rdd)
        self.redis_address = self.ray_processesMonitor.master.master_addr
        return self

    def _start_restricted_worker(self, redis_address, redis_password, object_store_memory):
        num_cores = 0
        command = "ray start --redis-address {} " \
                  "--redis-password  {} --num-cpus {} --object-store-memory {}".\
            format(redis_address, redis_password, num_cores, object_store_memory)
        print("".format(command))
        process_info = session_execute(command=command, fail_fast=True)
        ProcessMonitor.register_shutdown_hook(pgid=process_info.pgid)

    def _start_driver(self, object_store_memory="10g"):
        import ray
        self._start_restricted_worker(self.redis_address, self.ray_context.password,
                                      object_store_memory=resourceToBytes(object_store_memory))
        ray.shutdown()
        ray.init(redis_address=self.redis_address,
                 redis_password=self.ray_context.password)
