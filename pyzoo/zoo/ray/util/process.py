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
import subprocess
import signal
import atexit
import sys

from zoo.ray.util import gen_shutdown_per_node, is_local

class ProcessInfo(object):
    def __init__(self, out, err, errorcode, pgid, pid=None, node_ip=None):
        self.out=out
        self.err=err
        self.pgid=pgid
        self.pid=pid
        self.errorcode=errorcode
        self.tag="default"
        self.master_addr=None
        self.node_ip=node_ip

def session_execute(command, env=None, tag=None, fail_fast=False, timeout=120):
    pro = subprocess.Popen(
        command,
        shell=True,
        env=env,
        cwd=None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid)
    pgid = os.getpgid(pro.pid)
    print("The pgid for the current session is: {}".format(pgid))
    out, err = pro.communicate(timeout=timeout)
    out=out.decode("utf-8")
    err=err.decode("utf-8")
    print(out)
    print(err)
    errorcode=pro.returncode
    if errorcode != 0:
        # https://bip.weizmann.ac.il/course/python/PyMOTW/PyMOTW/docs/atexit/index.html
        # http://www.pybloggers.com/2016/02/how-to-always-execute-exit-functions-in-python/    register for signal handling
        if fail_fast:
            raise Exception(err)
        print(err)
    else:
        print(out)
    return ProcessInfo(out, err, pro.returncode, pgid, tag)


class ProcessMonitor:
    def __init__(self, process_infos, sc, ray_rdd):
        self.sc = sc
        self.ray_rdd = ray_rdd
        self.master = []
        self.slaves = []
        self.pgids=[] # TODO: change me to dict
        for process_info in process_infos:
            self.pgids.append(process_info.pgid)
            if process_info.master_addr:
                self.master.append(process_info)
            else:
                self.slaves.append(process_info)
        self.register_cluster_shutdown_hook()
        assert len(self.master) == 1, "We should got 1 master only, but we got {}".format(len(self.master))
        self.master = self.master[0]
        if not is_local(self.sc):
            self.print_ray_remote_err_out()



    def print_ray_remote_err_out(self):
        if self.master.errorcode != 0:
            raise Exception(self.master.err)
        for slave in self.slaves:
            if slave.errorcode != 0:
                raise Exception(slave.err)

        print(self.master.out)
        for slave in self.slaves:
            # TODO: implement __str__ for class ProcessInfo
            print(slave.out)

    def register_cluster_shutdown_hook(self):
        def _shutdown():
            import ray
            ray.shutdown()
            self.ray_rdd.map(gen_shutdown_per_node(self.pgids)).collect()

        def _signal_shutdown(_signo, _stack_frame):
            _shutdown()
            sys.exit(0)

        atexit.register(_shutdown)
        signal.signal(signal.SIGTERM, _signal_shutdown)
        signal.signal(signal.SIGINT, _signal_shutdown)

    @staticmethod
    def register_shutdown_hook_local(pgid):
        def _shutdown():
            gen_shutdown_per_node(pgid)

        def _signal_shutdown(_signo, _stack_frame):
            _shutdown()
            sys.exit(0)

        # TODO: are there any other signal we want to handle?
        atexit.register(_shutdown)
        signal.signal(signal.SIGTERM, _signal_shutdown)
        signal.signal(signal.SIGINT, _signal_shutdown)

        # pgids=list(itertools.chain.from_iterable(pgids))
        # exceptions=list(itertools.chain.from_iterable(exceptions))
