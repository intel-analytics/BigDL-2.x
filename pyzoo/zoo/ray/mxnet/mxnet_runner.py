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
import socket
from contextlib import closing
from dmlc_tracker.tracker import get_host_ip
import ray.services


class MXNetRunner(object):
    """Manages a MXNet model for training."""

    def __init__(self, data_creator, model_creator, loss_creator, metrics_creator, config):
        self.data_creator = data_creator
        self.model_creator = model_creator
        self.loss_creator = loss_creator
        self.metrics_creator = metrics_creator
        self.config = config  # TODO: add check for config keys
        self.is_worker = False
        self.epoch = 0

    def setup_distributed(self, env):
        env["DMLC_NODE_HOST"] = self.get_node_ip()
        if env["DMLC_ROLE"] == "worker":
            self.is_worker = True

        if self.is_worker:
            os.environ.update(env)
            import mxnet as mx
            if "seed" in self.config:
                mx.random.seed(self.config["seed"])
            self.kv = mx.kv.create(self.config["kvstore"])
            self.train_dataset, self.test_dataset = self.data_creator(self.config, self.kv)
            self.model = self.model_creator(self.config)
            self.loss = self.loss_creator(self.config)
            self.metrics = self.metrics_creator(self.config)
            from mxnet import gluon
            self.trainer = gluon.Trainer(self.model.collect_params(), self.config["optimizer"],
                                         optimizer_params=self.config["optimizer_params"],
                                         kvstore=self.kv)
        else:  # server
            # Need to use the environment on each raylet process for the correct python environment.
            modified_env = os.environ.copy()
            modified_env.update(env)
            subprocess.Popen("python -c 'import mxnet'", shell=True, env=modified_env)

    def step(self):
        """Runs a training epoch and updates the model parameters."""
        # TODO: make this as a train_function input same as PyTorchRunner
        self.epoch += 1
        stats = dict()
        stats["epoch"] = self.epoch
        if self.is_worker:
            import time
            tic = time.time()
            self.train_dataset.reset()
            self.metrics.reset()  # metrics will accumulate for one batch
            btic = time.time()
            for i, batch in enumerate(self.train_dataset):
                import mxnet as mx
                from mxnet import gluon
                # MXNet treats all CPUs on a single machine as a single device.
                # So whether you specify cpu(0) or cpu(), MXNet will use all CPU cores on the machine.
                data = gluon.utils.split_and_load(batch.data[0].astype("float32"), ctx_list=[mx.cpu()], batch_axis=0)
                label = gluon.utils.split_and_load(batch.label[0].astype("float32"), ctx_list=[mx.cpu()], batch_axis=0)
                outputs = []
                Ls = []
                from mxnet import autograd as ag
                with ag.record():
                    for x, y in zip(data, label):
                        z = self.model(x)  # forward
                        L = self.loss(z, y)
                        # store the loss and do backward after we have done forward
                        # on all GPUs for better speed on multiple GPUs.
                        Ls.append(L)
                        outputs.append(z)
                    ag.backward(Ls)
                self.trainer.step(batch.data[0].shape[0])
                self.metrics.update(label, outputs)
                if "log_interval" in self.config and not (i + 1) % self.config["log_interval"]:
                    # This would print on driver for each pid.
                    print_output = ""
                    print_output += 'Epoch[%d] Batch[%d]  Speed: %f samples/sec %s=%f' % (
                        self.epoch, i, self.config["batch_size"] / (time.time() - btic), "loss", Ls[0].asnumpy().mean())
                    names, accs = self.metrics.get()
                    if not isinstance(names, list):
                        names = [names]
                        accs = [accs]
                    for name, acc in zip(names, accs):
                        print_output += ' %s=%f' % (name, acc)
                    print(print_output)
                btic = time.time()
            epoch_time = time.time() - tic
            stats["epoch_time"] = epoch_time
            names, accs = self.metrics.get()
            if not isinstance(names, list):
                names = [names]
                accs = [accs]
            for name, acc in zip(names, accs):
                stats[name] = acc
        return stats

    def validate(self):
        stats = dict()
        stats["epoch"] = self.epoch
        if self.is_worker:
            self.metrics.reset()
            self.test_dataset.reset()
            for batch in self.test_dataset:
                import mxnet as mx
                from mxnet import gluon
                data = gluon.utils.split_and_load(batch.data[0].astype("float32", copy=False),
                                                  ctx_list=[mx.cpu()], batch_axis=0)
                label = gluon.utils.split_and_load(batch.label[0].astype("float32", copy=False),
                                                   ctx_list=[mx.cpu()], batch_axis=0)
                outputs = [self.model(X) for X in data]
                self.metrics.update(label, outputs)
            names, accs = self.metrics.get()
            if not isinstance(names, list):
                names = [names]
                accs = [accs]
            for name, acc in zip(names, accs):
                stats[name] = acc
        return stats

    def shutdown(self):
        """Attempts to shut down the runner."""
        if self.is_worker:
            del self.model
            del self.train_dataset
            del self.test_dataset
            del self.kv
            del self.trainer
            del self.loss
        # TODO: also delete downloaded data as well?

    def get_node_ip(self):
        """Returns the IP address of the current node."""
        return ray.services.get_node_ip_address()

    def find_free_port(self):
        """Finds a free port on the current node."""
        return find_free_port()


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class MXNetTrainer(object):
    def __init__(self,
                 data_creator,
                 model_creator,
                 loss_creator,
                 metrics_creator,
                 config,
                 # Specify cpu resources for actors so that two actors won't use the same raylet.
                 worker_cpus=None):
        self.data_creator = data_creator
        self.model_creator = model_creator
        self.loss_creator = loss_creator
        self.metrics_creator = metrics_creator
        self.config = config
        self.num_workers = config["num_workers"]
        self.num_servers = config["num_servers"] if "num_servers" in self.config else self.num_workers
        self.num_runners = self.num_servers + self.num_workers

        # Generate actor class
        Runner = ray.remote(num_cpus=worker_cpus)(MXNetRunner) if worker_cpus else ray.remote(MXNetRunner)

        # Start runners
        self.runners = [
            Runner.remote(
                self.data_creator,
                self.model_creator,
                self.loss_creator,
                self.metrics_creator,
                self.config)
            for i in range(self.num_runners)
        ]

        # Compute URL for initializing distributed setup
        # ips = ray.get(
        #     [runner.get_node_ip.remote() for runner in self.runners])
        # ports = ray.get(
        #     [runner.find_free_port.remote() for runner in self.runners])

        env = {
            "DMLC_PS_ROOT_URI": str(get_host_ip()),
            "DMLC_PS_ROOT_PORT": str(find_free_port()),
            "DMLC_NUM_SERVER": str(self.num_servers),
            "DMLC_NUM_WORKER": str(self.num_workers),
        }
        envs = []
        for i in range(self.num_workers + self.num_servers):
            current_env = env.copy()
            current_env['DMLC_ROLE'] = 'server' if i < self.num_servers else 'worker'
            envs.append(current_env)

        env['DMLC_ROLE'] = 'scheduler'
        modified_env = os.environ.copy()
        modified_env.update(env)
        # Need to contain system env to run bash
        subprocess.Popen("python -c 'import mxnet'", shell=True, env=modified_env)

        ray.get([
            runner.setup_distributed.remote(envs[i])
            for i, runner in enumerate(self.runners)
        ])

    def train(self):
        """Runs a training epoch."""
        stats = ray.get([w.step.remote() for w in self.runners])
        return stats

    def validate(self):
        """Evaluates the model on the validation data set."""
        stats = ray.get([w.validate.remote() for w in self.runners])
        return stats

    def shutdown(self):
        """Shuts down runners and releases resources."""
        for runner in self.runners:
            runner.shutdown.remote()
            runner.__ray_terminate__.remote()

# TODO: add model save and restore
