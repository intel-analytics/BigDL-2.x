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

import numpy as np
import logging
import pickle

import ray

from zoo.orca.learn.tf.tf_runner import TFRunner
from zoo.orca.learn.horovod.horovod_ray_runner import HorovodWorker
from zoo.orca.learn.horovod.horovod_ray_runner import HorovodRayRunner
from zoo.ray import RayContext

logger = logging.getLogger(__name__)


class TFWorker(HorovodWorker, TFRunner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TFRayEstimator(HorovodRayRunner):
    def __init__(self,
                 model_creator,
                 compile_args,
                 data_creator,
                 config=None,
                 verbose=False,
                 backend="tf"):
        """Sets up the TensorFlow trainer.

        Args:
            model_creator (dict -> Model): This function takes in the `config`
                dict and returns a compiled TF model.
            data_creator (dict -> tf.Dataset, tf.Dataset): Creates
                the training and validation data sets using the config.
                `config` dict is passed into the function.
            config (dict): configuration passed to 'model_creator',
                'data_creator'. Also contains `fit_config`, which is passed
                into `model.fit(data, **fit_config)` and
                `evaluate_config` which is passed into `model.evaluate`.
            num_replicas (int): Sets number of workers used in distributed
                training. Workers will be placed arbitrarily across the
                cluster.
            use_gpu (bool): Enables all workers to use GPU.
            verbose (bool): Prints output of one model if true.
        """
        self.model_creator = model_creator
        self.data_creator = data_creator
        self.compile_args = compile_args
        self.config = {} if config is None else config
        self.verbose = verbose

        params = {
            "model_creator": model_creator,
            "data_creator": data_creator,
            "compile_args": compile_args,
            "config": self.config,
            "verbose": self.verbose,
        }

        if "inter_op_parallelism" not in self.config:
            self.config["inter_op_parallelism"] = 1

        if "intra_op_parallelism" not in config:
            self.config["intra_op_parallelism"] = self.cores_per_node

        super().__init__(RayContext.get(), worker_cls=TFWorker, worker_param=params)

        # Generate actor class
        # todo: are these resource quotas right?
        # should they be exposed to the client codee?
        # Compute URL for initializing distributed setup
        if backend == "tf":
            ips = ray.get(
                [worker.get_node_ip.remote() for worker in self.remote_workers])
            ports = ray.get(
                [worker.find_free_port.remote() for worker in self.remote_workers])

            urls = ["{ip}:{port}".format(ip=ips[i], port=ports[i])
                    for i in range(len(self.remote_workers))]

            # Get setup tasks in order to throw errors on failure
            ray.get([
                worker.setup_distributed.remote(urls, i, len(self.remote_workers))
                for i, worker in enumerate(self.remote_workers)])
        elif backend == "horovod":
            ray.get([
                worker.setup_horovod.remote()
                for i, worker in enumerate(self.remote_workers)])
        else:
            raise Exception("Only \"tf\" and \"horovod\" are legal "
                            "value of backend, but got {}".format(backend))

    def fit(self):
        """Runs a training epoch."""

        # see ./tf_runner.py:setup_distributed
        # for an explanation of only taking the first worker's data
        worker_stats = ray.get([w.step.remote() for w in self.remote_workers])
        stats = worker_stats[0].copy()
        return stats

    def evaluate(self):
        """Evaluates the model on the validation data set."""
        logger.info("Starting validation step.")

        # see ./tf_runner.py:setup_distributed
        # for an explanation of only taking the first worker's data
        stats = ray.get([w.validate.remote() for w in self.remote_workers])
        stats = stats[0].copy()
        return stats

    def get_model(self):
        """Returns the learned model."""
        state = ray.get(self.remote_workers[0].get_state.remote())
        return self._get_model_from_state(state)

    def save(self, checkpoint):
        """Saves the model at the provided checkpoint.

        Args:
            checkpoint (str): Path to target checkpoint file.

        """

        state = ray.get(self.remote_workers[0].get_state.remote())

        with open(checkpoint, "wb") as f:
            pickle.dump(state, f)

        return checkpoint

    def restore(self, checkpoint):
        """Restores the model from the provided checkpoint.

        Args:
            checkpoint (str): Path to target checkpoint file.

        """
        with open(checkpoint, "rb") as f:
            state = pickle.load(f)

        state_id = ray.put(state)
        ray.get([worker.set_state.remote(state_id) for worker in self.remote_workers])

    def shutdown(self):
        """Shuts down workers and releases resources."""
        for worker in self.remote_workers:
            worker.shutdown.remote()
            worker.__ray_terminate__.remote()

    def _get_model_from_state(self, state):
        """Creates model and load weights from state"""

        model = self.model_creator(self.config)
        model.set_weights(state["weights"])

        # This part is due to ray.get() changing scalar np.int64 object to int
        state["optimizer_weights"][0] = np.array(
            state["optimizer_weights"][0], dtype=np.int64)

        if model.optimizer.weights == []:
            model._make_train_function()
        model.optimizer.set_weights(state["optimizer_weights"])

        return model
