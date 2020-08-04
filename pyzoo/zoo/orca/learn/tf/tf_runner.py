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

# Copyright 2017 The Ray Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import json
import os

import ray
import ray.services
from contextlib import closing
import logging
import socket
logger = logging.getLogger(__name__)


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _try_import_strategy():
    """Late import for Tesnorflow"""
    import tensorflow as tf
    return tf.distribute.experimental.MultiWorkerMirroredStrategy


class TFRunner:
    """Manages a TensorFlow model for training."""

    def __init__(self, model_creator, data_creator, compile_args_creator,
                 config=None,
                 verbose=False):
        """Initializes the runner.
        Args:
            model_creator (dict -> Model): see tf_trainer.py.
            data_creator (dict -> tf.Dataset, tf.Dataset): see tf_trainer.py.
            config (dict): see tf_trainer.py.
            verbose (bool): Outputs training data if true.
        """

        self.model_creator = model_creator
        self.data_creator = data_creator
        self.compile_args_creator = compile_args_creator
        self.config = {} if config is None else config
        self.inter_op_parallelism = self.config.get("inter_op_parallelism", 1)
        self.intra_op_parallelism = self.config.get("intra_op_parallelism", 1)
        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(self.inter_op_parallelism)
        tf.config.threading.set_intra_op_parallelism_threads(self.intra_op_parallelism)
        os.environ["OMP_NUM_THREADS"] = self.config.get("OMP_NUM_THREADS", str(self.intra_op_parallelism))
        os.environ["KMP_BLOCKING_TIME"] = self.config.get("KMP_BLOCKING_TIME",
                                                          os.environ.get("KMP_BLOCKING_TIME", "0"))

        self.epoch = 0
        self.verbose = verbose

    def setup(self):
        """Initializes the model."""
        logger.debug("Creating dataset")
        self.train_dataset, self.test_dataset = self.data_creator(self.config)

        logger.debug("Creating model")
        self.model = self.model_creator(self.config)
        self.model.compile(**self.compile_args_creator(self.config))
        self.backend = "tf-local"

    def setup_horovod(self):
        import horovod.tensorflow.keras as hvd
        hvd.init()
        train_dataset, test_dataset = self.data_creator(self.config)
        self.train_dataset = train_dataset.shard(hvd.size(), hvd.rank())
        self.test_dataset = test_dataset.shard(hvd.size(), hvd.rank())

        self.model = self.model_creator(self.config)
        compile_args = self.compile_args_creator(self.config)
        compile_args["optimizer"] = hvd.DistributedOptimizer(compile_args["optimizer"])

        self.model.compile(**compile_args)
        self.backend = "horovod"

    def setup_distributed(self, urls, world_rank, world_size):
        """Sets up TensorFLow distributed environment and initializes the model.
        Args:
            urls (str): the URLs that each node uses to connect.
            world_rank (int): the index of the runner.
            world_size (int): the total number of runners.
        """
        assert len(urls) == world_size
        tf_config = {
            "cluster": {
                "worker": urls
            },
            "task": {
                "index": world_rank,
                "type": "worker"
            }
        }
        os.environ["TF_CONFIG"] = json.dumps(tf_config)

        MultiWorkerMirroredStrategy = _try_import_strategy()

        # MultiWorkerMirroredStrategy handles everything for us, from
        # sharding the dataset (or even sharding the data itself if the loader
        # reads files from disk) to merging the metrics and weight updates
        #
        # worker 0 is the "chief" worker and will handle the map-reduce
        # every worker ends up with the exact same metrics and model
        # after model.fit
        #
        # because of this, we only really ever need to query its state
        self.strategy = MultiWorkerMirroredStrategy()

        self.train_dataset, self.test_dataset = self.data_creator(self.config)

        logger.debug("Creating model with MultiWorkerMirroredStrategy")
        with self.strategy.scope():
            self.model = self.model_creator(self.config)
            self.model.compile(**self.compile_args_creator(self.config))

        # For use in model.evaluate()
        self.local_model = None
        self.backend = "tf-distributed"

    def step(self):
        """Runs a training epoch and updates the model parameters."""
        fit_default_config = {"verbose": self.verbose}
        fit_default_config.update(self.config.get("fit_config", {}))

        if self.backend == "horovod":
            import horovod.tensorflow.keras as hvd
            bcast_callback = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]
            if hvd.rank() == 0:
                fit_default_config["verbose"] = 1
            else:
                fit_default_config["verbose"] = 0

            if "callbacks" in fit_default_config:
                callbacks = fit_default_config["callbacks"]
                fit_default_config["callbacks"] = bcast_callback + callbacks
            else:
                fit_default_config["callbacks"] = bcast_callback

        history = self.model.fit(self.train_dataset, **fit_default_config)
        if history is None:
            stats = {}
        else:
            stats = {"train_" + k: v[-1] for k, v in history.history.items()}

        self.epoch += 1
        return stats

    def validate(self):
        """Evaluates the model on the validation data set."""
        stats = {}
        evaluate_config = {"verbose": self.verbose}
        evaluate_config.update(self.config.get("evaluate_config", {}))

        if self.backend == "horovod":
            import horovod.tensorflow.keras as hvd
            if hvd.rank() == 0:
                evaluate_config["verbose"] = 1
            else:
                evaluate_config["verbose"] = 0

        results = self.model.evaluate(self.test_dataset, **evaluate_config)
        if results is None:
            # Using local Model since model.evaluate() returns None
            # for MultiWorkerMirroredStrategy
            logger.warning("Running a local model to get validation score.")
            self.local_model = self.model_creator(self.config)
            self.local_model.set_weights(self.model.get_weights())
            results = self.local_model.evaluate(self.test_dataset,
                                                **evaluate_config)

        if isinstance(results, list):
            stats = {
                "validation_" + k: v
                for k, v in zip(self.model.metrics_names, results)
            }
        else:
            stats = {"loss": results}

        return stats

    def get_state(self):
        """Returns the state of the runner."""
        return {
            "epoch": self.epoch,
            "weights": self.model.get_weights(),
            "optimizer_weights": self.model.optimizer.get_weights()
        }

    def set_state(self, state):
        """Sets the state of the model."""

        self.model = self.model_creator(self.config)
        self.epoch = state["epoch"]
        self.model.set_weights(state["weights"])

    def shutdown(self):
        """Attempts to shut down the worker."""
        del self.model
        del self.train_dataset
        del self.test_dataset

    def get_node_ip(self):
        """Returns the IP address of the current node."""
        return ray.services.get_node_ip_address()

    def find_free_port(self):
        """Finds a free port on the current node."""
        return find_free_port()
