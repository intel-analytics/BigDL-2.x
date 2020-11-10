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

import io
import types
import logging
import numbers
import torch
import numpy as np

from zoo.orca.learn.pytorch.training_operator import TrainingOperator
from zoo.orca.learn.pytorch.torch_runner import TorchRunner
from zoo.ray import RayContext

import ray
from ray.exceptions import RayActorError

logger = logging.getLogger(__name__)


def check_for_failure(remote_values):
    """Checks remote values for any that returned and failed.
    :param remote_values: List of object IDs representing functions
            that may fail in the middle of execution. For example, running
            a SGD training loop in multiple parallel actor calls.
    :return Bool for success in executing given remote tasks.
    """
    unfinished = remote_values
    try:
        while len(unfinished) > 0:
            finished, unfinished = ray.wait(unfinished)
            finished = ray.get(finished)
        return True
    except RayActorError as exc:
        logger.exception(str(exc))
    return False


def shards_ref_to_creator(shards_ref):

    def data_creator(config):
        from zoo.orca.data.utils import ray_partition_get_data_label, index_data, get_size
        from torch.utils.data import Dataset, DataLoader

        class NDArrayDataset(Dataset):
            def __init__(self, x, y):
                self.x = x  # features
                self.y = y  # labels

            def __len__(self):
                return get_size(self.y)

            def __getitem__(self, i):
                return index_data(self.x, i), index_data(self.y, i)

        # TODO: refactor batch_size in fit
        assert "batch_size" in config, "batch_size must be set in config"
        params = {"batch_size": config["batch_size"], "shuffle": True}
        for arg in ["shuffle", "sampler", "batch_sampler", "num_workers", "collate_fn", "pin_memory",
                    "drop_last", "timeout", "worker_init_fn", "multiprocessing_context"]:
            if arg in config:
                params[arg] = config[arg]
        data, label = ray_partition_get_data_label(ray.get(shards_ref),
                                                   allow_tuple=False,
                                                   allow_list=False)
        print("Data size on worker: ", len(label))
        dataset = NDArrayDataset(data, label)
        data_loader = DataLoader(dataset, **params)
        return data_loader

    return data_creator


class PyTorchRayEstimator:
    def __init__(
            self,
            *,
            model_creator,
            optimizer_creator,
            loss_creator=None,
            scheduler_creator=None,
            training_operator_cls=TrainingOperator,
            initialization_hook=None,
            config=None,
            scheduler_step_freq="batch",
            use_tqdm=False,
            backend="pytorch",
            workers_per_node=1):

        # todo remove ray_ctx to run on workers
        ray_ctx = RayContext.get()
        if not (isinstance(model_creator, types.FunctionType) and
                isinstance(optimizer_creator, types.FunctionType)):  # Torch model is also callable.
            raise ValueError(
                "Must provide a function for both model_creator and optimizer_creator")

        self.model_creator = model_creator
        self.optimizer_creator = optimizer_creator
        self.loss_creator = loss_creator
        self.scheduler_creator = scheduler_creator
        self.training_operator_cls = training_operator_cls
        self.scheduler_step_freq = scheduler_step_freq
        self.use_tqdm = use_tqdm

        if not training_operator_cls and not loss_creator:
            raise ValueError("If a loss_creator is not provided, you must "
                             "provide a custom training operator.")

        self.initialization_hook = initialization_hook
        self.config = {} if config is None else config
        worker_config = self.config.copy()
        params = dict(
            model_creator=self.model_creator,
            optimizer_creator=self.optimizer_creator,
            loss_creator=self.loss_creator,
            scheduler_creator=self.scheduler_creator,
            training_operator_cls=self.training_operator_cls,
            scheduler_step_freq=self.scheduler_step_freq,
            use_tqdm=self.use_tqdm,
            config=worker_config)

        if backend == "pytorch":
            cores_per_node = ray_ctx.ray_node_cpu_cores // workers_per_node
            num_nodes = ray_ctx.num_ray_nodes * workers_per_node
            RemoteRunner = ray.remote(num_cpus=cores_per_node)(TorchRunner)
            self.remote_workers = [
                RemoteRunner.remote(**params) for i in range(num_nodes)
            ]
            ray.get([
                worker.setup.remote(cores_per_node)
                for i, worker in enumerate(self.remote_workers)
            ])

            head_worker = self.remote_workers[0]
            address = ray.get(head_worker.setup_address.remote())

            logger.info(f"initializing pytorch process group on {address}")

            ray.get([
                worker.setup_torch_distribute.remote(address, i, num_nodes)
                for i, worker in enumerate(self.remote_workers)
            ])

        elif backend == "horovod":
            from zoo.orca.learn.horovod.horovod_ray_runner import HorovodRayRunner
            self.horovod_runner = HorovodRayRunner(ray_ctx,
                                                   worker_cls=TorchRunner,
                                                   worker_param=params,
                                                   workers_per_node=workers_per_node)
            self.remote_workers = self.horovod_runner.remote_workers
            cores_per_node = self.horovod_runner.cores_per_node
            ray.get([
                worker.setup.remote(cores_per_node)
                for i, worker in enumerate(self.remote_workers)
            ])

            ray.get([
                worker.setup_horovod.remote()
                for i, worker in enumerate(self.remote_workers)
            ])
        else:
            raise Exception("Only \"pytorch\" and \"horovod\" are legal "
                            "values of backend, but got {}".format(backend))
        self.num_workers = len(self.remote_workers)

    def train(self,
              data,
              epochs=1,
              profile=False,
              reduce_results=True,
              info=None):
        """
        Trains a PyTorch model given training data for several epochs.

        Calls `operator.train_epoch()` on N parallel workers simultaneously
        underneath the hood.
        :param data: An instance of SparkXShards or a function that takes config as
        argument and returns a PyTorch DataLoader for training.
        :param epochs: (int) Number of epochs to train the model.
        :param profile: (bool) Returns time stats for the training procedure.
        :param reduce_results: (bool) Whether to average all metrics across
                all workers into one dict. If a metric is a non-numerical
                value (or nested dictionaries), one value will be randomly
                selected among the workers. If False, returns a list of dicts.
        :param info: (dict) Optional dictionary passed to the training
                operator for ``train_epoch`` and ``train_batch``.

        :return
            (dict | list) A dictionary of metrics for training.
                You can provide custom metrics by passing in a custom
                ``training_operator_cls``. If ``reduce_results=False``,
                this will return a list of metric dictionaries whose
                length will be equal to ``num_workers``.
        """
        from zoo.orca.data import SparkXShards
        if isinstance(data, SparkXShards):
            from zoo.orca.data.utils import process_spark_xshards
            ray_xshards = process_spark_xshards(data, self.num_workers)

            def transform_func(worker, shards_ref):
                data_creator = shards_ref_to_creator(shards_ref)
                # Should not wrap DistributedSampler on DataLoader for SparkXShards input.
                return worker.train_epochs.remote(data_creator, epochs, profile, info, False)

            stats_shards = ray_xshards.transform_shards_with_actors(self.remote_workers,
                                                                    transform_func,
                                                                    gang_scheduling=True)
            worker_stats = stats_shards.collect_partitions()
        else:
            assert isinstance(data, types.FunctionType), \
                "data should be either an instance of SparkXShards or a callable function"

            success, worker_stats = self._train_epochs(data,
                                                       epochs=epochs,
                                                       profile=profile,
                                                       info=info)

        epoch_stats = list(map(list, zip(*worker_stats)))
        if reduce_results:
            for i in range(len(epoch_stats)):
                epoch_stats[i] = self._process_stats(epoch_stats[i])
            return epoch_stats
        else:
            return epoch_stats

    def _process_stats(self, worker_stats):
        stats = {
            "num_samples": sum(
                stats.pop("num_samples", np.nan) for stats in worker_stats)
        }

        for stat_key in worker_stats[0]:
            if isinstance(worker_stats[0], numbers.Number):
                stats[stat_key] = np.nanmean(
                    [s.get(stat_key, np.nan) for s in worker_stats])
            else:
                stats[stat_key] = worker_stats[0][stat_key]
        return stats

    def _train_epochs(self, data_creator, epochs=1, profile=False, info=None):
        params = dict(data_creator=data_creator, epochs=epochs, profile=profile, info=info)
        remote_worker_stats = []
        for i, w in enumerate(self.remote_workers):
            stats = w.train_epochs.remote(**params)
            remote_worker_stats.append(stats)

        success = check_for_failure(remote_worker_stats)
        if success:
            return success, ray.get(remote_worker_stats)
        else:
            return success, None

    def validate(self, data_creator, num_steps=None, profile=False, info=None):
        """Evaluates the model on the validation data set.

        :param data_creator: (callable) a funtion that takes a config dict as input
                  and return a data loader containing the validation data.
        :param num_steps: (int) Number of batches to compute update steps on.
               This corresponds also to the number of times
                ``TrainingOperator.validate_batch`` is called.
        :param profile: (bool) Returns time stats for the evaluation procedure.
        :param info: (dict) Optional dictionary passed to the training
                operator for `validate` and `validate_batch`.
        :return: A dictionary of metrics for validation.
                You can provide custom metrics by passing in a custom
                ``training_operator_cls``.
        """
        if not callable(data_creator):
            raise ValueError(
                "Must provide a callable data_creator, "
                "but got a data_creator of type: {}".format(type(data_creator)))

        params = dict(data_creator=data_creator,
                      num_steps=num_steps,
                      profile=profile,
                      info=info)

        remote_worker_stats = [
            w.validate.remote(**params) for w in self.remote_workers
        ]
        return self._process_stats(ray.get(remote_worker_stats))

    def get_model(self):
        """Returns the learned model(s)."""
        state = self.get_state_dict()
        model = self.model_creator(self.config)
        model_state = state["models"][0]
        model.load_state_dict(model_state)
        return model.module if hasattr(model, "module") else model

    def save(self, checkpoint):
        """Saves the Estimator state to the provided checkpoint path.

        :param checkpoint: (str) Path to target checkpoint file.
        """
        state_dict = self.get_state_dict()
        torch.save(state_dict, checkpoint)
        return checkpoint

    def get_state_dict(self):
        stream_ids = [
            worker.state_stream.remote()
            for worker in self.remote_workers
        ]
        # get the first task id that finished executing.
        [stream_id], stream_ids = ray.wait(stream_ids, num_returns=1, timeout=None)
        byte_obj = ray.get(stream_id)
        _buffer = io.BytesIO(byte_obj)
        state_dict = torch.load(
            _buffer,
            map_location="cpu")
        return state_dict

    def load(self, checkpoint):
        """Loads the Estimator and all workers from the provided checkpoint.

        :param checkpoint: (str) Path to target checkpoint file.
        """
        state_dict = torch.load(checkpoint)
        self.load_state_dict(state_dict)

    def load_state_dict(self, state_dict, blocking=True):
        _buffer = io.BytesIO()
        torch.save(state_dict, _buffer)
        state_stream = _buffer.getvalue()
        state_id = ray.put(state_stream)

        remote_calls = [
            worker.load_state_stream.remote(state_id)
            for worker in self.remote_workers
        ]
        if blocking:
            ray.get(remote_calls)

    def shutdown(self, force=False):
        """Shuts down workers and releases resources."""
        if not force:
            cleanup = [
                worker.shutdown.remote() for worker in self.remote_workers
            ]
            try:
                ray.get(cleanup)
                [
                    worker.__ray_terminate__.remote()
                    for worker in self.remote_workers
                ]
            except RayActorError:
                logger.warning(
                    "Failed to shutdown gracefully, forcing a shutdown.")

                for worker in self.remote_workers:
                    logger.warning("Killing worker {}.".format(worker))
                    ray.kill(worker)
        else:
            for worker in self.remote_workers:
                logger.debug("Killing worker {}.".format(worker))
                ray.kill(worker)

        self.remote_workers = []
