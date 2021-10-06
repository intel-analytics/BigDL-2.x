import json
import logging
import os

from re import VERBOSE
from subprocess import call
from sys import version

from pyspark import BarrierTaskContext
from pyspark.context import SparkContext
import tensorflow as tf
from numpy import array
from contextlib import closing
import socket

from zoo.orca.data.utils import ray_partition_get_data_label


def find_free_port(tc):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        tc.barrier()
        return f"{s.getsockname()[0]}:{s.getsockname()[1]}"

def handle_datasets_train(data_creator, validation_data_creator):   
        train_dataset = data_creator()
        if validation_data_creator is not None:
            test_dataset = validation_data_creator()
        else:
            test_dataset = None
        return train_dataset, test_dataset

class DatasetHandler:

    def __init__(self, rank, size):
        self.rank = rank
        self.size = size

    def handle_datasets_train(self, data_creator,
                              validation_data_creator,
                              config, epochs, steps_per_epoch,
                              validation_steps):

        config, local_batch_size = self._handle_batch_size(config)
        config['rank'] = self.rank
        config['size'] = self.size
        train_dataset = data_creator(config, config["batch_size"])
        if isinstance(train_dataset, list) and \
            all([isinstance(x, dict) for x in train_dataset]):
            assert steps_per_epoch is not None, "steps_per_epoch must be provided for xshard"
            train_dataset = self._handle_xshards(train_dataset,
                                                 steps=steps_per_epoch * epochs,
                                                 local_batch_size=local_batch_size,
                                                 shuffle=True)
        else:
            train_dataset = self._handle_sharding(train_dataset)

        if validation_data_creator is not None:
            test_dataset = validation_data_creator(config, config["batch_size"])
            if isinstance(test_dataset, list) and \
                    all([isinstance(x, dict) for x in test_dataset]):
                assert validation_steps is not None, "validation_steps must be provided" \
                                                     "when use xshards for evaluate"
                test_dataset = self._handle_xshards(test_dataset,
                                                    steps=validation_steps,
                                                    local_batch_size=local_batch_size,
                                                    shuffle=False)
            else:
                test_dataset = self._handle_sharding(test_dataset)
        else:
            test_dataset = None

        return train_dataset, test_dataset

    def handle_dataset_validation(self, data_creator, config, steps):
        config, local_batch_size = self._handle_batch_size(config)
        config['rank'] = self.rank
        config['size'] = self.size
        dataset = data_creator(config, config["batch_size"])
        if isinstance(dataset, list) and all([isinstance(x, dict) for x in dataset]):
            assert steps is not None, "steps must be provided for xshard"
            dataset = self._handle_xshards(dataset,
                                           steps=steps,
                                           local_batch_size=local_batch_size,
                                           shuffle=False)
        else:
            dataset = self._handle_sharding(dataset)

        return dataset

    def _handle_xshards(self, dataset, steps, local_batch_size, shuffle):
        raise NotImplementedError

    def _handle_sharding(self, dataset):
        raise NotImplementedError

    def _handle_batch_size(self, config):
        raise NotImplementedError

    @staticmethod
    def get_handler(backend, rank, size):

        if backend == "tf-distributed":
            return TFDistributedDatasetHandler(rank, size)

        if backend == "tf-local":
            return LocalDatasetHandler(rank, size)

        raise Exception(f"invalid backend: {backend}")


class TFDistributedDatasetHandler(DatasetHandler):

    def _handle_xshards(self, dataset, steps, local_batch_size, shuffle):
        import tensorflow as tf

        data, label = ray_partition_get_data_label(dataset,
                                                    allow_tuple=True,
                                                    allow_list=False)
        def dataset_fn(input_context):
            dataset = tf.data.Dataset.from_tensor_slices((data, label))
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy = \
                tf.data.experimental.AutoShardPolicy.OFF
            dataset = dataset.with_options(options)
            dataset = dataset.repeat()
            dataset = dataset.take(steps * local_batch_size)
            if shuffle:
                dataset = dataset.shuffle(local_batch_size * min(steps, 10))
            dataset = dataset.batch(local_batch_size)
            return dataset

        from tensorflow.python.distribute import distribution_strategy_context as ds_context
        strategy = ds_context.get_strategy()
        dataset = strategy.experimental_distribute_datasets_from_function(dataset_fn)
        return dataset

    def _handle_sharding(self, dataset):
        return dataset

    def _handle_batch_size(self, config):
        assert "batch_size" in config, "batch_size must be set in config"
        local_batch_size = config["batch_size"] // self.size
        return config, local_batch_size


class LocalDatasetHandler(DatasetHandler):

    def _handle_xshards(self, dataset, steps, local_batch_size, shuffle):
        import tensorflow as tf
        data, label = ray_partition_get_data_label(dataset,
                                                    allow_tuple=True,
                                                    allow_list=False)
        dataset = tf.data.Dataset.from_tensor_slices((data, label))
        dataset = dataset.repeat()
        dataset = dataset.take(steps * local_batch_size)
        if shuffle:
            dataset = dataset.shuffle(local_batch_size * min(steps, 10))
        dataset = dataset.batch(local_batch_size)
        return dataset

    def _handle_sharding(self, dataset):
        return dataset

    def _handle_batch_size(self, config):
        assert "batch_size" in config, "batch_size must be set in config"
        return config, config["batch_size"]



class SparkRunner:
    def __init__(self, model_creator, compile_args_creator,
                 size,
                 config=None,
                 verbose=False,
                 model_weights=None,
                 backend="tf-distributed",
                 mode="fit"
                ):
        """Initializes the runner.
                Args:
                    model_creator (dict -> Model): see tf_trainer.py.
                    data_creator (dict -> tf.Dataset, tf.Dataset): see tf_trainer.py.
                    config (dict): see tf_trainer.py.
                    verbose (bool): Outputs training data if true.
                """

        self.model_creator = model_creator
        self.compile_args_creator = compile_args_creator
        self.config = {} if config is None else config
        self.inter_op_parallelism = self.config.get("inter_op_parallelism", 1)
        self.intra_op_parallelism = self.config.get("intra_op_parallelism", 1)
        self.epoch = 0
        self.verbose = verbose
        self.model_weights = model_weights
        self.size = size
        self.mode = mode
        self.backend = backend
        if self.backend == "tf-distributed":
            if mode == "fit" or mode == "evaluate":
                self.setup_distributed(self.mode)

    def setup_distributed(self, mode):
        """Sets up TensorFLow distributed environment and initializes the model.
        Args:
            urls (str): the URLs that each node uses to connect.
            world_rank (int): the index of the runner.
            world_size (int): the total number of runners.
        """
        tc = BarrierTaskContext().get()
        rank = tc.partitionId()
        free_port = find_free_port(tc)
        cluster = tc.allGather(str(free_port))
        self.cluster = cluster
        self.rank = rank
        print("cluster is: ", cluster)

        import os
        os.environ["TF_CONFIG"] = json.dumps({
            'cluster': {
                'worker': cluster
            },
            'task': {'type': 'worker', 'index': rank}
        })
        ips = set([node.split(":")[0] for node in cluster])
        os.environ["no_proxy"] = ",".join(ips)

        from tensorflow.python.distribute import distribution_strategy_context as ds_context
        # self.strategy = ds_context.get_strategy()
        # print("stragety: ", self.strategy)
        # if not self.strategy:
        #     self.strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        if mode == "fit":
            self.strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        else:
            from tensorflow.python.distribute import distribution_strategy_context as ds_context
            self.strategy = ds_context.get_strategy()

        # with self.strategy.scope():
        #     model = self.model_creator(self.config)


        # For use in model.evaluate()
        self.local_model = None
        self.backend = "tf-distributed"
        # self.size = size


    def distributed_train_func(self, data_creator, config, epochs=1, verbose=1,
             callbacks=None, validation_data_creator=None, class_weight=None,
             steps_per_epoch=None, validation_steps=None, validation_freq=1):
        """
        Sets up TensorFLow distributed environment, initializes the model,
        runs a training epoch and updates the model parameters
        """
        # tc = BarrierTaskContext().get()
        # rank = tc.partitionId()
        # free_port = find_free_port(tc)
        # cluster = tc.allGather(str(free_port))
        # self.cluster = cluster
        # self.rank = rank
        # print(cluster)
        #
        # import os
        # os.environ["TF_CONFIG"] = json.dumps({
        # 'cluster': {
        #     'worker': cluster
        # },
        # 'task': {'type': 'worker', 'index': rank}
        # })
        # ips = set([node.split(":")[0] for node in cluster])
        # os.environ["no_proxy"] = ",".join(ips)
        #
        # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        with self.strategy.scope():
            model = self.model_creator(self.config)
            dataset_handler = DatasetHandler.get_handler(self.backend, self.rank, self.size)
            train_dataset, test_dataset = dataset_handler \
                .handle_datasets_train(data_creator=data_creator,
                                       validation_data_creator=validation_data_creator,
                                       config=config, epochs=epochs,
                                       steps_per_epoch=steps_per_epoch,
                                       validation_steps=validation_steps)

        history = model.fit(train_dataset,
                                 epochs=epochs,
                                 verbose=verbose,
                                 callbacks=callbacks,
                                 validation_data=test_dataset,
                                 class_weight=class_weight,
                                 # initial_epoch=epoch,
                                 steps_per_epoch=steps_per_epoch,
                                 validation_steps=validation_steps,
                                 validation_freq=validation_freq)

        return (model, history)
        
    def step(self, data_creator, epochs=1, batch_size=32, verbose=1,
             callbacks=None, validation_data_creator=None, class_weight=None,
             steps_per_epoch=None, validation_steps=None, validation_freq=1,
             data_config=None):
        """
        Get model training results and new model.
        """
        config = self.config.copy()
        if data_config is not None:
            config.update(data_config)
        config["batch_size"] = batch_size

        model, history = self.distributed_train_func(data_creator, config, epochs, verbose,
             callbacks=callbacks, validation_data_creator=validation_data_creator, class_weight=class_weight,
             steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, validation_freq=validation_freq)

        weights = model.get_weights()
        if history is None:
            stats = {}
        else:
            stats = {k: v[-1] for k, v in history.history.items()}
        if self.rank == 0:
            # if model_dir is not None:
            #     model.save_weights(model_dir)
            return ([weights, stats])
        else:
            return []
    
    def validate(self, data_creator, batch_size=32, verbose=1, sample_weight=None,
                 steps=None, callbacks=None, data_config=None):
        """
        Evaluates the model on the validation data set.
        """
        config = self.config.copy()
        if data_config is not None:
            config.update(data_config)
        config["batch_size"] = batch_size

        # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        with self.strategy.scope():
            model = self.model_creator(self.config)
            model.set_weights(self.model_weights)

        with self.strategy.scope():
            dataset_handler = DatasetHandler.get_handler(self.backend,
                                                         self.rank,
                                                         self.size)

            dataset = dataset_handler.handle_dataset_validation(data_creator,
                                                                config=config,
                                                                steps=steps)

        params = dict(
            verbose=verbose,
            sample_weight=sample_weight,
            steps=steps,
            callbacks=callbacks,
        )
        results = model.evaluate(dataset, **params)

        if results is None:
            model_weights = self.model_weights
            local_model = self.model_creator(self.config)
            local_model = local_model.set_weights(model_weights)
            results = local_model.evaluate(dataset, **params)
        
        if isinstance(results, list):
            stats = {
                "validation_" + k: v
                for k, v in zip(model.metrics_names, results)
            }
        else:
            stats = {"results": results}
        
        if self.rank == 0:
            return [stats]
        else:
            return []


    def predict(self, data_creator, batch_size, verbose, steps, callbacks, data_config):
        config = self.config.copy()
        if data_config is not None:
            config.update(data_config)

        dataset = data_creator(config, batch_size)
        partition = dataset
        params = dict(
            batch_size=batch_size,
            verbose=verbose,
            steps=steps,
            callbacks=callbacks,
        )

        if self.backend == "tf-distributed":
            local_model = self.model_creator(self.config)
            if self.model_weights:
                local_model.set_weights(self.model_weights)
        else:
            local_model = self.model_creator(self.config)

        def predict_fn(shard):
            y = local_model.predict(shard["x"], **params)
            return {"prediction": y}

        new_part = [predict_fn(shard) for shard in partition]

        return new_part

