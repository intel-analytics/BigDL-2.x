import json
import logging
import os

from re import VERBOSE
from subprocess import call
from sys import version

from pyspark import BarrierTaskContext
from pyspark.context import SparkContext
from example import model_creator
import tensorflow as tf
from numpy import array
from contextlib import closing
import socket

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

class SparkRunner:
    def __init__(self, model_creator, data_creator, validation_data_creator,
                config=None, epochs=1, batch_size=32, verbose=None, callbacks=None, 
                class_weight=None, data_config=None):
        self.model_creator = model_creator
        self.data_creator = data_creator
        self.validation_data_creator = validation_data_creator
        self.config = {} if config is None else config

        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.callbacks = callbacks
        self.class_weight = class_weight
        self.data_config = data_config

    def distributed_train_func(self, *args):
        """
        Sets up TensorFLow distributed environment, initializes the model,
        runs a training epoch and updates the model parameters
        """
        tc = BarrierTaskContext().get()
        rank = tc.partitionId()
        free_port = find_free_port(tc)
        cluster = tc.allGather(str(free_port))
        self.cluster = cluster
        print(cluster)

        import os
        os.environ["TF_CONFIG"] = json.dumps({
        'cluster': {
            'worker': cluster
        },
        'task': {'type': 'worker', 'index': rank}
        })
        ips = set([node.split(":")[0] for node in cluster])
        os.environ["no_proxy"] = ",".join(ips)

        strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        with strategy.scope():
            model = self.model_creator()
        
        data_creator = self.data_creator
        validation_data_creator = self.validation_data_creator
        epochs = self.epochs
        verbose = self.verbose
        callbacks = self.callbacks

        train_dataset, test_dataset = handle_datasets_train(data_creator, 
                                                            validation_data_creator)
        
        history = model.fit(train_dataset, test_dataset, epochs, verbose, callbacks)

        if history is None:
            stats = {}
        else:
            stats = {"train_" + k: v[-1] for k, v in history.history.items()}
        return model, [stats]
        
    def step(self, *args):
        """
        Get model training results and new model.
        """
        model, stats = self.distributed_train_func()
        weights = model.get_weights()

        return [weights], [stats]
    
    def validate(self, *args):
        """
        Evaluates the model on the validation data set.
        """
        params = dict(
            verbose=self.verbose,
            callbacks=self.callbacks
        )

        model, stats = self.distributed_train_func()
        weights = model.get_weights()

        data_creator = self.data_creator
        validation_data_creator = self.validation_data_creator
        train_dataset, test_dataset = handle_datasets_train(data_creator, validation_data_creator)

        results = model.evaluate(train_dataset, test_dataset, **params)

        if results is None:
            model_weights = weights[0]
            local_model = self.model_creator()
            local_model = local_model.set_weights(model_weights)
            results = local_model.evaluate(train_dataset, test_dataset, **params)
        
        if isinstance(results, list):
            stats = {
                "validation_" + k: v
                for k, v in zip(model.metrics_names, results)
            }
        else:
            stats = {"results": results}
        
        stats = {"results": results}

        return [stats]
    
    #def save_model(self)
    
    



