import json
from subprocess import call
from sys import version

from pyspark import BarrierTaskContext
from pyspark.context import SparkContext
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
    def __init__(self, model_creator, data_creator, validation_data_creator, config=None, 
                epochs=1, batch_size=32, verbose=None, callbacks=None, class_weight=None):
        self.model_creator = model_creator
        self.data_creator = data_creator
        self.validation_data_creator = validation_data_creator
        self.config = {} if config is None else config

        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.callbacks = callbacks
        self.class_weight = class_weight

    def disributed_train_func(self, *args):
        """
        Sets up TensorFLow distributed environment and initializes the model.
        Runs a training epoch and updates the model parameters.
        """
        tc = BarrierTaskContext().get()
        rank = tc.partitionId()
        free_port = find_free_port(tc)
        cluster = tc.allGather(str(free_port))
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

        self.strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

        with self.strategy.scope():
            self.model = self.model_creator()
        
        data_creator = self.data_creator
        validation_data_creator = self.validation_data_creator
        epochs = self.epochs
        verbose = self.verbose
        callbacks = self.callbacks

        train_dataset, test_dataset = handle_datasets_train(data_creator, 
                                                        validation_data_creator)
        
        history = self.model.fit(train_dataset, test_dataset, epochs, verbose, callbacks)

        if history is None:
            stats = {}
        else:
            stats = {"train_" + k: v[-1] for k, v in history.history.items()}
        
        return [stats]
        #return [model.get_weights()]
    



