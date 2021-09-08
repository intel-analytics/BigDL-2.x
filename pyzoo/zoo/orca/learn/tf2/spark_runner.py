import json
from subprocess import call

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

class SparkRunner:
    def __init__(self, model_creator, data_creator, validation_data_creator, config=None):
        self.model_creator = model_creator
        self.data_creator = data_creator
        self.validation_data_creator = validation_data_creator
        self.config = {} if config is None else config

    def handle_datasets_train(self, data_creator, validation_data_creator):   
        train_dataset = data_creator()
        if validation_data_creator is not None:
            test_dataset = validation_data_creator()
        else:
            test_dataset = None
        return train_dataset, test_dataset

    def disributed_train_func(self, *args):
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

        strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

        with strategy.scope():
            model = self.model_creator()

        data_creator = self.data_creator
        validation_data_creator = self.validation_data_creator
        train_dataset, test_dataset = self.handle_datasets_train(data_creator, validation_data_creator)
        
        model.fit(train_dataset, test_dataset, epochs=10, verbose=1)

        return [model.get_weights()]
    
    def main(self):
        import numpy as np

        model_creator = self.model_creator
        data_creator = self.data_creator
        validation_data_creator = self.validation_data_creator

        sc = SparkContext()
        sparkRDD = sc.parallelize(list(range(40)), 4)
        workerRDD = sparkRDD.repartition(2)
        spark_func = SparkRunner(model_creator, data_creator, validation_data_creator).disributed_train_func
        res = workerRDD.barrier().mapPartitions(spark_func).collect()
        assert np.all(res[0][0] == res[1][0])
        sc.stop()

        return res

