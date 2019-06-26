from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time

import numpy as np
import ray

from zoo.ray.allreduce.model import RayModel
from zoo.ray.allreduce.ps import ShardedParameterServer
from zoo.ray.util import utils

logger = logging.getLogger(__name__)


@ray.remote(resources={"trainer":1})
class ModelWorker(object):
    def __init__(self, rayModel, gen_ray_data_set, num_workers):
        self.num_workers = num_workers
        self.ray_model = rayModel.resolve()
        self.ray_data_set = gen_ray_data_set

    # @ray.remote(num_return_vals=2)
    def set_parameters_compute_gradients(self, *parameters):
        """
        It would return a sharded grads here.
        Each parameter should be a 1-D vector
        """
        # concate grads
        flat_parameters = np.concatenate(parameters)
        self.ray_model.set_flat_parameters(flat_parameters)
        if self.ray_data_set.has_next():
            input_data, label_data = self.ray_data_set.next_batch()
        else:
            raise Exception("End of data sequence")
        # grads, acc, loss = self.ray_model.compute_gradients(input, label)
        grads, loss = self.ray_model.compute_gradients(self._generate_feed_dict(self.ray_model.inputs,
                                                                         utils.to_list(input_data),
                                                                         self.ray_model.targets,
                                                                         utils.to_list(label_data)))
        print(loss)

        flat_grads = np.concatenate([g.flatten() for g in grads])
        print("flat_grads {}".format(flat_grads.shape))
        sharded_grads = utils.split(flat_grads, self.num_workers)
        return sharded_grads

    def get_weights(self):
        return self.ray_model.get_flat_parameters()

    # TODO: Get rid of this and use the raw dataset only.
    def _generate_feed_dict(self, inputs_op, inputs, targets_op, targets):
        fdict = {}
        if inputs:
            # we don't need to feed data with dataset API
            fdict.update(dict(zip(inputs_op, inputs)))
        if targets:
            fdict.update(dict(zip(targets_op, targets)))
        return fdict


class DistributedEstimator(object):

    def __init__(self,
                 model_fn,
                 ray_dataset_train,
                 num_worker,
                 keras_model=None):
        self.rayModel = RayModel.from_model_fn(model_fn) if model_fn \
            else RayModel.from_model_fn(keras_model)

        self.num_worker = num_worker
        self.batch_size = ray_dataset_train.get_batchsize()
        self.ray_model_resolved = self.rayModel.resolve()

        weights = self.ray_model_resolved.get_flat_parameters()
        sharded_weights = utils.split(weights, self.num_worker)
        # This weights would be used for both PS and ModelWorker
        sharded_weight_ids = [ray.put(w) for w in sharded_weights]
        self.workers = []
        self.pss = []
        logger.info(
            "Creating parameter server ({} total)".format(
                num_worker))

        for ps_index in range(num_worker):
            self.pss.append(
                ShardedParameterServer.remote(sharded_weight_ids[ps_index],
                                              ray_model=self.rayModel))

        logger.info(
            "Creating model workers ({} total)".format(
                num_worker))
        for worker_index in range(num_worker):
            self.workers.append(
                ModelWorker.remote(self.rayModel, ray_dataset_train, num_worker))

    def train(self, steps=10):
        for i in range(1, steps + 1):
            self.step(i)

        self.ray_model_resolved.set_flat_parameters(ray.get(self.workers[0].get_weights.remote()))
        return self

    def evaluate(self, ray_dataset):
        pass

    def step(self, step_id):
        start = time.time()
        # workers of sharded_grads
        sharded_grad_ids = []
        results = []
        for worker in self.workers:
            # 1) pull the latest weights from ps
            parameters = [ps.get_parameters.remote() for ps in self.pss]
            # 2) compute the grads
            sharded_grad = worker.set_parameters_compute_gradients._remote(args=parameters, kwargs=None, num_return_vals=self.num_worker)
            sharded_grad_ids.append(sharded_grad)

        # TODO: we don't need to print this for every iteration
        # print("Iteration: {}, loss is {}".format(step_id, np.mean([ray.get(loss) for loss in losses])))

        grads_per_ps = list(zip(*sharded_grad_ids))
        assert len(grads_per_ps[0]) == self.num_worker, "we should get correct grads for each ps"
        # 3) push and aggregate grads on ps
        for index, grads in enumerate(grads_per_ps):
            results.append(self.pss[index].apply_gradients.remote(*grads))
        # wait for complete
        ray.wait(object_ids=results, num_returns=len(results))
        end = time.time()
        print("Iteration: {}, throughput: {}".format(step_id, self.batch_size * self.num_worker / (
        end - start)))



