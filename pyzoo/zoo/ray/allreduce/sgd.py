from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import ray

from zoo.ray.util import utils

logger = logging.getLogger(__name__)


@ray.remote(resources={"trainer":1})
class ModelWorker(object):
    def __init__(self, rayModel, ray_data_set, num_workers):
        self.num_workers = num_workers
        self.ray_model = rayModel.resolve()
        self.ray_data_set = ray_data_set.action()

    # @ray.remote(num_return_vals=2)
    def set_parameters_compute_gradients(self, *parameters):
        """
        It would return a sharded grads here.
        Each parameter should be a 1-D vector
        """
        # concate grads
        flat_parameters = np.concatenate(parameters)
        self.ray_model.set_flat_parameters(flat_parameters)

        input_data, label_data = self.ray_data_set.next_batch()

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

    # TODO: Get rid of this and use the raw dataset instead.
    def _generate_feed_dict(self, inputs_op, inputs, targets_op, targets):
        fdict = {}
        if inputs:
            # we don't need to feed data with dataset API
            fdict.update(dict(zip(inputs_op, inputs)))
        if targets:
            fdict.update(dict(zip(targets_op, targets)))
        return fdict

