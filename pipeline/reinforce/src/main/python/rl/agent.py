#
# Copyright 2016 The BigDL Authors.
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

from trajectory import PolicySampler
from utils import *
import numpy as np
import scipy.signal


class Agent(object):

    def sample(self, model, num_steps=None, num_trajs=None):
        '''
        Samples trajectories from the environment using the provided model
        the number of steps or the number of trajectories sampled is specified
        by num_steps or num_trajs.
        One of num_steps and num_trajs should be provided.
        :param model: model representing current policy/value_function/q_function
        :param num_steps: number of steps sampled
        :param num_trajs: number of trajectories sampled
        :return: Sampled Trajectories
        '''

        raise NotImplementedError

class REINFORCEAgent(Agent):

    def __init__(self, env, horizon = 1000):
        self.env = env
        self.horizon = horizon
        self.sampler = PolicySampler(env, horizon)

    def sample(self, model, num_steps=None, num_trajs=None):
        if num_steps is None and num_trajs is None:
            raise ValueError("one of num_steps and num_trajs must be provided")
        if num_steps is not None:
            return self._sample_to_max_steps(model, num_steps)
        else:
            return self._sample_num_trajs(model, num_trajs)

    def _sample_to_max_steps(self, model, num_steps):
        num_steps_so_far = 0
        samples = []
        while num_steps_so_far < num_steps:
            traj, traj_size = _process_traj(self.sampler.get_data(model, num_steps - num_steps_so_far))
            samples.append(traj)
            num_steps_so_far += traj_size
        return samples

    def _sample_num_trajs(self, model, num_trajs):
        i = 0
        samples = []
        while i < num_trajs:
            traj, traj_size = _process_traj(self.sampler.get_data(model, self.horizon))
            samples.append(traj)
            i += 1
        return samples

class DistributedAgents(object):
    def __init__(self, sc, create_agent, parallelism):
        self.create_agent_func = create_agent
        self.parallelism = parallelism
        self.sc = sc

    def __enter__(self):
        func = self.create_agent_func
        self.agents = self.sc.parallelize(range(self.parallelism), self.parallelism).map(func).cache()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.agents.unpersist()


def _process_traj(traj):
    traj_size = len(traj.data["actions"])
    for key in traj.data:
        traj.data[key] = np.stack(traj.data[key])
    return traj, traj_size
