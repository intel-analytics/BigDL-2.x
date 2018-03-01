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
import numpy as np

class Trajectory(object):

    fields = ["observations", "actions", "rewards", "terminal"]

    def __init__(self):
        self.data = {k: [] for k in self.fields}
        self.last_r = 0.0


    def add(self, **kwargs):
        '''
        add a single step to this trajectory,
        e.g. traj.add(observations=obs, actions=action, rewards=reward, terminal=terminal)
        '''
        for k, v in kwargs.items():
            self.data[k] += [v]

    def is_terminal(self):
        return self.data["terminal"][-1]

class Sampler(object):
    '''
    Helper class to sample one trajectory from the environment
    using the given model (policy or value_func/q_func estimator)
    '''

    def get_data(self, model, max_steps):
        '''
        Sample one trajectory from the environment, using the given model
        to `max_steps` steps
        '''
        raise NotImplementedError


class PolicySampler(Sampler):
    '''
    Helper class to sample one trajectory from the environment
    using the given policy
    '''

    def __init__(self, env, horizon=None):
        self.horizon = horizon
        self.env = env
        self.last_obs = env.reset()

    def get_data(self, policy, max_steps):
        return self._run_policy(
            self.env, policy, max_steps, self.horizon)

    def _run_policy(self, env, policy, max_steps, horizon):
        length = 0

        traj = Trajectory()

        for _ in range(max_steps):
            action_distribution = policy.forward(self.last_obs)
            action = np.random.multinomial(1, action_distribution).argmax()
            observation, reward, terminal, info = env.step(action)

            length += 1
            if length >= horizon:
                terminal = True

            # Collect the experience.
            traj.add(observations=self.last_obs,
                     actions=action,
                     rewards=reward,
                     terminal=terminal)

            self.last_obs = observation

            if terminal:
                self.last_obs = env.reset()
                break
        return traj
