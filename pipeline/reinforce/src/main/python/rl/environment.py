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

class Environment(object):

    def reset(self):
        '''
        reset this environment
        :return: 
        '''
        raise NotImplementedError

    def step(self, action):
        '''
        Run one timestep of the environment's dynamics
        '''
        raise NotImplementedError

class GymEnvWrapper(object):

    def __init__(self, env):
        self.gym = env

    def reset(self):
        return self.gym.reset()

    def step(self, action):
        return self.gym.step(action)