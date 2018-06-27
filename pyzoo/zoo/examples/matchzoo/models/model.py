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
from __future__ import print_function
from __future__ import absolute_import
import sys

class BasicModel(object):
    def __init__(self, config):
        self._name = 'BasicModel'
        self.config = {}
        self.check_list = []
        #self.setup(config)
        #self.check()

    def set_default(self, k, v):
        if k not in self.config:
            self.config[k] = v

    def setup(self, config):
        pass

    def check(self):
        for e in self.check_list:
            if e not in self.config:
                print(e, end='\n')
                print('[Model] Error %s not in config' % e, end='\n')
                return False
        return True

    def build(self):
        pass

    def check_list(self,check_list):
        self.check_list = check_list
