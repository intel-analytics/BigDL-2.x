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

import yaml
import redis


class Config:
    def __init__(self, file_path=None):
        if file_path:
            with open(file_path) as f:
                config = yaml.load(f)
        else:
            config = {'data': {'host': None, 'port': None}}
        if not config['data']['host']:
            config['data']['host'] = "localhost"
        if not config['data']['port']:
            config['data']['port'] = "6379"
        self.db = redis.StrictRedis(host=config['data']['host'],
                                    port=config['data']['port'], db=0)
