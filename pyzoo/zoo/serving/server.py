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
import subprocess
import yaml


class ClusterServing:

    def __init__(self):
        self.proc = None
        self.conf_path = "../../../scripts/cluster-serving/config.yaml"

        subprocess.Popen(['chmod', 'a+x', '../../../scripts/cluster-serving/config.yaml'])
        subprocess.Popen(['chmod', 'a+x', '../../../scripts/cluster-serving/start-cluster-serving.sh'])

    def start(self):
        """
        Start the serving by running start script
        :return:
        """
        self.proc = subprocess.Popen(
            ['../../../scripts/cluster-serving/start-cluster-serving.sh'], shell=True)

    def stop(self):
        """
        Stop the serving by sending stop signal
        aka. removing running flag
        :return:
        """
        self.proc.kill()

    def restart(self):
        self.stop()
        self.start()

    def set_config(self, field, param, value):
        with open(self.conf_path, 'r') as f:
            config = yaml.load(f)
            try:
                config[field][param] = value
            except Exception:
                print("You have provided invalid configuration, "
                      "please check Configuration Guide.")
                return
        with open(self.conf_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
