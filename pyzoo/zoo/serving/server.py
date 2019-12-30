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
import shutil
import os
import time


class ClusterServing:

    def __init__(self):
        self.name = 'cluster-serving'
        self.proc = None
        self.conf_path = os.path.abspath(__file__ + "/../../bin/cluster-serving/config.yaml")
        if not os.path.exists(self.conf_path):
            print('WARNING: Config file does not exist in your pip directory,'
                  'are you sure that you install serving by pip?')
            self.conf_path = os.path.abspath(__file__ + "/../../../../scripts/cluster-serving/config.yaml")

            subprocess.Popen(['cat', __file__])
            if not os.path.exists(self.conf_path):
                raise EOFError("Can not find your config file.")
        shutil.copyfile(self.conf_path, 'config.yaml')

        self.serving_sh_path = os.path.join(__file__ + "/../../"
                                            'bin/cluster-serving/start-cluster-serving.sh')

        subprocess.Popen(['chmod', 'a+x', self.conf_path])
        subprocess.Popen(['chmod', 'a+x', self.serving_sh_path])

    def docker_run(self):
        with open(self.conf_path, 'r') as f:
            config = yaml.load(f)
            if not config['model']['path']:
                raise EOFError('You have not set model path, you have to set'
                               'model path before you start docker container.')
            model_path = config['model']['path']
            cmd = 'docker run -it --name cluster-serving --net=host -v ' + \
                model_path + ':/opt/work/model/ -v config.yaml:/opt/work/config.yaml' + \
                + ' intelanalytics/zoo-cluster-serving:beta'
            subprocess.Popen(cmd, shell=True)

        subprocess.Popen('docker run')

    def docker_remove(self):
        subprocess.Popen('docker rm -f ' + self.name)

    def start(self):
        """
        Start the serving by running start script
        :return:
        """
        self.proc = subprocess.Popen(
            [self.serving_sh_path], shell=True)

    def stop(self):
        """
        Stop the serving by sending stop signal
        aka. removing running flag
        :return:
        """
        os.remove("running")

    def restart(self):
        self.stop()
        while not self.proc.poll():
            # if return null, the subprocess is still running, wait
            time.sleep(3)
        self.start()

    def set_config(self, field, param, value):
        """
        Setting config by loading and rewrite the config file
        :param field: model/data/params
        :param param: should correspond with field
        :param value: the value of the field and param specified
        :return:
        """
        if not os.path.exists('config.yaml'):
            if not os.path.exists(self.conf_path):
                raise EOFError("Source config does not exist...what have you done...")
            shutil.copyfile(self.conf_path, '.')
        with open(self.conf_path, 'r') as f:
            config = yaml.load(f)
            try:
                config[field][param] = value
            except SyntaxError:
                print("You have provided invalid configuration, "
                      "please check Configuration Guide.")
                return
        with open(self.conf_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
