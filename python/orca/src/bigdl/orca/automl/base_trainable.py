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
import zoo.chronos.feature.utils


class BaseModel:

    def setup(self, config):
        pass

    def step(self):
        pass

    def save_checkpoint(self, checkpoint_dir):
        return zoo.chronos.feature.utils.save(os.path.join(checkpoint_dir, "model"))

    def load_checkpoint(self, checkpoint_path):
        return zoo.chronos.feature.utils.restore(checkpoint_path)

    def cleanup(self):
        self._trainer.shutdown()