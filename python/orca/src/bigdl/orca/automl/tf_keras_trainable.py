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


class BaseTrainable:


    @classmethod
    def default_resource_request(cls, config):
        return Resources(
            cpu=0,
            gpu=0,
            extra_cpu=config["num_replicas"],
            extra_gpu=int(config["use_gpu"]) * config["num_replicas"])

    def setup(self, config):
        self._trainer = TFTrainer(
            model_creator=config["model_creator"],
            data_creator=config["data_creator"],
            config=config.get("trainer_config", {}),
            num_replicas=config["num_replicas"],
            use_gpu=config["use_gpu"],
            num_cpus_per_worker=config.get("num_cpus_per_worker", 1))

    def step(self):

        train_stats = self._trainer.train()
        validation_stats = self._trainer.validate()

        train_stats.update(validation_stats)

        return train_stats

    def save_checkpoint(self, checkpoint_dir):
        return zoo.chronos.feature.utils.save(os.path.join(checkpoint_dir, "model"))

    def load_checkpoint(self, checkpoint_path):
        return zoo.chronos.feature.utils.restore(checkpoint_path)

    def cleanup(self):
        self._trainer.shutdown()