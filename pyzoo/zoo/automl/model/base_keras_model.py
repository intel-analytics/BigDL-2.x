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
from zoo.automl.model.abstract import BaseModel
from zoo.automl.common.util import *
from zoo.automl.common.metrics import Evaluator
import pickle


class KerasBaseModel(BaseModel):
    def __init__(self,
                 model_creator,
                 check_optional_config=False):
        self.check_optional_config = check_optional_config
        self.model_creator = model_creator
        self.model = None
        self.config = None

    def fit_eval(self, x, y, validation_data=None, mc=False, verbose=0, epochs=1, metric="mse",
                 **config):
        def update_config():
            config.setdefault("input_dim", x.shape[-1])
            config.setdefault("output_dim", y.shape[-1])
            config.update({"metric": metric})
        update_config()
        self._check_config(config)
        self.model = self.model_creator(config)
        self.config = config

        hist = self.model.fit(x, y,
                              validation_data=validation_data,
                              batch_size=self.config.get("batch_size", 32),
                              epochs=epochs,
                              verbose=verbose
                              )

        if validation_data is None:
            result = hist.history.get(metric)[-1]
        else:
            result = hist.history.get('val_' + str(metric))[-1]
        return result

    def evaluate(self, x, y, metrics=['mse']):
        """
        Evaluate on x, y
        :param x: input
        :param y: target
        :param metrics: a list of metrics in string format
        :return: a list of metric evaluation results
        """
        y_pred = self.predict(x)
        return [Evaluator.evaluate(m, y, y_pred) for m in metrics]

    def predict(self, x):
        """
        Prediction on x.
        :param x: input
        :return: predicted y
        """
        if not self.model:
            raise RuntimeError("You must call fit_eval or restore first before calling predict!")
        return self.model.predict(x, batch_size=self.config.get(["batch_size"], 32))

    def predict_with_uncertainty(self, x, n_iter=100):
        if not self.model:
            raise RuntimeError("You must call fit_eval or restore first before calling predict!")
        result = np.stack([self.model(x, training=True) for _ in range(n_iter)])
        prediction = result.mean(axis=0)
        uncertainty = result.std(axis=0)
        return prediction, uncertainty

    def state_dict(self):
        state = {
            "config": self.config,
            "weights": self.model.get_weights(),
            "optimizer_weights": self.model.optimizer.get_weights()
        }
        return state

    def load_state_dict(self, state):
        self.model.set_weights(state["weights"])
        self.model.optimizer.set_weights(state["optimizer_weights"])
        self.config = state["config"]

    def save(self, model_path, config_path, **config):
        if not self.model:
            raise RuntimeError("You must call fit_eval or restore first before calling save!")
        state_dict = self.state_dict()
        with open(model_path, "wb") as f:
            pickle.dump(state_dict, f)

    def restore(self, model_path, **config):
        with open(model_path, "rb") as f:
            state_dict = pickle.load(f)
        self.load_state_dict(state_dict)

    def _get_required_parameters(self):
        return {"metric"}

    def _get_optional_parameters(self):
        return {"batch_size"}
