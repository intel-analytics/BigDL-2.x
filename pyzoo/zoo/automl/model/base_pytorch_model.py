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
import torch
from zoo.automl.model.abstract import BaseModel
from zoo.automl.common.util import *
from zoo.automl.common.metrics import Evaluator


class PytorchBaseModel(BaseModel):
    def __init__(self, model_creator, optimizer_creator, loss_creator, config,
                 check_optional_config=False):
        self.check_optional_config = check_optional_config
        self._check_config(**config)
        self.model = model_creator(config)
        self.optimizer = optimizer_creator(self.model, config)
        self.criterion = loss_creator(config)
        self.config = config

    def fit_eval(self, x, y, validation_data=None, mc=False, verbose=0, epochs=1, metric=["mse"],
                 **config):
        epoch_losses = []
        for i in range(epochs):
            train_loss = self._train_epoch(x, y)
            epoch_losses.append(train_loss)
        train_stats = {"loss": np.mean(epoch_losses), "last_loss": epoch_losses[-1]}
        # todo: support input validation data None
        assert validation_data is not None, "You must input validation data!"
        val_stats = self._validate(validation_data[0], validation_data[1], metric=metric)
        return train_stats.update(val_stats)

    def _train_epoch(self, x, y):
        # todo: support torch data loader
        batch_size = self.config["batch_size"]
        self.model.train()
        batch_idx = 0
        total_loss = 0
        for i in range(0, x.size(0), batch_size):
            if i + batch_size > x.size(0):
                x, y = x[i:], y[i:]
            else:
                x, y = x[i:(i + batch_size)], y[i:(i + batch_size)]
            self.optimizer.zero_grad()
            yhat = self._forward(x, y)
            loss = self.criterion(yhat, y)
            loss.backward()
            self.optimizer.step()
            batch_idx += 1
            total_loss += loss.item()
        train_loss = total_loss / batch_idx
        return train_loss

    def _forward(self, x, y):
        return self.model(x)

    def _validate(self, x, y, metric):
        self.model.eval()
        with torch.no_grad():
            yhat = self.model(x)
            val_loss = self.criterion(yhat, y)
            eval_result = Evaluator.evaluate(metric=metric,
                                             y_true=y, y_pred=yhat.numpy(),
                                             multioutput='uniform_average')
        return {"val_loss": val_loss.item(),
                metric: eval_result}

    def _print_model(self):
        # print model and parameters
        print(self.model)
        print(len(list(self.model.parameters())))
        for i in range(len(list(self.model.parameters()))):
            print(list(self.model.parameters())[i].size())

    def evaluate(self, x, y, metric=['mse']):
        yhat = self.predict(x)
        eval_result = [Evaluator.evaluate(m, y_true=y, y_pred=yhat, multioutput="raw_values")
                       for m in metric]
        return eval_result

    def predict(self, x, mc=False):
        if mc:
            self.model.train()
        else:
            self.model.eval()
        yhat = self.model(x)
        return yhat

    def predict_with_uncertainty(self, x, n_iter=100):
        result = np.zeros((n_iter,) + (x.shape[0], self.config["output_size"]))
        for i in range(n_iter):
            result[i, :, :] = self.predict(x, mc=True)

        prediction = result.mean(axis=0)
        uncertainty = result.std(axis=0)
        return prediction, uncertainty

    def state_dict(self):
        state = {
            "config": self.config,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        return state

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.config = state["config"]
        self.optimizer.load_state_dict(state["optimizer"])

    def save(self, model_path, config_path, **config):
        state_dict = self.state_dict()
        torch.save(state_dict, model_path)

    def restore(self, model_path, **config):
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict)

    def _get_required_parameters(self):
        return {}

    def _get_optional_parameters(self):
        return {"batch_size",
                'lr',
                "dropout",
                "optim"
                }
