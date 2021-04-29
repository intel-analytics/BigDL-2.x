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
from unittest import TestCase
from zoo.automl.model import PytorchModelBuilder
from torch.utils.data import Dataset, DataLoader

import pytest

import numpy as np
import torch
import torch.nn as nn


def get_data(train_size=1000, valid_size=400):
    def get_linear_data(a, b, size):
        x = np.arange(0, 10, 10 / size, dtype=np.float32)
        y = a*x + b
        return x, y
    train_x, train_y = get_linear_data(2, 5, train_size)
    val_x, val_y = get_linear_data(2, 5, valid_size)
    data = {'x': train_x, 'y': train_y, 'val_x': val_x, 'val_y': val_y}
    return data

class CustomDataset(Dataset):
    def __init__(self, mode="train", train_size=1000, valid_size=400):
        self.data = get_data(train_size=train_size, valid_size=valid_size)
        self.mode = mode
        self.train_size = train_size
        self.valid_size = valid_size

    def __len__(self):
        if self.mode == "train":
            return self.train_size
        if self.mode == "valid":
            return self.valid_size
        return None

    def __getitem__(self, idx):
        if self.mode == "train":
            return torch.from_numpy(self.data['x'][idx].reshape(-1, 1)),\
                   torch.from_numpy(self.data['y'][idx].reshape(-1, 1))
        if self.mode == "valid":
            return torch.from_numpy(self.data['val_x'][idx].reshape(-1, 1)),\
                   torch.from_numpy(self.data['val_y'][idx].reshape(-1, 1))
        return None, None


def train_dataloader_creator(config):
    return DataLoader(CustomDataset(mode="train",
                                    train_size=config["train_size"]),
                      batch_size=config["batch_size"],
                      shuffle=True)


def valid_dataloader_creator(config):
    return DataLoader(CustomDataset(mode="valid",
                                    valid_size=config["valid_size"]),
                      batch_size=config["batch_size"],
                      shuffle=True)


def model_creator_pytorch(config):
    """Returns a torch.nn.Module object."""
    return nn.Linear(1, config.get("hidden_size", 1))


def optimizer_creator(model, config):
    """Returns optimizer defined upon the model parameters."""
    return torch.optim.SGD(model.parameters(), lr=config.get("lr", 1e-2))


def loss_creator(config):
    return nn.MSELoss()


class TestBasePytorchModel(TestCase):
    data = get_data()

    def test_dataloader_fit_evaluate(self):
        modelBuilder = PytorchModelBuilder(model_creator=model_creator_pytorch,
                                           optimizer_creator=optimizer_creator,
                                           loss_creator=loss_creator)
        model = modelBuilder.build(config={
            "lr": 1e-2,
            "batch_size": 32,
        })
        extra_config = {"train_size": 500, "valid_size": 100}
        val_result = model.fit_eval(x=None,
                                    y=None,
                                    training_dataloader_creator=train_dataloader_creator,
                                    validation_dataloader_creator=valid_dataloader_creator,
                                    epochs=20,
                                    **extra_config)
        assert model.config["train_size"] == 500
        assert model.config["valid_size"] == 100
        assert val_result is not None

    def test_fit_evaluate(self):
        modelBuilder = PytorchModelBuilder(model_creator=model_creator_pytorch,
                                           optimizer_creator=optimizer_creator,
                                           loss_creator=loss_creator)
        model = modelBuilder.build(config={
            "lr": 1e-2,
            "batch_size": 32,
        })
        val_result = model.fit_eval(x=self.data["x"],
                                    y=self.data["y"],
                                    validation_data=(self.data["val_x"], self.data["val_y"]),
                                    epochs=20)
        assert val_result is not None

    def test_evaluate(self):
        modelBuilder = PytorchModelBuilder(model_creator=model_creator_pytorch,
                                           optimizer_creator=optimizer_creator,
                                           loss_creator=loss_creator)
        model = modelBuilder.build(config={
            "lr": 1e-2,
            "batch_size": 32,
        })
        model.fit_eval(x=self.data["x"],
                       y=self.data["y"],
                       validation_data=(self.data["val_x"], self.data["val_y"]),
                       epochs=20)
        mse_eval = model.evaluate(x=self.data["val_x"], y=self.data["val_y"])
        try:
            import onnx
            import onnxruntime
            mse_eval_onnx = model.evaluate_with_onnx(x=self.data["val_x"], y=self.data["val_y"])
            np.testing.assert_almost_equal(mse_eval, mse_eval_onnx)
        except ImportError:
            pass
        # incremental training test
        model.fit_eval(x=self.data["x"],
                       y=self.data["y"],
                       validation_data=(self.data["val_x"], self.data["val_y"]),
                       epochs=20)
        mse_eval = model.evaluate(x=self.data["val_x"], y=self.data["val_y"])
        try:
            import onnx
            import onnxruntime
            mse_eval_onnx = model.evaluate_with_onnx(x=self.data["val_x"], y=self.data["val_y"])
            np.testing.assert_almost_equal(mse_eval, mse_eval_onnx)
        except ImportError:
            pass

    def test_predict(self):
        modelBuilder = PytorchModelBuilder(model_creator=model_creator_pytorch,
                                           optimizer_creator=optimizer_creator,
                                           loss_creator=loss_creator)
        model = modelBuilder.build(config={
            "lr": 1e-2,
            "batch_size": 32,
        })
        model.fit_eval(x=self.data["x"],
                       y=self.data["y"],
                       validation_data=(self.data["val_x"], self.data["val_y"]),
                       epochs=20)
        pred = model.predict(x=self.data["val_x"])
        pred_full_batch = model.predict(x=self.data["val_x"], batch_size=len(self.data["val_x"]))
        np.testing.assert_almost_equal(pred, pred_full_batch)
        try:
            import onnx
            import onnxruntime
            pred_onnx = model.predict_with_onnx(x=self.data["val_x"])
            np.testing.assert_almost_equal(pred, pred_onnx)
        except ImportError:
            pass

    def test_create_not_torch_model(self):
        def model_creator(config):
            return torch.Tensor(3, 5)

        modelBuilder = PytorchModelBuilder(model_creator=model_creator,
                                           optimizer_creator=optimizer_creator,
                                           loss_creator=loss_creator)
        with pytest.raises(ValueError):
            model = modelBuilder.build(config={
                "lr": 1e-2,
                "batch_size": 32,
            })


if __name__ == "__main__":
    pytest.main([__file__])
