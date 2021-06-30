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
from torch.utils.data import TensorDataset, DataLoader

import types

from zoo.automl.model.abstract import BaseModel, ModelBuilder
from zoo.automl.common.util import *
from zoo.automl.common.metrics import Evaluator
import pandas as pd

from zoo.orca.automl.pytorch_utils import LR_NAME, DEFAULT_LR

PYTORCH_REGRESSION_LOSS_MAP = {"mse": "MSELoss",
                               "mae": "L1Loss",
                               "huber_loss": "SmoothL1Loss"}


class PytorchBaseModel(BaseModel):
    def __init__(self, model_creator, optimizer_creator, loss_creator,
                 check_optional_config=False):
        self.check_optional_config = check_optional_config
        self.model_creator = model_creator
        self.optimizer_creator = optimizer_creator
        self.loss_creator = loss_creator
        self.config = None
        self.model = None
        self.model_built = False
        self.onnx_model = None
        self.onnx_model_built = False

    def _create_loss(self):
        if isinstance(self.loss_creator, torch.nn.modules.loss._Loss):
            self.criterion = self.loss_creator
        else:
            self.criterion = self.loss_creator(self.config)

    def _create_optimizer(self):
        import types
        if isinstance(self.optimizer_creator, types.FunctionType):
            self.optimizer = self.optimizer_creator(self.model, self.config)
        else:
            # use torch default parameter values if user pass optimizer name or optimizer class.
            try:
                self.optimizer = self.optimizer_creator(self.model.parameters(),
                                                        lr=self.config.get(LR_NAME, DEFAULT_LR))
            except:
                raise ValueError("We failed to generate an optimizer with specified optim "
                                 "class/name. You need to pass an optimizer creator function.")

    def build(self, config):
        # check config and update
        self._check_config(**config)
        self.config = config
        # build model
        if "selected_features" in config:
            config["input_feature_num"] = len(config['selected_features'])\
                + config['output_feature_num']
        self.model = self.model_creator(config)
        if not isinstance(self.model, torch.nn.Module):
            raise ValueError("You must create a torch model in model_creator")
        self.model_built = True
        self._create_loss()
        self._create_optimizer()

    def _reshape_input(self, x):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return x

    def _np_to_creator(self, data):
        def data_creator(config):
                x, y = PytorchBaseModel.covert_input(data)
                x = self._reshape_input(x)
                y = self._reshape_input(y)
                return DataLoader(TensorDataset(x, y),
                                  batch_size=int(config["batch_size"]),
                                  shuffle=True)
        return data_creator

    def fit_eval(self, data, validation_data=None, mc=False, verbose=0, epochs=1, metric=None,
                 metric_func=None,
                 **config):
        """
        :param data: data could be a tuple with numpy ndarray with form (x, y) or a
               data creator which takes a config dict and returns a
               torch.utils.data.DataLoader. torch.Tensor should be generated from the
               dataloader.
        :param validation_data: validation data could be a tuple with numpy ndarray
               with form (x, y) or a data creator which takes a config dict and returns a
               torch.utils.data.DataLoader. torch.Tensor should be generated from the
               dataloader.
        fit_eval will build a model at the first time it is built
        config will be updated for the second or later times with only non-model-arch
        params be functional
        TODO: check the updated params and decide if the model is needed to be rebuilt
        """
        # todo: support input validation data None
        assert validation_data is not None, "You must input validation data!"

        if not metric:
            raise ValueError("You must input a valid metric value for fit_eval.")

        # update config settings
        def update_config():
            if not isinstance(data, types.FunctionType):
                x = self._reshape_input(data[0])
                y = self._reshape_input(data[1])
                config.setdefault("past_seq_len", x.shape[-2])
                config.setdefault("future_seq_len", y.shape[-2])
                config.setdefault("input_feature_num", x.shape[-1])
                config.setdefault("output_feature_num", y.shape[-1])

        if not self.model_built:
            update_config()
            self.build(config)
        else:
            tmp_config = self.config.copy()
            tmp_config.update(config)
            self._check_config(**tmp_config)
            self.config.update(config)

        # get train_loader and validation_loader
        if isinstance(data, types.FunctionType):
            train_loader = data(self.config)
            validation_loader = validation_data(self.config)
        else:
            assert isinstance(data, tuple) and isinstance(validation_data, tuple),\
                f"data/validation_data should be a tuple or\
                 data creator function but found {type(data)}"
            assert isinstance(data[0], np.ndarray) and isinstance(validation_data[0], np.ndarray),\
                f"x should be a np.ndarray but found {type(x)}"
            assert isinstance(data[1], np.ndarray) and isinstance(validation_data[1], np.ndarray),\
                f"y should be a np.ndarray but found {type(y)}"
            train_data_creator = self._np_to_creator(data)
            valid_data_creator = self._np_to_creator(validation_data)
            train_loader = train_data_creator(self.config)
            validation_loader = valid_data_creator(self.config)

        epoch_losses = []
        for i in range(epochs):
            train_loss = self._train_epoch(train_loader)
            epoch_losses.append(train_loss)
        train_stats = {"loss": np.mean(epoch_losses), "last_loss": epoch_losses[-1]}
        val_stats = self._validate(validation_loader, metric_name=metric, metric_func=metric_func)
        self.onnx_model_built = False
        return val_stats

    @staticmethod
    def to_torch(inp):
        if isinstance(inp, np.ndarray):
            return torch.from_numpy(inp)
        if isinstance(inp, (pd.DataFrame, pd.Series)):
            return torch.from_numpy(inp.values)
        return inp

    @staticmethod
    def covert_input(data):
        x = PytorchBaseModel.to_torch(data[0]).float()
        y = PytorchBaseModel.to_torch(data[1]).float()
        return x, y

    def _train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        batch_idx = 0
        tqdm = None
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(train_loader))
        except ImportError:
            pass
        for x_batch, y_batch in train_loader:
            self.optimizer.zero_grad()
            yhat = self._forward(x_batch, y_batch)
            loss = self.criterion(yhat, y_batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            batch_idx += 1
            if tqdm:
                pbar.set_description("Loss: {}".format(loss.item()))
                pbar.update(1)
        if tqdm:
            pbar.close()
        train_loss = total_loss/batch_idx
        return train_loss

    def _forward(self, x, y):
        return self.model(x)

    def _validate(self, validation_loader, metric_name, metric_func=None):
        if not metric_name:
            assert metric_func, "You must input valid metric_func or metric_name"
            metric_name = metric_func.__name__
        self.model.eval()
        with torch.no_grad():
            yhat_list = []
            y_list = []
            for x_valid_batch, y_valid_batch in validation_loader:
                yhat_list.append(self.model(x_valid_batch).numpy())
                y_list.append(y_valid_batch.numpy())
            yhat = np.concatenate(yhat_list, axis=0)
            y = np.concatenate(y_list, axis=0)
        # val_loss = self.criterion(yhat, y)
        if metric_func:
            eval_result = metric_func(y, yhat)
        else:
            eval_result = Evaluator.evaluate(metric=metric_name,
                                             y_true=y, y_pred=yhat,
                                             multioutput='uniform_average')
        return {metric_name: eval_result}

    def _print_model(self):
        # print model and parameters
        print(self.model)
        print(len(list(self.model.parameters())))
        for i in range(len(list(self.model.parameters()))):
            print(list(self.model.parameters())[i].size())

    def evaluate(self, x, y, metrics=['mse'], multioutput="raw_values"):
        # reshape 1dim input
        x = self._reshape_input(x)
        y = self._reshape_input(y)

        yhat = self.predict(x)
        eval_result = [Evaluator.evaluate(m, y_true=y, y_pred=yhat, multioutput=multioutput)
                       for m in metrics]
        return eval_result

    def predict(self, x, mc=False, batch_size=32):
        # reshape 1dim input
        x = self._reshape_input(x)

        if not self.model_built:
            raise RuntimeError("You must call fit_eval or restore first before calling predict!")
        x = PytorchBaseModel.to_torch(x).float()
        if mc:
            self.model.train()
        else:
            self.model.eval()
        test_loader = DataLoader(TensorDataset(x),
                                 batch_size=int(batch_size))
        yhat_list = []
        for x_test_batch in test_loader:
            yhat_list.append(self.model(x_test_batch[0]).detach().numpy())
        yhat = np.concatenate(yhat_list, axis=0)
        return yhat

    def predict_with_uncertainty(self, x, n_iter=100):
        result = np.zeros((n_iter,) + (x.shape[0], self.config["output_feature_num"]))
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
        self.config = state["config"]
        self.model = self.model_creator(self.config)
        self.model.load_state_dict(state["model"])
        self.model_built = True
        self._create_optimizer()
        self.optimizer.load_state_dict(state["optimizer"])
        self._create_loss()

    def save(self, checkpoint):
        if not self.model_built:
            raise RuntimeError("You must call fit_eval or restore first before calling save!")
        state_dict = self.state_dict()
        torch.save(state_dict, checkpoint)

    def restore(self, checkpoint):
        state_dict = torch.load(checkpoint)
        self.load_state_dict(state_dict)

    def evaluate_with_onnx(self, x, y, metrics=['mse'], dirname=None, multioutput="raw_values"):
        # reshape 1dim input
        x = self._reshape_input(x)
        y = self._reshape_input(y)

        yhat = self.predict_with_onnx(x, dirname=dirname)
        eval_result = [Evaluator.evaluate(m, y_true=y, y_pred=yhat, multioutput=multioutput)
                       for m in metrics]
        return eval_result

    def _build_onnx(self, x, dirname=None):
        if not self.model_built:
            raise RuntimeError("You must call fit_eval or restore\
                               first before calling onnx methods!")
        try:
            import onnx
            import onnxruntime
        except:
            raise RuntimeError("You should install onnx and onnxruntime to use onnx based method.")
        if dirname is None:
            dirname = tempfile.mkdtemp(prefix="onnx_cache_")
        # code adapted from
        # https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
        torch.onnx.export(self.model,
                          x,
                          os.path.join(dirname, "cache.onnx"),
                          export_params=True,
                          opset_version=10,
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})
        self.onnx_model = onnx.load(os.path.join(dirname, "cache.onnx"))
        onnx.checker.check_model(self.onnx_model)
        self.ort_session = onnxruntime.InferenceSession(os.path.join(dirname, "cache.onnx"))
        self.onnx_model_built = True

    def predict_with_onnx(self, x, mc=False, dirname=None):
        # reshape 1dim input
        x = self._reshape_input(x)

        x = PytorchBaseModel.to_torch(x).float()
        if not self.onnx_model_built:
            self._build_onnx(x[0:1], dirname=dirname)

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        ort_inputs = {self.ort_session.get_inputs()[0].name: to_numpy(x)}
        ort_outs = self.ort_session.run(None, ort_inputs)
        return ort_outs[0]

    def _get_required_parameters(self):
        return {}

    def _get_optional_parameters(self):
        return {"batch_size",
                LR_NAME,
                "dropout",
                "optim",
                "loss"
                }


class PytorchModelBuilder(ModelBuilder):

    def __init__(self, model_creator,
                 optimizer_creator,
                 loss_creator):
        from zoo.orca.automl.pytorch_utils import validate_pytorch_loss, validate_pytorch_optim
        self.model_creator = model_creator
        optimizer = validate_pytorch_optim(optimizer_creator)
        self.optimizer_creator = optimizer
        loss = validate_pytorch_loss(loss_creator)
        self.loss_creator = loss

    def build(self, config):
        model = PytorchBaseModel(self.model_creator,
                                 self.optimizer_creator,
                                 self.loss_creator)
        model.build(config)
        return model
