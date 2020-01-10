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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from zoo.automl.feature.time_sequence import TimeSequenceFeatureTransformer
from zoo.automl.model.abstract import BaseModel
from zoo.automl.common.util import *
from zoo.automl.common.metrics import Evaluator


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num, dropout, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_num, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, input_seq):
        # init hidden
        # input_seq: (batch_size, seq_len, feature_num)
        # h0 = torch.zeros(self.layer_num, input_seq.size[0], self.hidden_dim)
        # c0 = torch.zeros(self.layer_num, input_seq.size[0], self.hidden_dim)
        lstm_out, hidden = self.lstm(input_seq)
        # lstm_out: (batch_size, hidden_dim
        # reshaping the outputs to feed in fully connected layer
        # out = lstm_out[-1].contiguous().view(-1, self.hidden_dim)
        # out = self.linear(out, len(input_seq), -1)
        out = self.fc(lstm_out[:, -1, :])
        return out


class VanillaLSTMPytorch(BaseModel):

    def __init__(self, check_optional_config=True, future_seq_len=1):
        """
        Constructor of Vanilla LSTM model
        """
        self.model = None
        self.check_optional_config = check_optional_config
        self.future_seq_len = future_seq_len
        self.feature_num = None
        self.output_dim = 1
        self.metric = None
        self.criterion = None
        self.optimizer = None

    def _get_configs(self, config):
        super()._check_config(**config)
        self.metric = config.get('metric', 'mean_squared_error')
        self.batch_size = config.get('batch_size', 1024)
        self.hidden_dim = config.get('hidden_dim', 32)
        self.layer_num = config.get('layer_num', 2)
        self.dropout = config.get('dropout', 0.2)
        self.lr = config.get("lr", 0.001)

    def _load_data(self, input_data, batch_size):
        x, y = input_data
        data = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        data_loader = DataLoader(data, shuffle=True, batch_size=batch_size)
        return data_loader

    def _train_one_epoch(self, train_loader):
        self.model.train()
        train_losses = []
        for input_seqs, target_seqs in train_loader:
            self.model.zero_grad()

            outputs = self.model(input_seqs)
            loss = self.criterion(outputs, target_seqs)

            # get gradients
            loss.backward()

            # update parameters
            self.optimizer.step()

            train_losses.append(loss.item())
        return np.mean(train_losses)

    def _val_one_epoch(self, val_loader):
        self.model.eval()
        val_losses = []
        for val_input, val_target in val_loader:
            val_out = self.model(val_input)
            val_loss = self.criterion(val_out, val_target)
            val_losses.append(val_loss.item())
        return np.mean(val_losses)

    def _test_one_epoch(self, test_loader, mc=False):
        if not mc:
            self.model.eval()
        else:
            self.model.train()
        test_out_list = []
        for test_input in test_loader:
            test_out = self.model(test_input[0])
            test_out_list.append(test_out.detach().numpy())
        return np.concatenate(test_out_list)

    def _print_model(self):
        # print model and parameters
        print(self.model)
        print(len(list(self.model.parameters())))
        for i in range(len(list(self.model.parameters()))):
            print(list(self.model.parameters())[i].size())

    def fit_eval(self, x, y, validation_data, mc=False, verbose=0, **config):
        self._get_configs(config)
        # get data
        train_loader = self._load_data((x, y), self.batch_size)
        if validation_data:
            val_loader = self._load_data(validation_data, self.batch_size)

        self.feature_num = x.shape[2]
        self.output_dim = 1
        self.model = LSTMModel(self.feature_num, self.hidden_dim, self.layer_num, self.dropout,
                               self.output_dim)
        # self._print_model()

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        epochs = config.get('epochs', 20)
        assert(epochs > 0)
        val_epoch = 1
        for i in range(epochs):
            train_loss = self._train_one_epoch(train_loader)
            if verbose == 1:
                print("Epoch : {}/{}...".format(i, epochs),
                      "Loss: {:.6f}...".format(train_loss),
                      )
            if i % val_epoch == 0:
                if validation_data:
                    val_loss = self._val_one_epoch(val_loader)
                if verbose == 1:
                    print("Val loss: {:.6f}...".format(val_loss))
        if validation_data:
            result = val_loss
        else:
            result = train_loss
        return result

    def evaluate(self, x, y, metric=['mse']):
        """
        Evaluate on x, y
        :param x: input
        :param y: target
        :param metric: a list of metrics in string format
        :return: a list of metric evaluation results
        """
        y_pred = self.predict(x)
        assert y_pred.shape == y.shape
        return [Evaluator.evaluate(m, y, y_pred) for m in metric]

    def predict(self, x, mc=False):
        """
        Prediction on x.
        :param x: input
        :return: predicted y
        """
        test_x = TensorDataset(torch.from_numpy(x))
        test_loader = DataLoader(test_x, shuffle=False, batch_size=self.batch_size)
        predictions = self._test_one_epoch(test_loader, mc=mc)
        return predictions

    def predict_with_uncertainty(self, x, n_iter=100):
        test_x = TensorDataset(torch.from_numpy(x))
        test_loader = DataLoader(test_x, shuffle=False, batch_size=self.batch_size)
        result = np.zeros((n_iter,) + (x.shape[0], self.future_seq_len))

        for i in range(n_iter):
            result[i, :, :] = self._test_one_epoch(test_loader, mc=True)

        prediction = result.mean(axis=0)
        uncertainty = result.std(axis=0)
        return prediction, uncertainty

    def save(self, model_path, config_path):
        """
        save model to file.
        :param model_path: the model file.
        :param config_path: the config file
        :return:
        """
        torch.save(self.model.state_dict(), model_path)
        # os.rename("vanilla_lstm_tmp.h5", model_path)

        config_to_save = {
            "future_seq_len": self.future_seq_len,
            "feature_num": self.feature_num,
            "metric": self.metric,
            "batch_size": self.batch_size,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "layer_num": self.layer_num,
            "output_dim": self.output_dim,
            # "lr": self.lr
        }
        save_config(config_path, config_to_save)

    def restore(self, model_path, **config):
        """
        restore model from file
        :param model_path: the model file
        :param config: the trial config
        :return: the restored model
        """
        # self.model = None
        # self._build(**config)
        # self.model = keras.models.load_model(model_path)
        # self.model.load_weights(file_path)

        self.future_seq_len = config["future_seq_len"]
        self.feature_num = config["feature_num"]
        self.output_dim = config["output_dim"]
        # for continuous training
        saved_configs = ["future_seq_len", "metric", "batch_size", "hidden_dim",
                         "dropout", "layer_num", "output_dim"]
        assert all([c in config for c in saved_configs])
        self._get_configs(config)

        self.model = LSTMModel(self.feature_num, self.hidden_dim, self.layer_num, self.dropout,
                               self.output_dim)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def _get_required_parameters(self):
        return {
            # 'input_shape_x',
            # 'input_shape_y',
            # 'out_units'
        }

    def _get_optional_parameters(self):
        return {
            'hidden_dim',
            'layer_num',
            'hidden_dim',
            'dropout',
            'lr',
            'epochs',
            'batch_size'
        }


if __name__ == "__main__":
    dataset_path = os.getenv("ANALYTICS_ZOO_HOME") + "/bin/data/NAB/nyc_taxi/nyc_taxi.csv"
    df = pd.read_csv(dataset_path)
    from zoo.automl.common.util import split_input_df

    train_df, val_df, test_df = split_input_df(df, val_split_ratio=0.1, test_split_ratio=0.1)
    future_seq_len = 1
    feature_transformer = TimeSequenceFeatureTransformer(future_seq_len=future_seq_len)
    model = VanillaLSTMPytorch(check_optional_config=False, future_seq_len=future_seq_len)

    config = {
        # 'input_shape_x': x_train.shape[1],
        # 'input_shape_y': x_train.shape[-1],
        'selected_features': ['IS_WEEKEND(datetime)', 'MONTH(datetime)', 'IS_AWAKE(datetime)',
                              'HOUR(datetime)'],
        'batch_size': 64,
        'epochs': 20
    }
    x_train, y_train = feature_transformer.fit_transform(train_df, **config)
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    x_test, y_test = feature_transformer.transform(test_df, is_train=True)
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    mc = True
    print("fit_eval:", model.fit_eval(x_train, y_train, validation_data=(x_test, y_test),
                                      verbose=1,
                                      mc=mc, **config))
    print("evaluate:", model.evaluate(x_test, y_test))
    if not mc:
        y_pred = model.predict(x_test)
        assert y_pred.shape == (x_test.shape[0], future_seq_len)

        dirname = 'tmp'
        model.save('tmp.pth', 'tmp.json')
        config = load_config('tmp.json')
        model.restore('tmp.pth', **config)
        # restore(dirname, model=model, config=config)
        os.remove('tmp.pth')
        os.remove('tmp.json')

        y_pred_after = model.predict(x_test)
        assert np.allclose(y_pred, y_pred_after)
    else:
        y_pred, y_uncertainty = model.predict_with_uncertainty(x_test, n_iter=3)
        assert y_pred.shape == (x_test.shape[0], future_seq_len)
        assert y_uncertainty.shape == (x_test.shape[0], future_seq_len)
        assert np.any(y_uncertainty)

    # from matplotlib import pyplot as plt
    #
    # y_test = np.squeeze(y_test)
    # y_pred = np.squeeze(y_pred)

    # def plot_result(y_test, y_pred):
    #     # target column of dataframe is "value"
    #     # past sequence length is 50
    #     # pred_value = pred_df["value"].values
    #     # true_value = test_df["value"].values[50:]
    #     fig, axs = plt.subplots()
    #
    #     axs.plot(y_pred, color='red', label='predicted values')
    #     axs.plot(y_test, color='blue', label='actual values')
    #     axs.set_title('the predicted values and actual values (for the test data)')
    #
    #     plt.xlabel('test data index')
    #     plt.ylabel('number of taxi passengers')
    #     plt.legend(loc='upper left')
    #     plt.savefig("pytorch_lstm_result.png")
    #
    #
    # plot_result(y_test, y_pred)








