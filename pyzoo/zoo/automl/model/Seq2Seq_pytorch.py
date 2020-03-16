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

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from zoo.automl.feature.time_sequence import TimeSequenceFeatureTransformer
from zoo.automl.model.abstract import BaseModel
from zoo.automl.common.util import *
from zoo.automl.common.metrics import Evaluator


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num, dropout):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.layer_num = layer_num

        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_num, dropout=dropout, batch_first=True)

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, input_seq):
        # input_seq = [batch_size, seq_len, feature_num]
        outputs, (hidden, cell) = self.rnn(input_seq)
        # outputs = [batch size, seq len, hidden dim]
        # hidden = [batch size, layer num, hidden dim]
        # cell = [batch size, layer num, hidden dim]

        # outputs are always from the top hidden layer
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, layer_num, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num

        self.rnn = nn.LSTM(output_dim, hidden_dim, layer_num, dropout=dropout, batch_first=True)

        self.fc_out = nn.Linear(hidden_dim, output_dim)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, decoder_input, hidden, cell):
        # input = [batch size]
        # hidden = [batch size, layer num, hidden dim]
        # cell = [batch size, layer num, hidden dim]

        # input = decoder_input.view(-1, 1)
        # decoder_input = [batch size, 1], since output_dim is 1
        decoder_input = decoder_input.unsqueeze(1)
        # decoder_input = [batch_size, 1, 1]

        output, (hidden, cell) = self.rnn(decoder_input, (hidden, cell))

        # output = [batch size, seq len, hidden dim]
        # hidden = [batch size, layer num, hidden dim]
        # cell = [batch size, layer num, hidden dim]

        # seq len will always be 1 in the decoder, therefore:
        # output = [batch size, 1, hidden dim]
        # hidden = [batch size, layer num, hidden dim]
        # cell = [batch size, layer num, hidden dim]
        prediction = self.fc_out(output.squeeze())
        # prediction = [batch size, output dim]

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, target_seq_len=1):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.target_seq_len = target_seq_len

    def forward(self, source, target=None):
        # past_seq_len
        batch_size = source.shape[0]

        output_dim = self.decoder.output_dim

        # tensor to store the predicted outputs
        target_seq = torch.zeros(batch_size, self.target_seq_len, output_dim)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(source)

        # Populate the first target sequence with end of encoding series value
        # decoder_input : [batch_size, output_dim]
        decoder_input = source[:, -1, :output_dim]

        for i in range(self.target_seq_len):
            decoder_output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            target_seq[:, i] = decoder_output
            if target is None:
                # in test mode
                decoder_input = decoder_output
            else:
                decoder_input = target[:, i]
        return target_seq


class Seq2SeqPytorch(BaseModel):

    def __init__(self, check_optional_config=True, future_seq_len=1):
        """
        Constructor of Vanilla LSTM model
        """
        self.model = None
        self.check_optional_config = check_optional_config
        self.future_seq_len = future_seq_len
        self.feature_num = None
        self.output_dim = None
        self.metric = None
        self.batch_size = None
        self.criterion = None
        self.optimizer = None

    def _get_configs(self, config):
        super()._check_config(**config)
        self.metric = config.get('metric', 'mean_squared_error')
        self.batch_size = config.get('batch_size', 32)
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

            outputs = self.model(input_seqs, target_seqs)
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
            # test_input is a list with one element
            test_out = self.model(test_input[0])
            test_out_list.append(test_out.detach().numpy())
        predictions = np.concatenate(test_out_list)
        y_pred = np.squeeze(predictions, axis=2)
        return y_pred

    def _print_model(self):
        # print model and parameters
        print(self.model)
        print(len(list(self.model.parameters())))
        for i in range(len(list(self.model.parameters()))):
            print(list(self.model.parameters())[i].size())

    def _expand_y(self, y):
        """
        expand dims for y.
        :param y:
        :return:
        """
        while len(y.shape) < 3:
            y = np.expand_dims(y, axis=2)
        return y

    def _pre_processing(self, x, y, validation_data):
        """
        pre_process input data
        1. expand dims for y and vay_y
        2. get input lengths
        :param x:
        :param y:
        :param validation_data:
        :return:
        """
        y = self._expand_y(y)
        self.feature_num = x.shape[2]
        self.output_dim = y.shape[2]
        if validation_data is not None:
            val_x, val_y = validation_data
            val_y = self._expand_y(val_y)
        return x, y, (val_x, val_y)

    def fit_eval(self, x, y, validation_data=None, mc=False, verbose=0, **config):
        self._get_configs(config)
        x, y, validation_data = self._pre_processing(x, y, validation_data)
        # get data
        train_loader = self._load_data((x, y), self.batch_size)
        if validation_data:
            val_loader = self._load_data(validation_data, self.batch_size)

        encoder = Encoder(self.feature_num, self.hidden_dim, self.layer_num, self.dropout)
        decoder = Decoder(self.output_dim, self.hidden_dim, self.layer_num, self.dropout)

        self.model = Seq2Seq(encoder, decoder, target_seq_len=self.future_seq_len)
        print(encoder)
        print(decoder)
        # self._print_model()

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        epochs = config.get('epochs', 20)
        assert (epochs > 0)
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
        y_pred = self._test_one_epoch(test_loader, mc=mc)
        return y_pred

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

        encoder = Encoder(self.feature_num, self.hidden_dim, self.layer_num, self.dropout)
        decoder = Decoder(self.output_dim, self.hidden_dim, self.layer_num, self.dropout)

        self.model = Seq2Seq(encoder, decoder, target_seq_len=self.future_seq_len)
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
