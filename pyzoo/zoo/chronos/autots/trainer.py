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

from zoo.chronos.data import TSDataset
import zoo.orca.automl.hp as hp


def check_lstm_params(search_space):
    if search_space.get('input_feature_num') is None \
            or search_space.get('output_target_num') is None:
        raise ValueError("Need to specify input_feature_num and "
                         "output_target_num in search_space")


def check_tcn_params(search_space):
    if search_space.get('input_feature_num') is None \
            or search_space.get('output_target_num') is None:
        raise ValueError("Need to specify input_feature_num and "
                         "output_target_num in search_space")


class AutoTSTrainer:
    """
    Automated Trainer.
    """

    def __init__(self,
                 model="lstm",
                 search_space=dict(),
                 metric="mse",
                 loss=None,
                 preprocess=False,
                 backend="torch",
                 logs_dir="/tmp/autots_trainer",
                 cpus_per_trial=1,
                 name="autots_trainer"
                 ):
        """
        AutoTSTrainer trains a model for time series forecasting.
        User can choose one of the built-in models, or pass in a customized pytorch or keras model
        for tuning using AutoML.
        :param model: a string or a model creation function
            a string indicates a built-in model, currently "lstm", "tcn" are supported
        :param search_space: hyper parameter configurations. Some parameters are searchable and some
            are fixed parameters (such as input dimensions, etc.) Read the API docs for each auto model
        :param metric: String. The evaluation metric name to optimize. e.g. "mse"
        :param loss: String or pytorch/tf.keras loss instance or pytorch loss creator function.
        :param preprocess: whether to enable data preprocessing (rolling, feature generation, etc.)
        :param backend: The backend of the lstm model. We only support backend as "torch" for now.
        :param logs_dir: Local directory to save logs and results. It defaults to "/tmp/auto_lstm"
        :param cpus_per_trial: Int. Number of cpus for each trial. It defaults to 1.
        :param name: name of the AutoLSTM. It defaults to "auto_lstm"
        :param preprocess: Whether to enable feature processing
        """

        if loss is None:
            self.loss = search_space.get("loss")
        else:
            self.loss = loss
        if self.loss is None:
            if backend == "torch":
                import torch
                self.loss = torch.nn.MSELoss()
            else:
                # TODO support more backends
                raise ValueError(f"We only support backend as torch. Got {backend}")

        self.metric = metric
        self.backend = backend

        import types
        if model == "lstm":
            # if model is lstm
            from zoo.chronos.autots.model.auto_lstm import AutoLSTM
            check_lstm_params(search_space)
            self.model = AutoLSTM(
                input_feature_num=search_space.get('input_feature_num'),
                output_target_num=search_space.get('output_target_num'),
                optimizer=search_space.get('optimizer', "Adam"),
                loss=self.loss,
                metric=self.metric,
                hidden_dim=search_space.get('hidden_dim', 32),
                layer_num=search_space.get('layer_num', 1),
                lr=search_space.get('lr', 0.001),
                dropout=search_space.get('dropout', 0.2),
                backend=backend,
                logs_dir=logs_dir,
                cpus_per_trial=cpus_per_trial,
                name=name,
            )
        elif model == "tcn":
            # if model is tcn
            from zoo.chronos.autots.model.auto_tcn import AutoTCN
            check_tcn_params(search_space)
            self.model = AutoTCN(
                input_feature_num=search_space.get('input_feature_num'),
                output_target_num=search_space.get('output_target_num'),
                past_seq_len=self.past_seq_len,
                future_seq_len=self.future_seq_len,
                optimizer=search_space.get('optimizer', "Adam"),
                loss=self.loss,
                metric=self.metric,
                hidden_units=search_space.get('hidden_units'),
                levels=search_space.get('levels'),
                num_channels=search_space.get('num_channels'),
                kernel_size=search_space.get('kernel_size', 7),
                lr=search_space.get('lr', 0.001),
                dropout=0.2,
                backend=backend,
                logs_dir=logs_dir,
                cpus_per_trial=cpus_per_trial,
                name=name,
            )
        elif isinstance(model, types.FunctionType):
            # TODO if model is user defined
            self.model = model
            raise ValueError("3rd party model is not support for now")

        self.preprocess = preprocess

    def fit(self,
            data,
            epochs=1,
            batch_size=32,
            validation_data=None,
            metric_threshold=None,
            n_sampling=1,
            search_alg=None,
            search_alg_params=None,
            scheduler=None,
            scheduler_params=None
            ):
        """
        fit using AutoEstimator
        :param data: train data.
               For backend of "torch", data can be a TSDataset or a function that takes a
               config dictionary as parameter and returns a PyTorch DataLoader.
               For backend of "keras", data can be a TSDataset.
        :param epochs: Max number of epochs to train in each trial. Defaults to 1.
               If you have also set metric_threshold, a trial will stop if either it has been
               optimized to the metric_threshold or it has been trained for {epochs} epochs.
        :param batch_size: Int or hp sampling function from an integer space. Training batch size.
               It defaults to 32.
        :param validation_data: Validation data. Validation data type should be the same as data.
        :param metric_threshold: a trial will be terminated when metric threshold is met
        :param n_sampling: Number of times to sample from the search_space. Defaults to 1.
               If hp.grid_search is in search_space, the grid will be repeated n_sampling of times.
               If this is -1, (virtually) infinite samples are generated
               until a stopping condition is met.
        :param search_alg: str, all supported searcher provided by ray tune
               (i.e."variant_generator", "random", "ax", "dragonfly", "skopt",
               "hyperopt", "bayesopt", "bohb", "nevergrad", "optuna", "zoopt" and
               "sigopt")
        :param search_alg_params: extra parameters for searcher algorithm besides search_space,
               metric and searcher mode
        :param scheduler: str, all supported scheduler provided by ray tune
        :param scheduler_params: parameters for scheduler
        """
        train_d = data
        val_d = validation_data
        if self.preprocess is True:
            # TODO do we need more customizations for feature search?
            # a little bit of hacking to modify automodel's search_space before fit
            train_d, val_d = self.prepare_feature_search(
                    search_space=self.model.search_space,
                    train_data=data,
                    val_data=validation_data,
            )

        self.model.fit(
            data=train_d,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_d,
            metric_threshold=metric_threshold,
            n_sampling=n_sampling,
            search_alg=search_alg,
            search_alg_params=search_alg_params,
            scheduler=scheduler,
            scheduler_params=scheduler_params
        )

    def prepare_feature_search(self, search_space, train_data, val_data=None):
        """
        prepare the data creators and add selected features to search_space
        :param search_space: the search space
        :param train_data: train data
        :param val_data: validation data
        :return: data creators from train and validation data
        """
        import torch
        from torch.utils.data import Dataset, DataLoader
        from sklearn.preprocessing import StandardScaler
        import ray

        standard_scaler = StandardScaler()

        class TorchDataset(Dataset):
            def __init__(self, x, y):
                self.x = torch.from_numpy(x).float()
                self.y = torch.from_numpy(y).float()

            def __len__(self):
                return self.x.shape[0]

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]

        # append feature selection into search space
        train_data.gen_dt_feature().scale(standard_scaler, fit=True)
        all_features = train_data.feature_col
        if search_space.get('selected_features') is not None:
            raise ValueError("Do not specify ""selected_features"" in search space."
                             " The system will automatically generate this config for you")
        search_space['selected_features'] = hp.choice_n(all_features, 3, len(all_features))

        train_data_id = ray.put(train_data)

#        train_data_2 = ray.get(train_data_id)
        # train_data.scale(standard_scaler, fit=True) \
        #     .roll(lookback=self.past_seq_len,
        #           horizon=self.future_seq_len,
        #           feature_col=["DAYOFYEAR(datetime)"]) \
        #     .to_numpy()

        def train_data_creator(config):
            """
            train data creator function
            :param config:
            :return:
            """
            train_d = ray.get(train_data_id)
            # print(type(config['selected_features']), config['selected_features'])
            x, y = train_d.roll(lookback=config.get('past_seq_len', 5),
                                horizon=config.get('future_seq_len', 1),
                                feature_col=config['selected_features']) \
                          .to_numpy()
            print(x.shape, y.shape)
            return DataLoader(TorchDataset(x, y),
                              batch_size=config["batch_size"],
                              shuffle=True)

        def val_data_creator(config):
            """
            train data creator function
            :param config:
            :return:
            """
            x, y = val_data.gen_dt_feature() \
                           .scale(standard_scaler, fit=False) \
                           .roll(lookback=config.get('past_seq_len', 5),
                                 horizon=config.get('future_seq_len', 1),
                                 feature_col=config['selected_features']) \
                           .to_numpy()

            return DataLoader(TorchDataset(x, y),
                              batch_size=config["batch_size"],
                              shuffle=True)

        return train_data_creator, val_data_creator

    def get_best_model(self):
        """
        Get the tuned model
        :return:
        """
        return self.model.get_best_model()

    def get_best_config(self):
        """
        Get the best configuration
        :return:
        """
        pass

    def get_pipeline(self):
        """
        TODO do we still need to return a full pipeline?
        If not, we still need to tell user how to reconstruct the entire pipeline
        including data processing and the model
        :return:
        """
        pass