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
                name=name
            )
        elif model == "tcn":
            # if model is tcn
            from zoo.chronos.autots.model.auto_tcn import AutoTCN
            check_tcn_params(search_space)
            self.model = AutoTCN(
                input_feature_num=search_space.get('input_feature_num'),
                output_target_num=search_space.get('output_target_num'),
                past_seq_len=search_space.get('past_seq_len', 24),
                future_seq_len=search_space.get('future_seq_len', 1),
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
                name=name
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
               For backend of "torch", data can be a tuple of ndarrays or a function that takes a
               config dictionary as parameter and returns a PyTorch DataLoader.
               For backend of "keras", data can be a tuple of ndarrays.
               If data is a tuple of ndarrays, it should be in the form of (x, y),
                where x is training input data and y is training target data.
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
        if self.preprocess is True:
            # TODO add the data preprocessing part
            # which operations are included, configurable?
            # 1. fill
            # feature generation
            pass
        self.model.fit(
            data=data,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            metric_threshold=metric_threshold,
            n_sampling=n_sampling,
            search_alg=search_alg,
            search_alg_params=search_alg_params,
            scheduler=scheduler,
            scheduler_params=scheduler_params
        )

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
