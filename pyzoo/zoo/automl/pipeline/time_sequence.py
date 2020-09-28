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
import time

from zoo.automl.common.metrics import Evaluator
from zoo.automl.pipeline.abstract import Pipeline
from zoo.automl.common.util import *
from zoo.automl.feature.time_sequence import TimeSequenceFeatureTransformer
from zoo.automl.model.time_sequence import TimeSequenceModel
from zoo.automl.common.parameters import *


class TimeSequencePipeline(Pipeline):

    def __init__(self, feature_transformers=None, model=None, config=None, name=None):
        """
        initialize a pipeline
        :param model: the internal model
        :param feature_transformers: the feature transformers
        """
        self.feature_transformers = feature_transformers
        self.model = model
        self.config = config
        self.name = name
        self.time = time.strftime("%Y%m%d-%H%M%S")

    def describe(self):
        init_info = ['future_seq_len', 'dt_col', 'target_col', 'extra_features_col', 'drop_missing']
        print("**** Initialization info ****")
        for info in init_info:
            print(info + ":", self.config[info])
        print("")

    def fit(self, input_df, validation_df=None, mc=False, epoch_num=20):
        x, y = self.feature_transformers.transform(input_df, is_train=True)
        if validation_df is not None and not validation_df.empty:
            validation_data = self.feature_transformers.transform(validation_df)
        else:
            validation_data = None
        new_config = {'epochs': epoch_num}
        self.model.fit_eval(x, y, validation_data, mc=mc, verbose=1, **new_config)
        print('Fit done!')

    def _is_val_df_valid(self, validation_df):
        df_not_empty = isinstance(validation_df, pd.DataFrame) and not validation_df.empty
        df_list_not_empty = isinstance(validation_df, list) \
            and validation_df and not all([d.empty for d in validation_df])
        if validation_df is not None and (df_not_empty or df_list_not_empty):
            return True
        else:
            return False

    def _check_configs(self):
        required_configs = {'future_seq_len'}
        if not self.config.keys() & required_configs:
            raise ValueError("Missing required parameters in configuration. " +
                             "Required parameters are: " + str(required_configs))
        default_config = {'dt_col': 'datetime', 'target_col': 'value', 'extra_features_col': None,
                          'drop_missing': True, 'past_seq_len': 2, 'batch_size': 64, 'lr': 0.001,
                          'dropout': 0.2, 'epochs': 10, 'metric': 'mse'}
        for config, value in default_config.items():
            if config not in self.config:
                print('Config: \'{}\' is not specified. '
                      'A default value of {} will be used.'.format(config, value))

    def get_default_configs(self):
        default_configs = {'dt_col': 'datetime',
                           'target_col': 'value',
                           'extra_features_col': None,
                           'drop_missing': True,
                           'future_seq_len': 1,
                           'past_seq_len': 2,
                           'batch_size': 64,
                           'lr': 0.001,
                           'dropout': 0.2,
                           'epochs': 10,
                           'metric': 'mean_squared_error'}
        print("**** default config: ****")
        for config in default_configs:
            print(config + ":", default_configs[config])
        print("You can change any fields in the default configs by passing into "
              "fit_with_fixed_configs(). Otherwise, the default values will be used.")
        return default_configs

    def fit_with_fixed_configs(self, input_df, validation_df=None, mc=False, **user_configs):
        """
        Fit pipeline with fixed configs. The model will be trained from initialization
        with the hyper-parameter specified in configs. The configs contain both identity configs
        (Eg. "future_seq_len", "dt_col", "target_col", "metric") and automl tunable configs
        (Eg. "past_seq_len", "batch_size").
        We recommend calling get_default_configs to see the name and default values of configs you
        you can specify.
        :param input_df: one data frame or a list of data frames
        :param validation_df: one data frame or a list of data frames
        :param user_configs: you can overwrite or add more configs with user_configs. Eg. "epochs"
        :return:
        """
        # self._check_configs()
        if self.config is None:
            self.config = self.get_default_configs()
        if user_configs is not None:
            self.config.update(user_configs)
        ft_id_config_set = {'future_seq_len', 'dt_col', 'target_col',
                            'extra_features_col', 'drop_missing'}
        ft_id_configs = {a: self.config[a] for a in ft_id_config_set}
        self.feature_transformers = TimeSequenceFeatureTransformer(**ft_id_configs)
        model_id_config_set = {'future_seq_len'}
        ft_id_configs = {a: self.config[a] for a in model_id_config_set}
        self.model = TimeSequenceModel(check_optional_config=False, **ft_id_configs)
        all_available_features = self.feature_transformers.get_feature_list(input_df)
        self.config.update({"selected_features": all_available_features})
        (x_train, y_train) = self.feature_transformers.fit_transform(input_df, **self.config)
        if self._is_val_df_valid(validation_df):
            validation_data = self.feature_transformers.transform(validation_df)
        else:
            validation_data = None

        self.model.fit_eval(x_train, y_train,
                            validation_data=validation_data,
                            mc=mc,
                            verbose=1, **self.config)

    def evaluate(self,
                 input_df,
                 metrics=["mse"],
                 multioutput='raw_values'
                 ):
        """
        evaluate the pipeline
        :param input_df:
        :param metrics: subset of ['mean_squared_error', 'r_square', 'sMAPE']
        :param multioutput: string in ['raw_values', 'uniform_average']
                'raw_values' :
                    Returns a full set of errors in case of multioutput input.
                'uniform_average' :
                    Errors of all outputs are averaged with uniform weight.
        :return:
        """
        if isinstance(metrics, str):
            metrics = [metrics]
        # if not isinstance(metrics, list):
        #    raise ValueError("Expected metrics to be a list!")

        x, y = self.feature_transformers.transform(input_df, is_train=True)
        y_pred = self.model.predict(x)
        if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
            multioutput = 'uniform_average'
        y_unscale, y_pred_unscale = self.feature_transformers.post_processing(input_df,
                                                                              y_pred,
                                                                              is_train=True)

        return [Evaluator.evaluate(m, y_unscale, y_pred_unscale, multioutput=multioutput)
                for m in metrics]

    def predict(self, input_df):
        """
        predict test data with the pipeline fitted
        :param input_df:
        :return:
        """
        x, _ = self.feature_transformers.transform(input_df, is_train=False)
        y_pred = self.model.predict(x)
        y_output = self.feature_transformers.post_processing(input_df, y_pred, is_train=False)
        return y_output

    def predict_with_uncertainty(self, input_df, n_iter=100):
        x, _ = self.feature_transformers.transform(input_df, is_train=False)
        y_pred, y_pred_uncertainty = self.model.predict_with_uncertainty(x=x, n_iter=n_iter)
        y_output = self.feature_transformers.post_processing(input_df, y_pred, is_train=False)
        y_uncertainty = self.feature_transformers.unscale_uncertainty(y_pred_uncertainty)
        return y_output, y_uncertainty

    def save(self, ppl_file=None):
        """
        save pipeline to file, contains feature transformer, model, trial config.
        :param ppl_file:
        :return:
        """
        ppl_file = ppl_file or os.path.join(DEFAULT_PPL_DIR, "{}_{}.ppl".
                                            format(self.name, self.time))
        save_zip(ppl_file, self.feature_transformers, self.model, self.config)
        print("Pipeline is saved in", ppl_file)
        return ppl_file

    def config_save(self, config_file=None):
        """
        save all configs to file.
        :param config_file:
        :return:
        """
        config_file = config_file or os.path.join(DEFAULT_CONFIG_DIR, "{}_{}.json".
                                                  format(self.name, self.time))
        save_config(config_file, self.config, replace=True)
        return config_file


def load_ts_pipeline(file):
    feature_transformers = TimeSequenceFeatureTransformer()
    model = TimeSequenceModel(check_optional_config=False)

    all_config = restore_zip(file, feature_transformers, model)
    ts_pipeline = TimeSequencePipeline(feature_transformers=feature_transformers,
                                       model=model,
                                       config=all_config)
    print("Restore pipeline from", file)
    return ts_pipeline


def load_xgboost_pipeline(file, model_type="regressor"):
    from zoo.automl.feature.identity_transformer import IdentityTransformer
    from zoo.automl.model import XGBoost
    feature_transformers = IdentityTransformer()
    model = XGBoost(model_type=model_type)

    all_config = restore_zip(file, feature_transformers, model)
    ts_pipeline = TimeSequencePipeline(feature_transformers=feature_transformers,
                                       model=model,
                                       config=all_config)
    print("Restore pipeline from", file)
    return ts_pipeline
