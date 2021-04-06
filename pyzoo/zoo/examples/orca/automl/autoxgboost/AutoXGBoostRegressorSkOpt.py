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

import argparse

import pandas as pd

from sklearn.model_selection import train_test_split
from zoo import init_spark_on_local, init_spark_on_yarn
from zoo.ray import RayContext
from zoo.orca.automl.xgboost.AutoXGBoost import AutoXGBoost
from zoo.zouwu.config.recipe import *

# for loguniform & choice
from ray import tune


class XgbRegressorSkOptRecipeNew(Recipe):
    def __init__(
            self,
            num_rand_samples=10,
            n_estimators_range=(50, 1000),
            max_depth_range=(2, 15),
            lr=(1e-4, 1e-1),
            min_child_weight=[1, 2, 3],
    ):
        """
        """
        super(self.__class__, self).__init__()

        self.num_samples = num_rand_samples

        self.n_estimators_range = n_estimators_range
        self.max_depth_range = max_depth_range
        self.lr = tune.loguniform(lr[0], lr[1])
        self.min_child_weight = tune.choice(min_child_weight)

    def search_space(self, all_available_features):
        space = {
            "n_estimators": tune.randint(self.n_estimators_range[0],
                                         self.n_estimators_range[1]),
            "max_depth": tune.randint(self.max_depth_range[0],
                                      self.max_depth_range[1]),
            "min_child_weight": self.min_child_weight,
            "lr": self.lr
        }
        return space

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AutoXGBRegressor example')
    parser.add_argument('-p', '--path', type=str,
                        help='Training data path')
    parser.add_argument('--hadoop_conf', type=str,
                        help='The path to the hadoop configuration folder. Required if you '
                             'wish to run on yarn clusters. Otherwise, run in local mode.')
    parser.add_argument('--conda_name', type=str,
                        help='The name of conda environment. Required if you '
                             'wish to run on yarn clusters.')
    parser.add_argument('--executor_cores', type=int, default=4,
                        help='The number of executor cores you want to use.')
    parser.add_argument('-n', '--num_workers', type=int, default=2,
                        help='The number of workers to be launched.')
    opt = parser.parse_args()
    if opt.hadoop_conf:
        assert opt.conda_name is not None, "conda_name must be specified for yarn mode"
        sc = init_spark_on_yarn(
            hadoop_conf=opt.hadoop_conf,
            conda_name=opt.conda_name,
            num_executors=opt.num_workers,
            executor_cores=opt.executor_cores)
    else:
        sc = init_spark_on_local(cores="*")
    ray_ctx = RayContext(sc=sc)
    ray_ctx.init()

    df = pd.read_csv(opt.path, encoding='latin-1')
    feature_cols = ["FIPS", "Lower 95% Confidence Interval", "Upper 95% Confidence Interval",
                    "Average Annual Count", "Recent 5-Year Trend"]
    target_col = "Age-Adjusted Incidence Rate"
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=2)

    num_rand_samples = 10
    n_estimators_range = (800, 1000)
    max_depth_range = (10, 15)
    lr = (1e-4, 1e-1)
    min_child_weight = [1, 2, 3]

    config = {'random_state': 2,
              'min_child_weight': 3,
              'n_jobs': 2}

    estimator = AutoXGBoost().regressor(feature_cols=feature_cols,
                                        target_col=target_col,
                                        config=config,
                                        search_alg="skopt",
                                        search_alg_params=None,
                                        scheduler="AsyncHyperBand",
                                        scheduler_params=dict(
                                            max_t=50,
                                            grace_period=1,
                                            reduction_factor=3,
                                            brackets=3,
                                        ),
                                        )

    pipeline = estimator.fit(train_df,
                             validation_df=val_df,
                             metric="rmse",
                             recipe=XgbRegressorSkOptRecipeNew(
                                 num_rand_samples=num_rand_samples,
                                 n_estimators_range=n_estimators_range,
                                 max_depth_range=max_depth_range,
                                 lr=lr,
                                 min_child_weight=min_child_weight
                             ),
                             )

    print("Training completed.")

    pred_df = pipeline.predict(val_df)

    rmse = pipeline.evaluate(val_df, metrics=["rmse"])
    print("Evaluate: the square root of mean square error is", rmse)

    # print(pipeline.model.model.evals_result_.get('validation_0').get('rmse'))

    ray_ctx.stop()
    sc.stop()
