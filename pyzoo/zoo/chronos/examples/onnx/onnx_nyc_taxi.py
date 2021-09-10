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
import os
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from zoo.orca import init_orca_context, stop_orca_context
from zoo.chronos.autots.experimental.autotsestimator import AutoTSEstimator
from zoo.chronos.data import TSDataset
from zoo.orca.automl.metrics import Evaluator


def get_data(args):
    df = pd.read_csv(args.datadir, parse_dates=['timestamp'])
    return df


def get_tsdata():
    df = get_data(args)
    tsdata_train, tsdata_val, \
        tsdata_test = TSDataset.from_pandas(df,
                                            target_col=['value'],
                                            dt_col='timestamp',
                                            with_split=True,
                                            val_ratio=0.1,
                                            test_ratio=0.1)
    stand = StandardScaler()
    for tsdata in [tsdata_train, tsdata_val, tsdata_test]:
        tsdata.gen_dt_feature(one_hot_features=['HOUR', 'WEEK'])\
              .impute("last")\
              .scale(stand, fit=tsdata is tsdata_train)
    return tsdata_train, tsdata_val, tsdata_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=2,
                        help="The number of nodes to be used in the cluster. "
                        "You can change it depending on your own cluster setting.")
    parser.add_argument('--cluster_mode', type=str, default='local',
                        help="The mode for the Spark cluster.")
    parser.add_argument('--cores', type=int, default=4,
                        help="The number of cpu cores you want to use on each node."
                        "You can change it depending on your own cluster setting.")
    parser.add_argument('--memory', type=str, default="10g",
                        help="The memory you want to use on each node."
                        "You can change it depending on your own cluster setting.")

    parser.add_argument("--epochs", type=int, default=2,
                        help="Max number of epochs to train in each trial.")
    parser.add_argument("--n_sampling", type=int, default=1,
                        help="Number of times to sample from the search_space.")
    parser.add_argument("--datadir", type=str,
                        default="https://raw.githubusercontent.com/numenta/NAB/"
                        "v1.0/data/realKnownCause/nyc_taxi.csv",
                        help='Specify the address of the file.')
    args = parser.parse_args()

    # init_orca_context
    num_nodes = 1 if args.cluster_mode == "local" else args.num_workers
    init_orca_context(cluster_mode=args.cluster_mode, cores=args.cores,
                      memory=args.memory, num_nodes=num_nodes, init_ray_on_spark=True)
    tsdata_train, tsdata_val, tsdata_test = get_tsdata()

    autoest = AutoTSEstimator(model='lstm',
                              search_space="normal",
                              past_seq_len=120,
                              future_seq_len=1,
                              cpus_per_trial=2,
                              metric='mse',
                              name='auto_lstm')

    tsppl = autoest.fit(data=tsdata_train,
                        validation_data=tsdata_val,
                        epochs=args.epochs,
                        batch_size=128,
                        n_sampling=args.n_sampling)
    tsppl.save("lstm_tsppl_nyc")
    best_config = autoest.get_best_config()
    print(best_config)

    test_x, test_y = tsdata_test.roll(lookback=best_config['past_seq_len'],
                                      horizon=best_config['future_seq_len']).to_numpy()
    unscale_test_y = tsdata_test.unscale_numpy(test_y)

    yhat = tsppl.predict(tsdata_test, batch_size=32)
    mse, smape = [Evaluator.evaluate(m,
                                     y_pred=yhat[:-1],
                                     y_true=unscale_test_y,
                                     multioutput="uniform_average") for m in ['mse', 'smape']]
    print(f'evaluate mse is: {np.mean(mse)}')
    print(f'evaluate smape is: {np.mean(smape)}')

    mse, smape = tsppl.evaluate(tsdata_test,
                                metrics=['mse', 'smape'],
                                multioutput="uniform_average")
    print(f'evaluate mse is: {np.mean(mse)}')
    print(f'evaluate smape is: {np.mean(smape)}')

    my_tsppl = tsppl.load("lstm_tsppl_nyc")
    yhat_onnx = my_tsppl.predict_with_onnx(tsdata_test)

    mse, smape = [Evaluator.evaluate(m,
                                     y_pred=yhat_onnx[:-1],
                                     y_true=unscale_test_y,
                                     multioutput="uniform_average") for m in ['mse', 'smape']]
    print(f'evaluate_onnx mse is: {np.mean(mse)}')
    print(f'evaluate_onnx smape is: {np.mean(smape)}')

    mse, smape = my_tsppl.evaluate_with_onnx(tsdata_test, metrics=['mse', 'smape'])
    print(f'evaluate_onnx mse is: {np.mean(mse)}')
    print(f'evaluate_onnx smape is: {np.mean(smape)}')

    # stop orca
    stop_orca_context()
