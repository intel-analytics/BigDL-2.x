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
from sklearn.preprocessing import StandardScaler

from zoo.orca import init_orca_context, stop_orca_context
from zoo.chronos.autots.experimental.autotsestimator import AutoTSEstimator
from zoo.orca.automl.metrics import Evaluator
from zoo.chronos.data.repo_dataset import get_public_dataset


def get_tsdata():
    name = 'network_traffic'
    path = '~/.chronos/dataset/'
    tsdata_train, tsdata_val, \
        tsdata_test = get_public_dataset(name,
                                         path,
                                         redownload=False,
                                         with_split=True,
                                         val_ratio=0.1,
                                         test_ratio=0.1)
    minmax = StandardScaler()
    for tsdata in [tsdata_train, tsdata_val, tsdata_test]:
        tsdata.gen_dt_feature(one_hot_features=["HOUR", "WEEK"])\
              .impute("last")\
              .scale(minmax, fit=tsdata is tsdata_train)
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
    args = parser.parse_args()
    # init_orca_context
    num_nodes = 1 if args.cluster_mode == "local" else args.num_workers
    init_orca_context(cluster_mode=args.cluster_mode, cores=args.cores,
                      memory=args.memory, num_nodes=num_nodes, init_ray_on_spark=True)

    tsdata_train, tsdata_val, tsdata_test = get_tsdata()
    autoest = AutoTSEstimator(model='seq2seq',
                              search_space="normal",
                              past_seq_len=120,
                              future_seq_len=1,
                              cpus_per_trial=2,
                              metric='mse',
                              name='auto_seq2seq')

    tsppl = autoest.fit(data=tsdata_train,
                        validation_data=tsdata_val,
                        epochs=args.epochs,
                        batch_size=128,
                        n_sampling=args.n_sampling)

    best_config = autoest.get_best_config()
    print(best_config)

    tsppl.save("network_traffic_tsppl")
    test_x, test_y = tsdata_test.roll(lookback=best_config['past_seq_len'],
                                      horizon=best_config['future_seq_len']).to_numpy()
    unscale_test_y = tsdata_test.unscale_numpy(test_y)

    yhat = tsppl.predict(tsdata_test, batch_size=32)
    mse, smape = [Evaluator.evaluate(m,
                                     y_true=unscale_test_y,
                                     y_pred=yhat[:-1],
                                     multioutput="uniform_average") for m in ['mse', 'smape']]
    print(f'mse is: {mse}, smape is: {smape}')

    mse, smape = tsppl.evaluate(tsdata_test, metrics=['mse', 'smape'])
    print(f"evaluate mse is: {mse}")
    print(f"evaluate smape is: {smape}")

    # with_onnx
    my_tsppl = tsppl.load("network_traffic_tsppl")
    yaht_onnx = my_tsppl.predict_with_onnx(tsdata_test, batch_size=32)
    mse, smape = [Evaluator.evaluate(m,
                                     y_pred=yaht_onnx[:-1],
                                     y_true=unscale_test_y,
                                     multioutput="uniform_average") for m in ['mse', 'smape']]
    print(f'onnx mse is: {mse}, onnx smape is: {smape}')
    mse, smape = my_tsppl.evaluate_with_onnx(tsdata_test,
                                             metrics=['mse', 'smape'],
                                             multioutput='uniform_average')
    print(f'evaluate_onnx mse is: {mse}')
    print(f'evaluate_onnx smape is: {smape}')

    # stop_context
    stop_orca_context()
