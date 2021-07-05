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

from zoo.common.nncontext import init_nncontext
from zoo.models.anomalydetection import AnomalyDetector
import pandas as pd
from pyspark.sql import SQLContext
from pyspark import sql
from optparse import OptionParser
import sys

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--input_dir", dest="input_dir",
                      help="Required. The path where NBA nyc_taxi.csv locates.")
    parser.add_option("-b", "--batch_size", dest="batch_size", default="1024",
                      help="The number of samples per gradient update. Default is 1024.")
    parser.add_option("--nb_epoch", dest="nb_epoch", default="20",
                      help="The number of epochs to train the model. Default is 20.")
    parser.add_option("--unroll_len", dest="unroll_len", default="24",
                      help="The length of precious values to predict future value. Default is 24.")

    (options, args) = parser.parse_args(sys.argv)

    # if input_dir is not given
    if not options.input_dir:
        parser.print_help()
        parser.error('input_dir is required')

    sc = init_nncontext("Anomaly Detection Example")

    sqlContext = sql.SQLContext(sc)

    def load_and_scale(input_path):
        df = pd.read_csv(input_path)
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df['hours'] = df['datetime'].dt.hour
        df['awake'] = (((df['hours'] >= 6) & (df['hours'] <= 23)) | (df['hours'] == 0)).astype(int)
        print(df.head(10))
        sqlContext = SQLContext(sc)
        dfspark = sqlContext.createDataFrame(df[["value", "hours", "awake"]])
        feature_size = len(["value", "hours", "awake"])
        return AnomalyDetector.standardScale(dfspark), feature_size

    df_scaled, feature_size = load_and_scale(options.input_dir)
    data_rdd = df_scaled.rdd.map(lambda row: [x for x in row])
    unrolled = AnomalyDetector.unroll(data_rdd, int(options.unroll_len), predict_step=1)
    [train, test] = AnomalyDetector.train_test_split(unrolled, 1000)

    model = AnomalyDetector(feature_shape=(int(options.unroll_len), feature_size),
                            hidden_layers=[8, 32, 15], dropouts=[0.2, 0.2, 0.2])
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
    model.fit(train, batch_size=int(options.batch_size), nb_epoch=int(options.nb_epoch))
    test.cache()
    y_predict = model.predict(test, batch_per_thread=int(options.batch_size))\
        .map(lambda x: float(x[0]))
    y_truth = test.map(lambda x: float(x.label.to_ndarray()[0]))
    anomalies = AnomalyDetector.detect_anomalies(y_predict, y_truth, 50)

    print("anomalies: ", anomalies.take(10)[0:10])
    print("finished...")
    sc.stop()
