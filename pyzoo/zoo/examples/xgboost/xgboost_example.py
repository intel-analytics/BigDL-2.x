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
from optparse import OptionParser

import numpy as np
from bigdl.nn.criterion import *
from bigdl.nn.layer import *
from bigdl.optim.optimizer import *
from numpy.testing import assert_allclose
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from zoo.common.nncontext import *
from zoo.feature.common import *
from zoo.feature.image import *
from zoo.pipeline.api.keras import layers as ZLayer
from zoo.pipeline.api.keras.models import Model as ZModel
from zoo.pipeline.api.keras.optimizers import Adam as KAdam
from zoo.pipeline.nnframes import *
from zoo.util.tf import *
import csv


def Processdata(filepath, demo):
    '''
         preProcess the data read from filepath
    :param filepath:
    :return: assembledf:
    '''
    sparkConf = init_spark_conf().setMaster("local[1]").setAppName("testNNClassifer")
    sc = init_nncontext(sparkConf)
    sqlContext = SQLContext(sc)
    if demo:
        data = sc.parallelize([
            (1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 4.0, 8.0, 3.0, 116.3668),
            (1.0, 3.0, 8.0, 6.0, 5.0, 9.0, 5.0, 6.0, 7.0, 4.0, 116.367),
            (2.0, 1.0, 5.0, 7.0, 6.0, 7.0, 4.0, 1.0, 2.0, 3.0, 116.367),
            (2.0, 1.0, 4.0, 3.0, 6.0, 1.0, 3.0, 2.0, 1.0, 3.0, 116.3668)
        ])
        N = 11
        train_data = data
        test_data = data

    else:
        dataset = np.loadtxt(filepath, delimiter=',', skiprows=1)
        M, N = dataset.shape
        train_X = dataset[:(int)(0.8 * M), :]
        test_X = dataset[(int)(0.8 * M):, :]
        train_data = sc.parallelize(train_X.tolist())
        test_data = sc.parallelize(test_X.tolist())

    columns = ["c" + str(i) for i in range(1, N)]
    columns.append("label")
    df1 = train_data.toDF(columns)
    vecasembler = VectorAssembler(inputCols=columns, outputCol="features")
    traindf = vecasembler.transform(df1).select("features", "label").cache()
    df2 = test_data.toDF(columns)
    testdf = vecasembler.transform(df2).select("features", "label").cache()
    xgbRf0 = XGBRegressor()
    xgbRf0.setNthread(1)
    xgbRf0.setNumRound(10)
    xgbmodel = XGBRegressorModel(xgbRf0.fit(traindf))
    y0 = xgbmodel.transform(testdf)
    y0.show()
    sc.stop()


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-f", "--file-path", type=str, dest="data_path",
                      default=".", help="Path where data are stored")
    parser.add_option("-d", "--demo", action="store_true", dest="demo", default=False)
    parser.add_option("-m", "--master", type=str, dest="masterchoice")
    (option, args) = parser.parse_args(sys.argv)

    if option.data_path is None:
        errno("data path is not specified")
    datapath = option.data_path
    Processdata(datapath, option.demo)
