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


def demoexample():
    sparkConf = init_spark_conf().setMaster("local[1]").setAppName("testNNClassifer")
    sc = init_nncontext(sparkConf)
    sqlContext = SQLContext(sc)
    data = sc.parallelize([
        (1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 4.0, 8.0, 3.0, 116.3668),
        (1.0, 3.0, 8.0, 6.0, 5.0, 9.0, 5.0, 6.0, 7.0, 4.0, 116.367),
        (2.0, 1.0, 5.0, 7.0, 6.0, 7.0, 4.0, 1.0, 2.0, 3.0, 116.367),
        (2.0, 1.0, 4.0, 3.0, 6.0, 1.0, 3.0, 2.0, 1.0, 3.0, 116.3668)
    ])
    columns = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "label"]
    df = data.toDF(columns)
    vecasembler = VectorAssembler(inputCols=columns, outputCol="features")
    assembledf = vecasembler.transform(df).select("features", "label").cache()
    xgbRf0 = XGBRegressor()
    xgbRf0.setNthread(1)
    xgbmodel = XGBRegressorModel(xgbRf0.fit(assembledf))
    xgbmodel.save("/tmp/modelfile/")
    xgbmodel.setFeaturesCol("features")
    yxgb = xgbmodel.transform(assembledf)
    model = xgbmodel.load("/tmp/modelfile/")
    model.setFeaturesCol("features")
    y0 = model.transform(assembledf)


def preProcessdata(filepath):
    '''
    preProcess the data read from filepath
    :param filepath:
    :return: assembledf:
    '''
    dataset = np.loadtxt(filepath, delimiter=',')
    N = dataset.shape[1]
    X = dataset[:, 0: N - 1]
    Y = dataset[:, N - 1]
    sparkConf = init_spark_conf().setMaster("local[1]").setAppName("testNNClassifer")
    sc = init_nncontext(sparkConf)
    sqlContext = SQLContext(sc)

    data = sc.parallelize(dataset.tolist())
    columns = ["c" + str(i) for i in range(1, N)]
    columns.append("label")
    print(columns)
    df = data.toDF(columns)
    vecasembler = VectorAssembler(inputCols=columns, outputCol="features")
    assembledf = vecasembler.transform(df).select("features", "label").cache()
    return assembledf


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-f", "--folder", type=str, dest="data_path", default=".",
                      help="Path where data are stored")
    parser.add_option("-d", action="store_true", dest="demo", default=False)
    (option, args) = parser.parse_args(sys.argv)
    print(option.demo)
    if (option.demo):
        demoexample()
    else:
        if option.data_path is None:
            errno("data path is not specified")
        datapath = option.data_path

        for file in os.listdir(datapath):
            if (os.path.splitext(file)[-1] == ".csv"):
                filepath = os.path.join(option.data_path, file)
                print(filepath)
                assembledf = preProcessdata(filepath)
