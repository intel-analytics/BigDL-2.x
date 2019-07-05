import torch

from zoo.pipeline.api.net.torch_net import TorchNet, TorchIdentityCriterion
from zoo import init_nncontext
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from bigdl.nn.criterion import *
from bigdl.nn.layer import *
from bigdl.optim.optimizer import Adam
from pyspark.sql.types import *
from zoo.common.nncontext import *
from zoo.pipeline.nnframes import *


# create training data as Spark DataFrame
def get_classifier_df(sqlContext):
    data = sc.parallelize([
        ((2.0, 1.0), 1.0),
        ((1.0, 2.0), 0.0),
        ((2.0, 1.0), 1.0),
        ((1.0, 2.0), 0.0),
        ((2.0, 1.0), 1.0),
        ((1.0, 2.0), 0.0),
        ((2.0, 1.0), 1.0),
        ((1.0, 2.0), 0.0)
    ])

    schema = StructType([
        StructField("features", ArrayType(DoubleType(), False), False),
        StructField("label", DoubleType(), False)])
    df = sqlContext.createDataFrame(data, schema)
    return df

# define model with Pytorch
class SimpleTorchModel(nn.Module):
    def __init__(self):
        super(SimpleTorchModel, self).__init__()
        self.dense1 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.dense1(x)
        return x

if __name__ == '__main__':
    sparkConf = init_spark_conf().setAppName("testNNClassifer").setMaster('local[1]')
    sc = init_nncontext(sparkConf)
    sqlContext = SQLContext(sc)
    df = get_classifier_df(sqlContext)

    torch_model = SimpleTorchModel()
    def customLoss(output, label):
        return ((output - label) * (output - label)).sum()

    model = TorchNet.from_pytorch(torch_model, [1, 2], customLoss, [1, 1], [1, 1])
    classifier = NNClassifier(model, TorchIdentityCriterion(), SeqToTensor([2])) \
        .setBatchSize(4) \
        .setOptimMethod(Adam()) \
        .setLearningRate(0.1).setMaxEpoch(40)

    nnClassifierModel = classifier.fit(df)
    res = nnClassifierModel.transform(df)
    res.show()

