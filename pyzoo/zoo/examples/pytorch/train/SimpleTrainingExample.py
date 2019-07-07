import torch.nn as nn
import torch.nn.functional as F
from bigdl.nn.criterion import *
from bigdl.nn.layer import *
from bigdl.optim.optimizer import Adam
from pyspark.sql.types import *
from zoo import init_nncontext
from zoo.common.nncontext import *
from zoo.pipeline.api.net.torch_net import TorchNet, TorchIdentityCriterion
from zoo.pipeline.nnframes import *


# create training data as Spark DataFrame
def get_df(sqlContext):
    data = sc.parallelize([
        ((2.0, 1.0), (1.0, 0.0, 3.0)),
        ((1.0, 2.0), (1.0, 2.0, 3.0)),
        ((2.0, 1.0), (1.0, 0.0, 3.0)),
        ((1.0, 2.0), (1.0, 2.0, 3.0)),
        ((2.0, 1.0), (1.0, 0.0, 3.0)),
        ((1.0, 2.0), (1.0, 2.0, 3.0)),
        ((2.0, 1.0), (1.0, 0.0, 3.0)),
        ((1.0, 2.0), (1.0, 2.0, 3.0))
    ])

    schema = StructType([
        StructField("features", ArrayType(DoubleType(), False), False),
        StructField("label", ArrayType(DoubleType(), False), False)])
    df = sqlContext.createDataFrame(data, schema)
    return df

# define model with Pytorch
class SimpleTorchModel(nn.Module):
    def __init__(self):
        super(SimpleTorchModel, self).__init__()
        self.dense1 = nn.Linear(2, 4)
        self.dense2 = nn.Linear(4, 8)
        self.dense3 = nn.Linear(8, 3)

    def forward(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = F.relu(self.dense3(x))
        return x

if __name__ == '__main__':
    sparkConf = init_spark_conf().setAppName("testNNClassifer").setMaster('local[1]')
    sc = init_nncontext(sparkConf)
    sqlContext = SQLContext(sc)
    df = get_df(sqlContext)

    torch_model = SimpleTorchModel()
    def customLoss(output, label):
        return ((output - label) * (output - label)).sum()

    model = TorchNet.from_pytorch(torch_model, [1, 2], customLoss, [1, 1], [1, 1])
    classifier = NNEstimator(model, TorchIdentityCriterion(), SeqToTensor([2]), SeqToTensor([3])) \
        .setBatchSize(4) \
        .setOptimMethod(Adam()) \
        .setLearningRate(0.1).setMaxEpoch(10)

    nnClassifierModel = classifier.fit(df)
    res = nnClassifierModel.transform(df)
    res.show(10, False)

