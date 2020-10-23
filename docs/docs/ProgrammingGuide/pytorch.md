Analytics-Zoo supports distributed Pytorch training and inference on on Apache Spark. User can
define their model and loss function with Pytorch API, and run it in a distributed environment
with the wrapper layers provided by Analytics Zoo.

# System Requirement
Pytorch version: 1.5.0 or above  
torchvision: 0.6.0 or above  
cloudpickle: 1.2.2  
jep: 3.9.0  
Python: 3.7  

# Pytorch API

A few wrappers are defined in Analytics Zoo for Pytorch:

1. TorchModel: TorchModel is a wrapper class for Pytorch model.
User may create a TorchModel by providing a Pytorch model, e.g.
    ```python
    from zoo.pipeline.api.torch import TorchModel
    import torchvision
    zoo_model = TorchModel.from_pytorch(torchvision.models.resnet18(pretrained=True))
    ```
The above line creates TorchModel wrapping a ResNet model, and user can use the TorchModel for
training or inference with Analytics Zoo.

2. TorchLoss: TorchLoss is a wrapper for loss functions defined by Pytorch.
User may create a TorchLoss from a Pytorch Criterion, 
    ```python
    import torch
    from zoo.pipeline.api.torch import TorchLoss
    
    az_criterion = TorchLoss.from_pytorch(torch.nn.MSELoss())
    ```
    or from a custom loss function, which takes input and label as parameters

    ```python
    import torch
    from zoo.pipeline.api.torch import TorchLoss
    
    criterion = torch.nn.MSELoss()

    # this loss function is calculating loss for a multi-output model
    def lossFunc(input, label):
        loss1 = criterion(input[0], label[0])
        loss2 = criterion(input[1], label[1])
        loss = loss1 + 0.4 * loss2
        return loss
    
    az_criterion = TorchLoss.from_pytorch(lossFunc)
    ```
    
3. TorchOptim: TorchOptim wraps a torch optimizer for distributed training.
   ```python
   from zoo.pipeline.api.torch import TorchOptim
   import torch
   
   model = torchvision.models.resnet18(pretrained=True))
   adam = torch.optim.Adam(model.parameters())
   zoo_optimizer = TorchOptim.from_pytorch(adam)
   ```

# Examples
Here we provide a simple end to end example, where we use TorchModel and TorchLoss to
train a simple model with Spark DataFrame.
```python
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
import torch
import torch.nn as nn
from zoo.common.nncontext import *
from zoo.pipeline.api.torch import TorchModel, TorchLoss, TorchOptim
from zoo.pipeline.nnframes import *

from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession


# define model with Pytorch
class SimpleTorchModel(nn.Module):
    def __init__(self):
        super(SimpleTorchModel, self).__init__()
        self.dense1 = nn.Linear(2, 4)
        self.dense2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.dense1(x)
        x = torch.sigmoid(self.dense2(x))
        return x

if __name__ == '__main__':
    sc = init_spark_on_local(cores=1)
    spark = SparkSession \
        .builder \
        .getOrCreate()

    df = spark.createDataFrame(
        [(Vectors.dense([2.0, 1.0]), 1.0),
         (Vectors.dense([1.0, 2.0]), 0.0),
         (Vectors.dense([2.0, 1.0]), 1.0),
         (Vectors.dense([1.0, 2.0]), 0.0)],
        ["features", "label"])

    torch_model = SimpleTorchModel()
    torch_criterion = nn.MSELoss()
    torch_optimizer = torch.optim.Adam(torch_model.parameters())

    az_model = TorchModel.from_pytorch(torch_model)
    az_criterion = TorchLoss.from_pytorch(torch_criterion)
    az_optimizer = TorchOptim.from_pytorch(torch_optimizer)

    classifier = NNClassifier(az_model, az_criterion) \
        .setBatchSize(4) \
        .setOptimMethod(az_optimizer) \
        .setLearningRate(0.01) \
        .setMaxEpoch(10)

    nnClassifierModel = classifier.fit(df)

    print("After training: ")
    res = nnClassifierModel.transform(df)
    res.show(10, False)

```
You can simply use `python` to execute the script above. We expects to see the output like:
```python
+---------+-----+----------+
|features |label|prediction|
+---------+-----+----------+
|[2.0,1.0]|1.0  |1.0       |
|[1.0,2.0]|0.0  |0.0       |
|[2.0,1.0]|1.0  |1.0       |
|[1.0,2.0]|0.0  |0.0       |
+---------+-----+----------+
```

# FAQ
1. Does analytics-zoo's distributed pytorch support training or inference?  
Analytics-Zoo support both training and inference.

2. How to prepare the environment?  
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the enviroments, especially if you want to run on a yarn cluster(yarn-client mode only). 
```
conda create -n zoo python=3.7 #zoo is conda enviroment name, you can set another name you like.
conda activate zoo
pip install analytics-zoo[torch]
```
Note that the extra dependencies (including BigDL, torch, torchvision, jep, cloudpickle, conda-pack) will be installed by specifying [torch].  

3. How to determine how many resources do you use in analytics-zoo's distributed mode?  


4. Supported torch and torchvision version? 1.5 or 1.6  

5. How to migrate training from pytorch to AZ?  

6. How to distribute inference with pytorch model?  

