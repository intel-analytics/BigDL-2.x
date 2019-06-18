# ResNet
This example demonstrates how to use Analytics Zoo to train and evaluate the [ResNet](https://arxiv.org/abs/1512.03385) architecture on ImageNet data

## Data Processing
We use pipeline to process the input data.
Input data are transformed by several pipeline classes, such as HFlip, BGRImgNormalizer, RandomCropper, etc.

## Model
ShortcutType is a unique feature defined in ResNet. ShortcutType-B is used for ImageNet.
Model is implemented in <code>ResNet</code>

## Get the JAR
You can build one by refer to the
[Build Page](https://analytics-zoo.github.io/master/#ScalaUserGuide/install/#build-with-script-recommended) from the source code.

## Train ResNet on ImageNet
This example shows the best practise we've experimented in multi-node training
### Prepare ImageNet DataSet
The imagenet dataset preparation can be found from
[BigDL inception Prepare the data](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/inception#prepare-the-data).
### Training
* Spark standalone example command

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master spark://xxx.xxx.xxx.xxx:xxxx \
--executor-cores cores_per_executor \
--total-executor-cores total_cores_for_the_job \
--class com.intel.analytics.zoo.examples.resnet.TrainImageNet \
-f hdfs://xxx.xxx.xxx.xxx:xxxx/imagenet \
--batchSize 8192 --nEpochs 90 --learningRate 0.1 --warmupEpoch 5 \
 --maxLr 3.2 --cache /cache  --depth 50 --classes 1000
```
### Parameters
```
    --folder | -f   [the directory to reach the data]
    --batchSize     [default 8192, should be n*nodeNumber*coreNumber]
    --nEpochs       [number of epochs to train]
    --learningRate  [default 0.1]
    --warmupEpoch [warm up epochs]
    --maxLr [max learning rate, default to 3.2]
    --cache [directory to store snapshot]
    --depth         [number of layers for resnet, default to 50]
    --classes       [number of classes, default to 1000]
```
### Training reference
#### Hyper Parameters

**Global batch** : 8192

**Single batch per core** : 4

**Epochs** : 90

**Initial learning rate**: 0.1

**Warmup epochs**: 5

**Max learning rate**: 3.2

#### Training result (90 epochs)

**Top1 accuracy**: 0.76114

**Top5 accuracy**: 0.92724


