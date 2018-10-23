## Overview

In the image transfer learning example, we use a pre-trained Inception_V1 model as
image feature transformer and train another linear classifier to solve the dogs-vs-cats
classification problem.

In this example we are going to take a different approach. We will use a pre-trained
Inception_V1 model, but this time we will operate on the pre-trained model to freeze first of
a few layers, replace the classifier on the top, then fine tune the whole model.

## Download Analytics Zoo
You can download Analytics Zoo prebuilt release and nightly build package from [here](https://analytics-zoo.github.io/master/#release-download/) and extract it.

## Run the example

1. Download the pre-trained model

You can download the pre-trained model from
[Analytics Zoo](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/imagenet/bigdl_inception-v1_imagenet_0.4.0.model), and put it in `/tmp/zoo` or other path.

2. Prepare dataset
For this example we use kaggle [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) train
dataset. Download the data and run the following commands to copy about 1100 images of cats
and dogs into `samples` folder.

```bash
unzip train.zip -d /tmp/zoo/dogs_cats
cd /tmp/zoo/dogs_cats
mkdir samples
cp train/cat.7* samples
cp train/dog.7* samples
```
`7` is randomly chosen and can be replaced with other digit.

3. Finetuning the model

Run the following command for Spark local mode (`MASTER=local[*]`) or cluster mode, adjust
 the memory size according to your image:

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
    --master local[2] \
    --driver-memory 10g \
    --class com.intel.analytics.zoo.examples.nnframes.finetune.ImageFinetune \
    --modelPath /tmp/zoo/bigdl_inception-v1_imagenet_0.4.0.model \
    --batchSize 32 \
    --imagePath /tmp/zoo/dogs_cats/samples \
    --nEpochs 2
```

After training, you should see something like this.

```
2018-10-17 14:41:20 INFO  DistriOptimizer$:439 - [Epoch 2 1760/1745][Iteration 110][Wall Clock 285.732906186s] Epoch finished. Wall clock time is 306493.17224 ms
2018-10-17 14:41:20 INFO  DistriOptimizer$:109 - [Epoch 2 1760/1745][Iteration 110][Wall Clock 285.732906186s] Validate model...
2018-10-17 14:41:42 INFO  DistriOptimizer$:152 - [Epoch 2 1760/1745][Iteration 110][Wall Clock 285.732906186s] Top1Accuracy is Accuracy(correct: 471, count: 477, accuracy: 0.9874213836477987)
+-----------------------------------------------------------------------+-----+----------+
|image                                                                  |label|prediction|
+-----------------------------------------------------------------------+-----+----------+
|[file:/tmp/zoo/dogs_cats/samples/cat.7164.jpg,300,300,3,16,[B@207d4b53]|1.0  |1.0       |
|[file:/tmp/zoo/dogs_cats/samples/cat.7200.jpg,300,300,3,16,[B@56f422e9]|1.0  |1.0       |
|[file:/tmp/zoo/dogs_cats/samples/cat.7337.jpg,300,300,3,16,[B@76d40e8e]|1.0  |1.0       |
|[file:/tmp/zoo/dogs_cats/samples/cat.7354.jpg,300,300,3,16,[B@70fbe3dd]|1.0  |1.0       |
|[file:/tmp/zoo/dogs_cats/samples/cat.7436.jpg,300,300,3,16,[B@3a161942]|1.0  |1.0       |
|[file:/tmp/zoo/dogs_cats/samples/cat.7591.jpg,300,300,3,16,[B@4d828783]|1.0  |1.0       |
|[file:/tmp/zoo/dogs_cats/samples/cat.7789.jpg,300,300,3,16,[B@1b7440c3]|1.0  |1.0       |
|[file:/tmp/zoo/dogs_cats/samples/cat.7796.jpg,300,300,3,16,[B@343ebd7b]|1.0  |1.0       |
|[file:/tmp/zoo/dogs_cats/samples/cat.786.jpg,300,300,3,16,[B@3ee86eb7] |1.0  |1.0       |
|[file:/tmp/zoo/dogs_cats/samples/dog.7036.jpg,300,300,3,16,[B@678f4876]|2.0  |2.0       |
|[file:/tmp/zoo/dogs_cats/samples/dog.7218.jpg,300,300,3,16,[B@5ec47e1c]|2.0  |2.0       |
|[file:/tmp/zoo/dogs_cats/samples/dog.7412.jpg,300,300,3,16,[B@1fd0d5da]|2.0  |2.0       |
|[file:/tmp/zoo/dogs_cats/samples/dog.7428.jpg,300,300,3,16,[B@62309d41]|2.0  |2.0       |
|[file:/tmp/zoo/dogs_cats/samples/dog.7522.jpg,300,300,3,16,[B@7f61a589]|2.0  |2.0       |
|[file:/tmp/zoo/dogs_cats/samples/dog.7607.jpg,300,300,3,16,[B@2a810e7] |2.0  |2.0       |
|[file:/tmp/zoo/dogs_cats/samples/dog.7727.jpg,300,300,3,16,[B@1b4f6b6d]|2.0  |2.0       |
|[file:/tmp/zoo/dogs_cats/samples/dog.7915.jpg,300,300,3,16,[B@7d7337d0]|2.0  |2.0       |
|[file:/tmp/zoo/dogs_cats/samples/dog.7962.jpg,300,300,3,16,[B@fcc981a] |2.0  |2.0       |
|[file:/tmp/zoo/dogs_cats/samples/dog.7993.jpg,300,300,3,16,[B@4da18e2c]|2.0  |2.0       |
+-----------------------------------------------------------------------+-----+----------+
```

The model from fine tuning can achieve high accuracy on the validation set.

In this example, we use the Inception-V1 model. Please feel free to explore other models from
Caffe, Keras and Tensorflow. Analytics Zoo provides popular pre-trained model in https://analytics-zoo.github.io/master/#ProgrammingGuide/image-classification/#download-link
