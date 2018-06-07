# Overview

In the image transfer learning example, we use a pre-trained Inception_V1 model as
image feature transformer and train another linear classifier to solve the dogs-vs-cats
classification problem.

In this example we are going to take a slightly different approach. We will still use a pre-trained
caffe Inception_V1 model, but this time we will operate on the pre-trained model to freeze first of
a few layers, replace the classifier on the top, then fine tune the whole model

# Preparation

1. Download the pre-trained model

You can download the pre-trained BigDL model [here](https://github.com/intel-analytics/analytics-zoo/tree/legacy/models).

In this example, we are going to use the Inception-V1 model. Please feel free to use other models.

2. Prepare the dataset

For this example we use kaggle [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) train dataset.
After you download the file (train.zip), run the follow commands to extract the data.

This dataset contains 25000 labeled images. In our experimentation, however, only a few thousands images
could train a pretty good classifier. So you could use the `sample_training_data.py` script to sample a
smaller dataset, and this will greatly speed up your experimentation.


# Finetuning the model

If you use the script mentioned above to downsize the dataset, it is suffice to use spark-local
mode to run this example.
```
    spark-submit \
    --master local[physcial_core_number] \
    --driver-memory 10g --executor-memory 20g \
    --class com.intel.analytics.zoo.examples.nnframes.finetune.TransferLearning \
    ./dist/lib/analytics-zoo-VERSION-jar-with-dependencies.jar \
    --modelPath /tmp/bigdl_inception-v1_imagenet_0.4.0.model \
    --dataPath /tmp/train_sampled \
    --batchSize 32 \
    --nEpochs 2
```

After training, you should see something like this.

As you can see, we can get 98.3% accuracy only after the 2nd epoch. 

```
2018-05-04 17:00:07 INFO  DistriOptimizer$:436 - [Epoch 2 1968/1964][Iteration 246][Wall Clock 213.599345954s] Epoch finished. Wall clock time is 270805.62788 ms
2018-05-04 17:00:07 INFO  DistriOptimizer$:702 - [Epoch 2 1968/1964][Iteration 246][Wall Clock 213.599345954s] Validate model...
2018-05-04 17:00:21 INFO  DistriOptimizer$:744 - [Epoch 2 1968/1964][Iteration 246][Wall Clock 213.599345954s] Top1Accuracy is Accuracy(correct: 529, count: 538, accuracy: 0.983271375464684)
+------------------------------------------------------------------------------------+-----+----------+
|image                                                                               |label|prediction|
+------------------------------------------------------------------------------------+-----+----------+
|[file:/home/yang/sources/model/train_sampled/cat.10025.jpg,251,153,3,16,[B@6135bdd3]|1.0  |1.0       |
|[file:/home/yang/sources/model/train_sampled/cat.10025.jpg,251,153,3,16,[B@23b93a55]|1.0  |1.0       |
|[file:/home/yang/sources/model/train_sampled/cat.10151.jpg,375,499,3,16,[B@7da766e1]|1.0  |1.0       |
|[file:/home/yang/sources/model/train_sampled/cat.10194.jpg,464,499,3,16,[B@443b1e8c]|1.0  |1.0       |
|[file:/home/yang/sources/model/train_sampled/cat.10194.jpg,464,499,3,16,[B@82cf8fd] |1.0  |1.0       |
|[file:/home/yang/sources/model/train_sampled/cat.10206.jpg,499,258,3,16,[B@62ad003f]|1.0  |1.0       |
|[file:/home/yang/sources/model/train_sampled/cat.10206.jpg,499,258,3,16,[B@247be2e5]|1.0  |1.0       |
|[file:/home/yang/sources/model/train_sampled/cat.10412.jpg,381,360,3,16,[B@38a84798]|1.0  |1.0       |
|[file:/home/yang/sources/model/train_sampled/cat.1049.jpg,374,500,3,16,[B@572cab60] |1.0  |1.0       |
|[file:/home/yang/sources/model/train_sampled/cat.11777.jpg,375,499,3,16,[B@5d0ae542]|1.0  |1.0       |
|[file:/home/yang/sources/model/train_sampled/cat.11853.jpg,305,500,3,16,[B@261d48cc]|1.0  |1.0       |
|[file:/home/yang/sources/model/train_sampled/cat.12282.jpg,228,320,3,16,[B@5c735f10]|1.0  |1.0       |
|[file:/home/yang/sources/model/train_sampled/cat.1303.jpg,374,500,3,16,[B@6eb69855] |1.0  |1.0       |
|[file:/home/yang/sources/model/train_sampled/cat.1422.jpg,494,427,3,16,[B@649c4034] |1.0  |1.0       |
|[file:/home/yang/sources/model/train_sampled/cat.1637.jpg,374,500,3,16,[B@280b3f43] |1.0  |1.0       |
|[file:/home/yang/sources/model/train_sampled/cat.1637.jpg,374,500,3,16,[B@6ec67e8d] |1.0  |1.0       |
|[file:/home/yang/sources/model/train_sampled/cat.1644.jpg,299,400,3,16,[B@34682730] |1.0  |1.0       |
|[file:/home/yang/sources/model/train_sampled/cat.2436.jpg,375,499,3,16,[B@32735c69] |1.0  |1.0       |
|[file:/home/yang/sources/model/train_sampled/cat.2576.jpg,179,186,3,16,[B@43739c43] |1.0  |1.0       |
|[file:/home/yang/sources/model/train_sampled/cat.2741.jpg,385,352,3,16,[B@1e9cb0b6] |1.0  |1.0       |
+------------------------------------------------------------------------------------+-----+----------+
```
