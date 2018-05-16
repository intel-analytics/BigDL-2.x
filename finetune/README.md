# Summary

In the image transfer learning example, we use a pre-trained Inception_V1 model as
image feature transformer and train another linear classifier to solve the dogs-vs-cats
classification problem.

In this example we are going to take a slightly different approach. We will still use a pre-trained
 Inception_V1 model, but this time we will operate on the pre-trained model to freeze first of
a few layers, replace the classifier on the top, then fine tune the whole model

# Preparation

## Get the dogs-vs-cats datasets

Download the training dataset from https://www.kaggle.com/c/dogs-vs-cats and extract it.
The following commands copy about 1100 images of cats and dogs into demo/cats and demo/dogs separately.

```
mkdir -p demo/dogs
mkdir -p demo/cats
cp train/cat.7* demo/cats
cp train/dog.7* demo/dogs
```

## Get the pre-trained Inception-V1 model

Download the pre-trained Inception-V1 model from [Zoo](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/imagenet/bigdl_inception-v1_imagenet_0.4.0.model)

Alternatively, user may also download pre-trained caffe/Tensorflow/keras model. Please refer to
programming guide in [BigDL](https://bigdl-project.github.io/) 

# Training for dogs/cats classification

ImageTransferLearningExample.py takes 2 parameters:
1. Path to the pre-trained models. (E.g. path/to/model/bigdl_inception-v1_imagenet_0.4.0.model)
2. Path to the folder of the training images. (E.g. path/to/data/dogs-vs-cats/demo)

User may submit transfer_learning.py via spark-submit.
E.g.
```
export ZOO_HOME=~/sources/analytics-zoo/dist
${ZOO_HOME}/bin/spark-submit-with-zoo.sh --master local[4] --driver-memory 10g \
pyzoo/zoo/examples/nnframes/finetune/transfer_learning.py \
/tmp/models/bigdl_inception-v1_imagenet_0.4.0.model /tmp/datasets/cat_dog/demo
```

or run the script in Jupyter notebook or Pyspark and manually set parameters

After training, you should see something like this in the console:

```
+--------------------+------------+-----+----------+
|               image|        name|label|prediction|
+--------------------+------------+-----+----------+
|[file:/tmp/datase...|cat.7005.jpg|  1.0|       1.0|
|[file:/tmp/datase...|cat.7018.jpg|  1.0|       1.0|
|[file:/tmp/datase...| cat.702.jpg|  1.0|       1.0|
|[file:/tmp/datase...|cat.7020.jpg|  1.0|       1.0|
|[file:/tmp/datase...|cat.7027.jpg|  1.0|       1.0|
|[file:/tmp/datase...|cat.7034.jpg|  1.0|       1.0|
|[file:/tmp/datase...|cat.7039.jpg|  1.0|       1.0|
|[file:/tmp/datase...|cat.7045.jpg|  1.0|       1.0|
|[file:/tmp/datase...|cat.7064.jpg|  1.0|       1.0|
|[file:/tmp/datase...|cat.7066.jpg|  1.0|       1.0|
|[file:/tmp/datase...|cat.7069.jpg|  1.0|       1.0|
|[file:/tmp/datase...|cat.7077.jpg|  1.0|       1.0|
|[file:/tmp/datase...|cat.7083.jpg|  1.0|       1.0|
|[file:/tmp/datase...|cat.7085.jpg|  1.0|       1.0|
|[file:/tmp/datase...|cat.7103.jpg|  1.0|       1.0|
|[file:/tmp/datase...|cat.7114.jpg|  1.0|       1.0|
|[file:/tmp/datase...|cat.7115.jpg|  1.0|       1.0|
|[file:/tmp/datase...|cat.7124.jpg|  1.0|       1.0|
|[file:/tmp/datase...|cat.7134.jpg|  1.0|       1.0|
|[file:/tmp/datase...|cat.7136.jpg|  1.0|       1.0|
+--------------------+------------+-----+----------+
only showing top 20 rows

Test Error = 0.023166

```