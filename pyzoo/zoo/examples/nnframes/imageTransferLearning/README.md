## Summary

Python demo of transfer Learning based on Spark DataFrame (Dataset). 

Analytics Zoo provides the DataFrame-based API for image reading, preprocessing, model training
and inference. The related classes followed the typical estimator/transformer pattern of Spark ML
and can be used in a standard Spark ML pipeline.

In this example, we will show you how to use a pre-trained inception-v1 model trained on
imagenet dataset to solve the dogs-vs-cats classification problem by transfer learning with
Analytics Zoo. For transfer learning, we will treat the inception-v1 model as a feature extractor
and only train a linear model on these features.

## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/)
to install analytics-zoo via __pip__ or __download the prebuilt package__.

## Image Transfer Learning
1. For this example we use kaggle [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) train
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

2. Get the pre-trained Inception-V1 model
Download the pre-trained Inception-V1 model from [Analytics Zoo](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/imagenet/bigdl_inception-v1_imagenet_0.4.0.model),
and put it in `/tmp/zoo` or other path.

3. Run the image transfer learning:
ImageTransferLearningExample.py takes 2 parameters: Path to the pre-trained models and 
Path to the images.

- Run after pip install
You can easily use the following commands to run this example:
    ```bash
    export SPARK_DRIVER_MEMORY=5g
    python ImageTransferLearningExample.py /tmp/zoo/bigdl_inception-v1_imagenet_0.4.0.model /tmp/zoo/dogs_cats/samples
    ```
    See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-after-pip-install) for more running guidance after pip install.

- Run with prebuilt package
Run the following command for Spark local mode (`MASTER=local[*]`) or cluster mode:
    ```bash
    export SPARK_HOME=the root directory of Spark
    export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package

    ${ANALYTICS_ZOO_HOME}/bin/spark-submit-with-zoo.sh \
    --master local[1] \
    --driver-memory 5g \
    ImageTransferLearningExample.py \
    /tmp/zoo/bigdl_inception-v1_imagenet_0.4.0.model /tmp/zoo/dogs_cats/samples
    ```
    See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-without-pip-install) for more running guidance without pip install.

4. see the result
After training, you should see something like this in the console:
    
    ```
    +--------------------+------------+-----+--------------------+----------+
    |               image|        name|label|           embedding|prediction|
    +--------------------+------------+-----+--------------------+----------+
    |[file:/tmp/zoo/do...|   cat.7.jpg|  1.0|[1.4909998E-6, 2....|       1.0|
    |[file:/tmp/zoo/do...|cat.7007.jpg|  1.0|[7.552656E-6, 1.6...|       1.0|
    |[file:/tmp/zoo/do...|cat.7040.jpg|  1.0|[2.0968411E-5, 1....|       1.0|
    |[file:/tmp/zoo/do...|cat.7116.jpg|  1.0|[4.3927703E-6, 1....|       1.0|
    |[file:/tmp/zoo/do...|cat.7225.jpg|  1.0|[3.3697938E-7, 1....|       1.0|
    |[file:/tmp/zoo/do...| cat.733.jpg|  1.0|[5.439134E-5, 1.9...|       1.0|
    |[file:/tmp/zoo/do...|cat.7368.jpg|  1.0|[7.5563776E-7, 6....|       1.0|
    |[file:/tmp/zoo/do...|cat.7373.jpg|  1.0|[7.449715E-6, 1.7...|       1.0|
    |[file:/tmp/zoo/do...|cat.7542.jpg|  1.0|[6.415686E-5, 5.7...|       1.0|
    |[file:/tmp/zoo/do...|cat.7580.jpg|  1.0|[4.4665518E-5, 1....|       1.0|
    |[file:/tmp/zoo/do...|cat.7583.jpg|  1.0|[4.3137575E-6, 2....|       1.0|
    |[file:/tmp/zoo/do...|cat.7646.jpg|  1.0|[7.990455E-6, 1.0...|       1.0|
    |[file:/tmp/zoo/do...|cat.7693.jpg|  1.0|[6.299197E-6, 1.4...|       1.0|
    |[file:/tmp/zoo/do...|cat.7727.jpg|  1.0|[1.1037457E-5, 7....|       1.0|
    |[file:/tmp/zoo/do...|cat.7764.jpg|  1.0|[5.5489426E-9, 8....|       1.0|
    |[file:/tmp/zoo/do...|cat.7775.jpg|  1.0|[5.389813E-5, 7.4...|       1.0|
    |[file:/tmp/zoo/do...|cat.7816.jpg|  1.0|[1.247128E-5, 1.8...|       1.0|
    |[file:/tmp/zoo/do...| cat.783.jpg|  1.0|[1.4708958E-5, 1....|       1.0|
    |[file:/tmp/zoo/do...|cat.7960.jpg|  1.0|[8.348782E-7, 3.2...|       1.0|
    |[file:/tmp/zoo/do...|dog.7202.jpg|  2.0|[3.695763E-5, 1.5...|       2.0|
    +--------------------+------------+-----+--------------------+----------+
    only showing top 20 rows
    
    Test Error = 0.0298507 
    ```
    With master = local[1]. The transfer learning can finish in 8 minutes. As we can see,
    the model from transfer learning can achieve high accuracy on the validation set.
