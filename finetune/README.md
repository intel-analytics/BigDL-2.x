# Summary

In the image transfer learning example, we use a pre-trained Inception_V1 model as
image feature transformer and train another linear classifier to solve the dogs-vs-cats
classification problem.

In this example we are going to take a different approach. We will still use a pre-trained
 Inception_V1 model, but this time we will operate on the pre-trained model to freeze first of
a few layers, replace the classifier on the top, then fine tune the whole model

## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/)
to install analytics-zoo via __pip__ or __download the prebuilt package__.

## Image Fine Tuning
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

3. Run the image fine tuning:
image_finetuning_example.py takes 2 parameters: Path to the pre-trained models and 
Path to the images.

- Run after pip install
You can easily use the following commands to run this example:
    ```bash
    export SPARK_DRIVER_MEMORY=10g
    python image_finetuning_example.py /tmp/zoo/bigdl_inception-v1_imagenet_0.4.0.model /tmp/zoo/dogs_cats/samples
    ```
    See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-after-pip-install) for more running guidance after pip install.

- Run with prebuilt package
Run the following command for Spark local mode (`MASTER=local[*]`) or cluster mode:
    ```bash
    export SPARK_HOME=the root directory of Spark
    export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package

    ${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master local[2] \
    --driver-memory 10g \
    image_finetuning_example.py \
    /tmp/zoo/bigdl_inception-v1_imagenet_0.4.0.model /tmp/zoo/dogs_cats/samples
    ```
    See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-without-pip-install) for more running guidance without pip install.

4. see the result
After training, you should see something like this in the console:

```
+--------------------+------------+-----+----------+
|               image|        name|label|prediction|
+--------------------+------------+-----+----------+
|[file:/tmp/zoo/do...|cat.7028.jpg|  1.0|       1.0|
|[file:/tmp/zoo/do...|cat.7208.jpg|  1.0|       1.0|
|[file:/tmp/zoo/do...|cat.7244.jpg|  1.0|       1.0|
|[file:/tmp/zoo/do...|cat.7329.jpg|  1.0|       1.0|
|[file:/tmp/zoo/do...|cat.7434.jpg|  1.0|       1.0|
|[file:/tmp/zoo/do...| cat.744.jpg|  1.0|       1.0|
|[file:/tmp/zoo/do...|cat.7489.jpg|  1.0|       1.0|
|[file:/tmp/zoo/do...|cat.7511.jpg|  1.0|       1.0|
|[file:/tmp/zoo/do...|cat.7653.jpg|  1.0|       1.0|
|[file:/tmp/zoo/do...|cat.7770.jpg|  1.0|       1.0|
|[file:/tmp/zoo/do...|cat.7946.jpg|  1.0|       1.0|
|[file:/tmp/zoo/do...| dog.706.jpg|  2.0|       2.0|
|[file:/tmp/zoo/do...|dog.7239.jpg|  2.0|       2.0|
|[file:/tmp/zoo/do...|dog.7318.jpg|  2.0|       2.0|
|[file:/tmp/zoo/do...|dog.7386.jpg|  2.0|       2.0|
|[file:/tmp/zoo/do...|dog.7412.jpg|  2.0|       2.0|
|[file:/tmp/zoo/do...|dog.7742.jpg|  2.0|       2.0|
|[file:/tmp/zoo/do...|dog.7761.jpg|  2.0|       2.0|
|[file:/tmp/zoo/do...|dog.7842.jpg|  2.0|       2.0|
|[file:/tmp/zoo/do...|dog.7866.jpg|  2.0|       2.0|
+--------------------+------------+-----+----------+
only showing top 20 rows

Test Error = 0.00873362
```
