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
    --master local[2] \
    --driver-memory 10g \
    ImageTransferLearningExample.py \
    /tmp/zoo/bigdl_inception-v1_imagenet_0.4.0.model /tmp/zoo/dogs_cats/samples
    ```
    See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-without-pip-install) for more running guidance without pip install.

4. see the result
After training, you should see something like this in the console:
    
    ```
    +-------------------+-------------+-----+-------------------+--------------------+----------+
    |              image|         name|label|           features|           embedding|prediction|
    +-------------------+-------------+-----+-------------------+--------------------+----------+
    |[file:/some/path...|cat.10007.jpg|  1.0|[file:/some/path...|[1.44402220030315...|       1.0|
    |[file:/some/path...|cat.10008.jpg|  1.0|[file:/some/path...|[2.78127276942541...|       1.0|
    |[file:/some/path...|cat.10013.jpg|  1.0|[file:/some/path...|[1.72082152971597...|       1.0|
    |[file:/some/path...|cat.10017.jpg|  1.0|[file:/some/path...|[1.07376172309159...|       1.0|
    |[file:/some/path...|cat.10020.jpg|  1.0|[file:/some/path...|[6.77592743159038...|       1.0|
    |[file:/some/path...|cat.10021.jpg|  1.0|[file:/some/path...|[1.57088209107314...|       1.0|
    |[file:/some/path...|cat.10024.jpg|  1.0|[file:/some/path...|[2.72918850896530...|       2.0|
    |[file:/some/path...|cat.10048.jpg|  1.0|[file:/some/path...|[6.11712948739295...|       1.0|
    |[file:/some/path...|cat.10068.jpg|  1.0|[file:/some/path...|[8.66246239183965...|       1.0|
    |[file:/some/path...|cat.10069.jpg|  1.0|[file:/some/path...|[3.47972563758958...|       1.0|
    |[file:/some/path...|cat.10076.jpg|  1.0|[file:/some/path...|[1.33044534322834...|       1.0|
    |[file:/some/path...|cat.10081.jpg|  1.0|[file:/some/path...|[6.24413246441690...|       1.0|
    |[file:/some/path...|cat.10103.jpg|  1.0|[file:/some/path...|[4.13055857961808...|       1.0|
    |[file:/some/path...| cat.1011.jpg|  1.0|[file:/some/path...|[1.52658026308927...|       2.0|
    |[file:/some/path...|cat.10111.jpg|  1.0|[file:/some/path...|[9.06654804566642...|       1.0|
    |[file:/some/path...|cat.10113.jpg|  1.0|[file:/some/path...|[6.60018413327634...|       1.0|
    |[file:/some/path...|cat.10125.jpg|  1.0|[file:/some/path...|[1.46317620419722...|       1.0|
    |[file:/some/path...|cat.10131.jpg|  1.0|[file:/some/path...|[6.65911130681706...|       1.0|
    |[file:/some/path...|cat.10154.jpg|  1.0|[file:/some/path...|[3.50153422914445...|       1.0|
    |[file:/some/path...|cat.10155.jpg|  1.0|[file:/some/path...|[1.32401575683616...|       1.0|
    +--------------------+-------------+-----+--------------------+--------------------+----------+
    only showing top 20 rows
    
    Test Error = 0.0754258 
    ```
    With master = local[2]. The transfer learning can finish in 8 minutes. As we can see,
    the model from transfer learning can achieve over 90% accuracy on the validation set.
