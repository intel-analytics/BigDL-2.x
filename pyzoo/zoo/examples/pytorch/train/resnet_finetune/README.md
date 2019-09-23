# Summary

In this example we use a pre-trained ResNet model, adding an extra layer to the end, to train
a dog-vs-cat image classification model.

## Requirements
* Python 3.6
* JDK 1.8
* Pytorch & TorchVision 1.1.0
* Apache Spark 2.4.3(pyspark)
* Analytics-Zoo 0.6.0-SNAPSHOT.dev8 and above
* Jupyter Notebook, matplotlib

## Prepare environments
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the enviroments, especially if you want to run on a yarn cluster(yarn-client mode only). 
```
conda create -n zoo python=3.6 #zoo is conda enviroment name, you can set another name you like.
conda activate zoo
pip install analytics-zoo==0.6.0.dev8 jupyter matplotlib
conda install pytorch-cpu torchvision-cpu -c pytorch
```

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


2. Run the image fine tuning:
resnet_finetune.py takes 1 parameter: Path to the images.

- Run with Spark Local mode
You can easily use the following commands to run this example:
    ```bash
    python resnet_finetune.py /tmp/zoo/dogs_cats/samples
    ```

- Run with Yarn Client mode
If you want to run on a yarn cluster(yarn-client mode only), upload data to hdfs first:
    ```bash
    hdfs dfs -put /tmp/zoo/dogs_cats dogs_cats 
    ```
then, export env `HADOOP_CONF_DIR` and `ZOO_CONDA_NAME`:
    ```bash
    export HADOOP_CONF_DIR=[path to your hadoop conf directory]
    export ZOO_CONDA_NAME=[conda environment name you just prepared above]
    ```
run the following command: 
    ```bash
    python resnet_finetune.py dogs_cats/samples
    ```

3. see the result
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


```
