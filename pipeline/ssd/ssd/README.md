# SSD: Single Shot MultiBox Detector

[SSD](https://research.google.com/pubs/pub44872.html) is one of the state-of-the-art
 object detection pipeline.

Currently Bigdl has added support for ssd with vgg and alexnet as base model,
with 300x300 or 512x512 input resolution.

## Default Environment
* JDK 1.7.0_79
* Spark 2.10
* Scala 2.11

## Build BigDL analytics jar

1. clone analytics-zoo project
```bash
git clone https://github.com/intel-analytics/analytics-zoo.git
```

2. install image transformer library
```
# mvn install image transformer lib
cd transform/vision
mvn clean install
```

3. build ssd project
cd ${analytics-zoo}/pipeline/ssd
* Linux
```bash
mvn clean package -DskipTests
```
* Mac
```
mvn clean package -DskipTests -P mac
```

## Prepare the dataset

### Loacal images
If you want to have a try in local images, put them in a flat local folder.
Have tested with jpg or png images.

### Prepare labeled dataset for validation and training
1. [Pascal VOC dataset](data/pascal)
2. [Coco](data/coco)

### Convert unlabeled image folder to sequence file
```bash
./data/tool/convert_image_folder.sh image_folder output
```

where ```image_folder``` is your image folder, ```output``` is the output folder

### Import unlabeled image folder to Hbase
First put the ```hbase-site.xml``` to the folder ```spark-XXX/conf/```

the execute
```bash
./data/tool/create_hbase_hive.sh
```
please adjust the arguments if necessary

## Download pretrained model for evaluation

* [SSD 300x300 Vgg Pascal VOC](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/ssd/bigdl_ssd_vgg_300x300_voc.model)
* [SSD 512x512 Vgg Pascal VOC](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/ssd/bigdl_ssd_vgg_512x512_voc.model)
* [SSD 300x300 Vgg Coco](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/ssd/bigdl_ssd_vgg_300x300_coco.model)
* [SSD 512x512 Vgg Coco](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/ssd/bigdl_ssd_vgg_512x512_coco.model)
* [SSD 300x300 MobileNet Pascal VOC](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/ssd/bigdl_ssd_mobilenet_300x300_voc.model)

## Run the notebook

1. Get BigDL jar
* Linux
```bash
wget https://oss.sonatype.org/content/groups/public/com/intel/analytics/bigdl/dist-spark-2.1.1-scala-2.11.8-linux64/0.3.0-SNAPSHOT/dist-spark-2.1.1-scala-2.11.8-linux64-0.3.0-20171011.104355-69-dist.zip
unzip dist-spark-2.1.1-scala-2.11.8-linux64-0.3.0-20171011.104355-69-dist.zip -d ~/BigDL
```
* Mac
```
wget https://oss.sonatype.org/content/groups/public/com/intel/analytics/bigdl/dist-spark-2.1.1-scala-2.11.8-mac/0.3.0-SNAPSHOT/dist-spark-2.1.1-scala-2.11.8-mac-0.3.0-20171012.213708-67-dist.zip
unzip dist-spark-2.1.1-scala-2.11.8-mac-0.3.0-20171012.213708-67-dist.zip -d ~/BigDL
```
2. Get SSD jar
Download ssd model(vgg+pascal 300x300) from https://drive.google.com/uc?id=0B7vkTZblCs9hM3BmUUJ5Y2dfekk&export=download ,
then put the model at data/models

3. Start notebook

```bash
sh data/tool/start_notebook.sh
```
Please configure the ```SPARK_HOME``` and ```BigDL_HOME``` as necessary.


## Run the predict example
We assume that pretrained caffe models are stored in ```data_root=${ssd_root}/data/models```
Example command for running in Spark cluster (yarn)

```bash
spark-submit \
--master yarn \
--deploy-mode client \
--executor-cores 28 \
--num-executors 2 \
--driver-memory 30g \
--executor-memory 128g \
--class com.intel.analytics.zoo.pipeline.ssd.example.Predict \
pipeline-0.1-SNAPSHOT-jar-with-dependencies.jar \
-f $imageDataFolder \
--folderType seq \
-o output \
--caffeDefPath data/models/VGGNet/VOC0712/SSD_300x300/test.prototxt \
--caffeModelPath data/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel \
-t vgg16 \
--classname data/pascal/classname.txt \
-v false \
-s true \
-p 112 \
-b 112 \
-r 300
```

The output result is save to text file with the following format:

```
ImageName classId score xmin ymin xmax ymax
```

In the above commands

* -f: where you put your image data
* --folderType: It can be seq/local
* -o: where you put your image output data
* --caffeDefPath: caffe network definition prototxt file path
* --caffeModelPath: caffe serialized model file path
* -t: network type, it can be vgg16 or alexnet
* --classname: file that store detection class names, one line for one class
* -s: Whether to save detection results
* -v: whether to visualize detections
* -p: partition number
* -b: batch size, it should be n*(partition number)
* -r: input resolution, 300 or 512

## Run the test example

```bash
spark-submit \
--master yarn \
--deploy-mode client \
--executor-cores 28 \
--num-executors 2 \
--driver-memory 20g \
--executor-memory 128g \
--class com.intel.analytics.zoo.pipeline.ssd.example.Test \
pipeline-0.1-SNAPSHOT-jar-with-dependencies.jar \
-f $voc_test_data \
--caffeDefPath data/models/VGGNet/VOC0712/SSD_300x300/test.prototxt \
--caffeModelPath data/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel \
-t vgg16 \
--nclass 21 \
-i voc_2007_test \
-p 112 \
-b 112 \
-r 300
```

In the above commands

* -f: where you put your image data
* --caffeDefPath: caffe network definition prototxt file path
* --caffeModelPath: caffe serialized model file path
* -t: network type, it can be vgg16 or alexnet
* --nclass: number of detection classes.
* -i: image set name with the format ```voc_${year}_${imageset}```, e.g. voc_2007_test
* -p: partition number
* -b: batch size, it should be n*(partition number)
* -r: input resolution, 300 or 512

## Run the training example
1. Get model pretrained in imagenet
```bash
./data/models/get_models.sh
```

2. Submit spark job
```bash
spark-submit \
--master yarn \
--deploy-mode client \
--executor-cores 28 \
--num-executors 4 \
--driver-memory 20g \
--executor-memory 128g \
--class com.intel.analytics.zoo.pipeline.ssd.example.Train \
pipeline-0.1-SNAPSHOT-jar-with-dependencies.jar \
-f hdfs://xxxx/your_train_folder \
-v hdfs://xxxx/your_val_folder \
-t vgg16 \
-r 300 \
--caffeDefPath ../models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced_deploy.prototxt.txt \
--caffeModelPath ../models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel \
--warm 0.5 \
--schedule plateau \
-e 250 \
-l 0.0035 \
-b 112 \
-d 0.5 \
--patience 10 \
--classNum 21 \
--checkpoint hdfs://xxx/checkpoint/
```
In the above commands

* -f: where you put the training data
* -v: where you put the validation data
* -t: network type, it can be vgg16
* -r: input resolution
* --caffeDefPath: caffe prototxt file
* --caffeModelPath: pretrained caffe model file
* --warm warm up MAP
* --schedule: learning rate schedule type, can be multistep | plateau
* -e: max epoch
* -l: inital learning rate
* -b: batch size
* -d: learning rate decay
* --patience: epoch to wait when the map does not go up
* --classNum: class number
* --checkpoint: where to save your checkpoint model
