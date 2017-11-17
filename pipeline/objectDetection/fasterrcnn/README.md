# Faster-RCNN

Faster-RCNN is a popular object detection framework, which is described in 
[paper](https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf) published in NIPS 2015.
It's official code can be found [here](https://github.com/rbgirshick/py-faster-rcnn) 
and a python version can be found [here](https://github.com/SeaOfOcean/py-faster-rcnn).

Later, [PVANET](https://arxiv.org/abs/1611.08588) further reduces computational cost with a lighter network.
It's implementation can be found [here](https://github.com/sanghoon/pva-faster-rcnn)

This example demonstrates how to use BigDL to test a Faster-RCNN framework with either pvanet network or vgg16 network.

## Prepare the dataset
Download the test dataset

```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
```

Extract all of these tars into one directory named ```VOCdevkit```

```
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
```

It should have this basic structure

```
$VOCdevkit/                           # development kit
$VOCdevkit/VOCcode/                   # VOC utility code
$VOCdevkit/VOC2007                    # image sets, annotations, etc.
# ... and several other directories ...
```

## Generate Sequence Files

Get the bigdl.sh script 
```
wget https://raw.githubusercontent.com/intel-analytics/BigDL/master/scripts/bigdl.sh
source bigdl.sh
```

### convert labeled pascal voc dataset

```bash
dist/bin/bigdl.sh --
java -cp pipeline-0.1-SNAPSHOT-jar-with-dependencies.jar:spark-assembly-1.5.1-hadoop2.6.0.jar \
         com.intel.analytics.bigdl.pipeline.common.dataset.RoiImageSeqGenerator \
     -f $VOCdevkit -o output -i voc_2007_test
```

where ```-f``` is your devkit folder, ```-o``` is the output folder, and ```-i``` is the imageset name.

note that a spark jar is needed as dependency.

### convert unlabeled image folder
```bash
dist/bin/bigdl.sh --
java -cp pipeline-0.1-SNAPSHOT-jar-with-dependencies.jar:spark-assembly-1.5.1-hadoop2.6.0.jar \
         com.intel.analytics.bigdl.pipeline.common.dataset.RoiImageSeqGenerator \
     -f imageFolder -o output
```

where ```-f``` is your image folder, ```-o``` is the output folder

## Download pretrained model

You can use [this script](https://github.com/rbgirshick/py-faster-rcnn/blob/master/data/scripts/fetch_faster_rcnn_models.sh) to download
pretrained Faster-RCNN(VGG) models.

Faster-RCNN(PVANET) model can be found [here](https://www.dropbox.com/s/87zu4y6cvgeu8vs/test.model?dl=0), 
its caffe prototxt file can be found [here](https://github.com/sanghoon/pva-faster-rcnn/blob/master/models/pvanet_obsolete/full/test.pt)

## Prepare a file contain list of class name
Save the follow content to classname.txt
```
__background__
aeroplane
bicycle
bird
boat
bottle
bus
car
cat
chair
cow
diningtable
dog
horse
motorbike
person
pottedplant
sheep
sofa
train
tvmonitor
```

## Run the predict example
Example command for running in Spark cluster (yarn)

```
spark-submit \
--master yarn \
--deploy-mode client \
--executor-cores 28 \
--num-executors 2 \
--driver-memory 128g \
--executor-memory 128g \
--class com.intel.analytics.bigdl.pipeline.fasterrcnn.example.Predict \
pipeline-0.1-SNAPSHOT-jar-with-dependencies.jar \
-f $imageDataFolder \
--folderType seq \
-o output \
--caffeDefPath data/pvanet/full/test.pt \
--caffeModelPath data/pvanet/full/test.model \
-t pvanet  \
--classname data/models/VGGNet/VOC0712/classname.txt \
-v false
```

In the above commands

* -f: where you put your image data
* --folderType: It can be seq/local
* -o: where you put your image output data
* --caffeDefPath: caffe network definition prototxt file path
* --caffeModelPath: caffe serialized model file path
* -t: network type, it can be vgg16 or alexnet
* --classname: file that store detection class names, one line for one class
* -v: whether to visualize detections

## Run the test example

```
spark-submit \
--master yarn \
--deploy-mode client \
--executor-cores 28 \
--num-executors 2 \
--driver-memory 128g \
--executor-memory 128g \
--class com.intel.analytics.bigdl.pipeline.fasterrcnn.example.Test \
pipeline-0.1-SNAPSHOT-jar-with-dependencies.jar \
-f $voc_test_data \
--caffeDefPath data/pvanet/full/test.pt \
--caffeModelPath data/pvanet/full/test.model \
-t vgg16 \
--nclass 21 \
-i voc_2007_test
```

In the above commands

* -f: where you put your image data
* --caffeDefPath: caffe network definition prototxt file path
* --caffeModelPath: caffe serialized model file path
* -t: network type, it can be vgg16 or alexnet
* --nclass: number of detection classes.
* -i: image set name with the format ```voc_${year}_${imageset}```, e.g. voc_2007_test





