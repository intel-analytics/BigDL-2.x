# SSD: Single Shot MultiBox Detector

[SSD](https://research.google.com/pubs/pub44872.html) is one of the state-of-the-art
 object detection pipeline.

Currently Bigdl has added support for ssd with vgg as base model,
with 300*300 input resolution.

## Prepare the dataset


## Generate Sequence Files

Get the bigdl.sh script 
```
wget https://raw.githubusercontent.com/intel-analytics/BigDL/master/scripts/bigdl.sh
source bigdl.sh
```

### get coco dataset
Download Images and Annotations from [MSCOCO](http://mscoco.org/dataset/#download).

### convert unlabeled image folder
```bash
dist/bin/bigdl.sh --
java -cp pipeline-0.1-SNAPSHOT-jar-with-dependencies.jar:spark-assembly-1.5.1-hadoop2.6.0.jar \
         com.intel.analytics.zoo.pipeline.common.dataset.RoiImageSeqGenerator \
     -f imageFolder -o output
```

where ```-f``` is your image folder, ```-o``` is the output folder

## Pretrained model

Provided upon request

## Run the predict example
Example command for running in Spark cluster (yarn)

```
spark-submit \
--master yarn \
--deploy-mode client \
--executor-cores 2 \
--num-executors 2 \
--driver-memory 10g \
--executor-memory 30g \
--class com.intel.analytics.zoo.pipeline.ssd.example.Predict \
pipeline-0.1-SNAPSHOT-jar-with-dependencies.jar \
-f $imageDataFolder \
--folderType seq \
-o output \
--model ssd300x300coco.bigdl
--classname classname.txt \
-v false \
-b 4
```

In the above commands

* -f: where you put your image data
* --folderType: It can be seq/local
* -o: where you put your image output data
* --model: BigDL serialized model file path
* --classname: file that store detection class names,
 one line for one class. classnames for coco dataset can be found in
  src/main/resources/dataset/coco/classname.txt
* -v: whether to visualize detections
* -b: batch size