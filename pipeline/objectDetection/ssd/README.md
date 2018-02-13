# SSD: Single Shot MultiBox Detector

[SSD](https://research.google.com/pubs/pub44872.html) is one of the state-of-the-art
 object detection pipelines.

Currently Bigdl has added support for ssd with vgg as base model,
with 300x300 or 512x512 input resolution.

## Prepare the dataset

### Prepare labeled dataset for validation and training
1. [Pascal VOC](../data/pascal)
2. [Coco](../data/coco)

### Convert unlabeled image folder to sequence file
If you want to convert a folder of images to sequence file, run the following command
```bash
./data/tool/convert_image_folder.sh image_folder output
```

where ```image_folder``` is your image folder, ```output``` is the output folder

please adjust the arguments if necessary

## Validate pre-trained model
If you want to validate [pre-trained model]
(https://github.com/intel-analytics/analytics-zoo/tree/master/models) with Spark,
 you can follow the following command:

```bash
spark-submit \
--master yarn \
--deploy-mode client \
--executor-cores 28 \
--num-executors 2 \
--driver-memory 20g \
--executor-memory 128g \
--class com.intel.analytics.zoo.pipeline.ssd.example.Test \
object-detection-0.1-SNAPSHOT-jar-with-dependencies.jar \
-f $voc_test_data \
--model bigdl_ssd-vgg16-300x300_PASCAL_0.4.0.model \
-t vgg16 \
--class data/pascal/classname.txt \
-i voc_2007_test \
-p 112 \
-b 112 \
-r 300
```

In the above commands

* -f: where you put your image data
* --model: BigDL model path
* -t: network type, it can be vgg16
* --class: dataset class name file
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
object-detection-0.1-SNAPSHOT-jar-with-dependencies.jar \
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
--class data/pascal/classname.txt \
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
* --class: dataset class name file
* --checkpoint: where to save your checkpoint model
