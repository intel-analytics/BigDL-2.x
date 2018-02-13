# Faster-RCNN

Faster-RCNN is a popular object detection framework, which is described in 
[paper](https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf) published in NIPS 2015.
It's official code can be found [here](https://github.com/rbgirshick/py-faster-rcnn) 
and a python version can be found [here](https://github.com/SeaOfOcean/py-faster-rcnn).

Later, [PVANET](https://arxiv.org/abs/1611.08588) further reduces computational cost with a lighter network.
It's implementation can be found [here](https://github.com/sanghoon/pva-faster-rcnn)

This example demonstrates how to use BigDL to test a Faster-RCNN framework with either pvanet network or vgg16 network.

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
(https://github.com/intel-analytics/analytics-zoo/tree/master/models) with Spark, you can follow the following command:

```
spark-submit \
--master yarn \
--deploy-mode client \
--executor-cores 28 \
--num-executors 2 \
--driver-memory 128g \
--executor-memory 128g \
--class com.intel.analytics.zoo.pipeline.fasterrcnn.example.Test \
object-detection-0.1-SNAPSHOT-jar-with-dependencies.jar \
-f $voc_test_data \
--model  bigdl_frcnn-pvanet_PASCAL_0.4.0.model \
-t pvanet \
--class data/pascal/classname.txt \
-i voc_2007_test \
-b 56 \
-p 56
```

In the above commands

* -f: where you put your image data
* --model: BigDL model path
* -t: network type, it can be vgg16 or pvanet
* --class: dataset class name file
* -i: image set name with the format ```voc_${year}_${imageset}```, e.g. voc_2007_test
* -b: batch size, it should be n*(partition number)
* -p: partition number

# Run the training example

1. Get the pre-trained model

```
wget https://s3-ap-southeast-1.amazonaws.com/bigdl-models/object-detection/bigdl_vgg16_imagenet.model
```
2. Submit spark job

```
spark-submit \
--master yarn \
--deploy-mode client \
--executor-cores 14 \
--num-executors 2 \
--driver-memory 50g \
--executor-memory 200g \
--class com.intel.analytics.zoo.pipeline.fasterrcnn.example.Train \
object-detection-0.1-SNAPSHOT-jar-with-dependencies.jar \
-f hdfs://xxxx/your_train_folder \
-v hdfs://xxxx/your_val_folder \
--pretrain bigdl_vgg16_imagenet.model \
-i 6000 \
--optim adam \
-l 0.0001 \
-b 28 \
--class data/pascal/classname.txt \
--checkIter 200 \
--summary ../summary \
--checkpoint hdfs://XXX/checkpoint/
```
In the above commands

* -f: where you put the training data
* -v: where you put the validation data
* --pretrain: pretrained model
* -i: max iteration
* --optim: optimizer method
* -l: inital learning rate
* -b: batch size
* --class: dataset class name file
* --checkIter checkpoint iteration interval
* --summary tensorboard summary log dir
* --checkpoint: where to save your checkpoint model





