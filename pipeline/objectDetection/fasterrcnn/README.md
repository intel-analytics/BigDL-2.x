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





