# BigDL Model Zoo

BigDL provides a collection of pre-trained models for Image Classification and Object Detection. These models can be used for out-of-the-box inference if you are interested in categories already in the corresponding datasets. According to the business scenarios, users can embed the models locally, distributedly in Spark or other analytics platform such as Storm streaming.

***Image Classification models*** 

BigDL provides popular pre-trained models on the [ImageNet](http://image-net.org/index) dataset. For the usage of these models, please check below examples

[Scala example](./src/main/scala/com/intel/analytics/zoo/models/imageclassification/example/Predict.scala)

[Python example](./src/main/python/bigdl/zoo/example/ImageClassificationPredict.py)

In the above example, it's very easy to apply the model for inference simply with below code piece.

```scala
val model = Module.loadModule[Float](params.model) // load model
val data = ImageFrame.read(params.imageFolder, sc) // read image data
val predictor = Predictor(model) // model specific Predictor
val predict = predictor.predict(data)
```
Each pre-trained model contains an unified identifier following <font color=gray>*publisher_model_dataset_version*</font> pattern. e.g., bigdl_alexnet_imagenet_0.4.0 represents Alexnet trained by BigDL 0.4.0 on top of Imagenet dataset.

[Predictor](./src/main/scala/com/intel/analytics/zoo/models/Predictor.scala) is model specific predictor, it simplifies the inference by creating the preprocessor (transformer) on top of the identifier. Each module has its own preprocessor, check [Configure](./src/main/scala/com/intel/analytics/zoo/models/Configure.scala) for details, you can find preprocessors for all models from [ImagenetConfig](./src/main/scala/com/intel/analytics/zoo/models/imageclassification/util/ImageClassificationConfig.scala).

Alternatively, users can also do the inference directly using BigDL, the only difference is that you need to explicitly define the transformer.

Sample code as below for Alexnet:

```scala
val model = Module.loadModule[Float](params.model) // load model
val data = ImageFrame.read(params.imageFolder, sc) // read image data
val preprocessor =  Resize(256, 256) ->
                        PixelNormalizer(mean) -> CenterCrop(227, 227) ->
                        MatToTensor() -> ImageFrameToSample()
data -> preprocessor
val predict = model.predictImage(data)
```

***Object Detection models***

BigDL provides two typical kind of pre-trained Object Detection models : [SSD](https://arxiv.org/abs/1512.02325) and [Faster-RCNN](https://arxiv.org/abs/1506.01497) on dataset [PASCAL](http://host.robots.ox.ac.uk/pascal/VOC/) and [COCO](http://cocodataset.org/#home). For the usage of these models, please check below examples.

[Scala example](./src/main/scala/com/intel/analytics/zoo/models/objectdetection/example/Predict.scala)

Similar to Image Classification, it's very easy to apply the model for inference with below code piece.

```scala
val model = Module.loadModule[Float](params.model)
val data = ImageFrame.read(params.image, sc, params.nPartition)
val predictor = Predictor(model)
val output = predictor.predict(data)
```

For preprocessors for Object Detection models, please check [Object Detection Config](./src/main/scala/com/intel/analytics/zoo/models/objectdetection/utils/ObjectDetectionConfig.scala)

Similar to Image Classification, users can also do the inference directly using BigDL.
Sample code for SSD VGG on PASCAL as below:

```scala
val model = Module.loadModule[Float](params.model)
val data = ImageFrame.read(params.image, sc, params.nPartition)
val preprocessor = Resize(300, 300) ->
                         ChannelNormalize(123f, 117f, 104f, 1f, 1f, 1f) ->
                         MatToTensor() -> ImageFrameToSample()
val predict = model.predictImage(data)
```
##Download link
### Image Classification

1. Imagenet models

* [Alexnet](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/imagenet/bigdl_alexnet_imagenet_0.4.0.model)
* [Alexnet Quantize](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/imagenet/bigdl_alexnet-quantize_imagenet_0.4.0.model)
* [Inception-V1](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/imagenet/bigdl_inception-v1_imagenet_0.4.0.model)
* [Inception-V1 Quantize](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/imagenet/bigdl_inception-v1-quantize_imagenet_0.4.0.model)
* [VGG-16](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/imagenet/bigdl_vgg-16_imagenet_0.4.0.model)
* [VGG-16 Quantize](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/imagenet/bigdl_vgg-16-quantize_imagenet_0.4.0.model)
* [VGG-19](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/imagenet/bigdl_vgg-19_imagenet_0.4.0.model)
* [VGG-19 Quantize](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/imagenet/bigdl_vgg-19-quantize_imagenet_0.4.0.model)
* [Resnet-50](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/imagenet/bigdl_resnet-50_imagenet_0.4.0.model)
* [Resnet-50 Quantize](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/imagenet/bigdl_resnet-50-quantize_imagenet_0.4.0.model)
* [Densenet-161](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/imagenet/bigdl_densenet-161_imagenet_0.4.0.model)
* [Densenet-161 Quantize](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/imagenet/bigdl_densenet-161-quantize_imagenet_0.4.0.model)
* [Mobilenet](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/imagenet/bigdl_mobilenet_imagenet_0.4.0.model)
* [Squeezenet](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/imagenet/bigdl_squeezenet_imagenet_0.4.0.model)
* [Squeezenet Quantize](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/imagenet/bigdl_squeezenet-quantize_imagenet_0.4.0.model)

### Object Detection

1. PASCAL VOC models
* [SSD 300x300 MobileNet](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/object-detection/bigdl_ssd-mobilenet-300x300_PASCAL_0.4.0.model)
* [SSD 300x300 VGG](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/object-detection/bigdl_ssd-vgg16-300x300_PASCAL_0.4.0.model)
* [SSD 300x300 VGG Quantize](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/object-detection/bigdl_ssd-vgg16-300x300-quantize_PASCAL_0.4.0.model)
* [SSD 512x512 VGG](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/object-detection/bigdl_ssd-vgg16-512x512_PASCAL_0.4.0.model)
* [SSD 512x512 VGG Quantize](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/object-detection/bigdl_ssd-vgg16-512x512-quantize_PASCAL_0.4.0.model)
* [Faster-RCNN VGG](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/object-detection/bigdl_frcnn-vgg16_PASCAL_0.4.0.model)
* [Faster-RCNN VGG Compress](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/object-detection/bigdl_frcnn-vgg16-compress_PASCAL_0.4.0.model)
* [Faster-RCNN VGG Compress Quantize](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/object-detection/bigdl_frcnn-vgg16-compress-quantize_PASCAL_0.4.0.model)
* [Faster-RCNN PvaNet](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/object-detection/bigdl_frcnn-pvanet_PASCAL_0.4.0.model)
* [Faster-RCNN PvaNet Compress](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/object-detection/bigdl_frcnn-pvanet-compress_PASCAL_0.4.0.model)
* [Faster-RCNN PvaNet Compress Quantize](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/object-detection/bigdl_frcnn-pvanet-compress-quantize_PASCAL_0.4.0.model)

2. COCO models

* [SSD 300x300 VGG](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/object-detection/bigdl_ssd-vgg16-300x300_COCO_0.4.0.model)
* [SSD 300x300 VGG Quantize](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/object-detection/bigdl_ssd-vgg16-300x300-quantize_COCO_0.4.0.model)
* [SSD 512x512 VGG](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/object-detection/bigdl_ssd-vgg16-512x512_COCO_0.4.0.model)
* [SSD 512x512 VGG Quantize](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/object-detection/bigdl_ssd-vgg16-512x512-quantize_COCO_0.4.0.model)

