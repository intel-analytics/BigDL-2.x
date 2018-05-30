
## Analytics Zoo

*Analytics + AI Platform for Apache Spark and [BigDL](https://bigdl-project.github.io/master/#whitepaper/).*

---

Analytics Zoo makes it easy to build deep learning application on Spark and BigDL, by providing an end-to-end analytics + AI Platform (including *high level pipeline APIs*, *built-in deep learning models*, *reference use cases*, etc.).

 * [High level pipeline APIs](#high-level-pipeline-apis)
    * [`nnframes`](#nnframes): native deep learning support in *Spark DataFrames and ML Pipelines*
    * [`autograd`](#autograd): build custom layer/loss using *auto differentiation operations* 
    * [Transfer learning](#transfer-learning): customize pretained model for *feature extraction or fine-tuning*
  
 - [Built-in deep learning models](#built-in-deep-learning-models)
    - [Object detection API](#object-detection-api): high-level API and pretrained models (e.g., SSD and Faster-RCNN) for *object detection*
    - [Image classification API](#image-classification-api): high-level API and pretrained models (e.g., VGG, Inception, ResNet, MobileNet, etc.) for *image classification*
    - [Text classification API](#text-classification-api): high-level API and pre-defined models (using CNN, LSTM, etc.) for *text classification*
    - [Recommedation API](#recommendation-api): high-level API and pre-defined models (e.g., Neural Collaborative Filtering, Wide and Deep Learning, etc.) for *recommendation*
  
 - [Reference use cases](#reference-use-cases): a collection of end-to-end *reference use cases* (e.g., anomaly detection, sentiment analysis, fraud detection, image augmentation, object detection, variational autoencoder, etc.)

---
## High level pipeline APIs
Analytics Zoo provides a set of easy-to-use, high level pipeline APIs that natively support Spark DataFrames and ML Pipelines, autograd and custom layer/loss, trasnfer learning, etc.

### `nnframes`
`nnframes` provides *native deep learning support in Spark DataFrames and ML Pipelines*, so that you can easily build complex deep learning pipelines in just a few lines, as illustracted below. (See more details [here](ProgrammingGuide/nnframes.md))

 1.Load images into DataFrames using `NNImageReader`

```
   from zoo.common.nncontext import *
   from zoo.pipeline.nnframes import *
   sc = get_nncontext()
   imageDF = NNImageReader.readImages(image_path, sc)
```

 2.Process loaded data using *DataFrames transformations*

```
   getName = udf(lambda row: ...)
   getLabel = udf(lambda name: ...)
   df = imageDF.withColumn("name", getName(col("image"))).withColumn("label", getLabel(col('name')))
```

 3.Processing image using built-in *feature engineering operations*
   
```
   from zoo.feature.image import *
   transformer = RowToImageFeature() -> ImageResize(64, 64) -> ImageChannelNormalize(123.0, 117.0, 104.0) \
                 -> ImageMatToTensor() -> ImageFeatureToTensor())
```

 4.Define model using *Keras-style APIs*
  
```
   from zoo.pipeline.api.keras.layers import *
   from zoo.pipeline.api.keras.models import *
   model = Sequential().add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1, 28, 28))) \
                   .add(MaxPooling2D(pool_size=(2, 2))).add(Flatten()).add(Dense(10, activation='softmax')))
```

 5.Train model using *Spark ML Pipelines*
```
   classifier = NNClassifier(model, CrossEntropyCriterion(),transformer).setLearningRate(0.003) \
                   .setBatchSize(40).setMaxEpoch(1).setFeaturesCol("image").setCachingSample(False)
   nnModel = classifier.fit(df)
```
   
### `autograd`
`autograd` provides automatic differentiation for math operations, so that you can easily build your own *custom loss and layer* (in both Python and Scala), as illustracted below. (See more details [here](ProgrammingGuide/autograd.md))

 1.Define custom functions using `autograd`

```
   from zoo.pipeline.api.autograd import *
   
   def mean_absolute_error(y_true, y_pred):
       return mean(abs(y_true - y_pred), axis=1)
   
   def add_one_func(x):
       return x + 1.0
```

 2.Define model using Keras-style API and *custom `Lambda` layer*

```
   from zoo.pipeline.api.keras.layers import *
   from zoo.pipeline.api.keras.models import *
   model = Sequential().add(Dense(1, input_shape=(2,))) \
                       .add(Lambda(function=add_one_func))
```

 3.Train model with *custom loss function*

```
   model.compile(optimizer = SGD(), loss = mean_absolute_error)
   model.fit(x = ..., y = ...)
```

### Transfer learning
Using the high level transfer learning APIs, you can easily customize pretrained models for *feature extraction or fine-tuning*. (See more details [here](ProgrammingGuide/transferlearning.md))

 1.Load an existing model (pretrained in Caffe)
   
```
   from zoo.pipeline.api.net import *
   full_model = Net.load_caffe(def_path, model_path)
```

 2.Remove last few layers

```
   # create a new model by remove layers after pool5/drop_7x7_s1
   model = full_model.new_graph(["pool5/drop_7x7_s1"])
```

 3.Freeze first few layers

```
   # freeze layers from input to pool4/3x3_s2 inclusive
   model.freeze_up_to(["pool4/3x3_s2"])
```

 4.Add a few new layers

```
   from zoo.pipeline.api.keras.layers import *
   from zoo.pipeline.api.keras.models import *
   inputs = Input(name="input", shape=(3, 224, 224))
   inception = model.to_keras()(inputs)
   flatten = Flatten()(inception)
   logits = Dense(2)(flatten)
   newModel = Model(inputs, logits)
```

---
## Built-in deep learning models
Analytics Zoo provides several built-in deep learning models that you can use for a variety of problem types, such as *object detection*, *image classification*, *text classification*, *recommendation*, etc.

### Object detection API
Using *Analytics Zoo Object Detection API* (including a set of pretrained detection models such as SSD and Faster-RCNN), you can easily build your object detection applications (e.g., localizing and identifying multiple objects in images and videos), as illustrated below. (See more details [here](ProgrammingGuide/object-detection.md))

 1.Download object detection models in Analytics Zoo

   You can download a collection of detection models (pretrained on the PSCAL VOC dataset and COCO dataset) from [detection model zoo](ProgrammingGuide/object-detection.md#download-link).

 2.Use *Object Detection API* for off-the-shell inference

```
   from zoo.models.image.objectdetection import *
   model = ObjectDetector.load_model(model_path)
   image_set = ImageSet.read(img_path, sc)
   output = model.predict_image_set(image_set)
```

### Image classification API
Using *Analytics Zoo Image Classification API* (including a set of pretrained detection models such as VGG, Inception, ResNet, MobileNet,  etc.), you can easily build your image classification applications, as illustrated below. (See more details [here](ProgrammingGuide/image-classification.md))

 1.Download image classification models in Analytics Zoo

   You can download a collection of image classification models (pretrained on the ImageNet dataset) from [image classification model zoo](ProgrammingGuide/image-classification.md#download-link)

 2.Use *Image classification API* for off-the-shell inference

```
   from zoo.models.image.imageclassification import *
   model = ImageClassifier.load_model(model_path)
   image_set = ImageSet.read(img_path, sc)
   output = model.predict_image_set(image_set)
```

### Text classification API

*Analytics Zoo Text Classification API* provides a set of pre-defined models (using CNN, LSTM, etc.) for text classifications. (See more details [here](ProgrammingGuide/text-classification.md))

### Recommendation API

*Analytics Zoo Recommendation API* provides a set of pre-defined models (such as Neural Collaborative Filtering, Wide and Deep Learning, etc.) for recommendations. (See more details [here](ProgrammingGuide/recommendation.md))

---
## Reference use cases
Analytics Zoo provides a collection of end-to-end reference use cases, including *anomaly detection (for time series data)*, *sentiment analysis*, *fraud detection*, *image augmentation*, *object detection*, *variational autoencoder*, etc. (See more details [here](ProgrammingGuide/usercases-overview.md))



















































