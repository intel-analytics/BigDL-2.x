# Image classification streaming using Flink and Analytics Zoo

Now, we will use an example to introduce how to use Analytics Zoo with Resnet50 model to accelerate prediction on Flink streaming. See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/apps/model-inference-examples/model-inference-flink/src/main/scala/com/intel/analytics/zoo/apps/model/inference/flink/Resnet50ImageClassification) for the whole program.

There are three main sections in this tutorial.

- [Loading data](#Loading-data)  

- [Defining an Analytics Zoo InferenceModel](#Defining-an-Analytics-Zoo-InferenceModel)  

- [Getting started the Flink program](#Getting-started-the-Flink-program)  

## Loading data

In this tutorial, we will use the **ImageNet** dataset. It has 1000 classes. The images in ImageNet are various sizes. Let us show some of the predicting images.

![img](https://i.loli.net/2019/09/21/Q9gCTVjzv3m5sFI.png)

Let us load images from the image folder.

```scala
# Load images from folder, and hold images as a list
val fileList = new File("/path/to/imageFolder").listFiles.toList
```

Then, you may pre-process data as you need. In this sample, `trait ImageProcessing` is prepared to provide approaches to convert format, resize and normalize. The methods are defined [here](https://github.com/intel-analytics/analytics-zoo/blob/master/apps/model-inference-examples/model-inference-flink/src/main/scala/com/intel/analytics/zoo/apps/model/inference/flink/Resnet50ImageClassification/ImageProcessing.scala).

The input image is supposed to be converted as below:

```scala
val inputs = fileList.map(file => {
	# Three steps. The first step is reading each image. The second is preprocessing image. 
 	# At last, convert data to the required format.
	# Read image to Array[byte]
	val imageBytes = FileUtils.readFileToByteArray(file)

	# Apply methods from trait ImageProcessing.
	# byteArrayToMat(Array[Byte]): convert Array[byte] to OpenCVMat.
	val imageMat = byteArrayToMat(imageBytes)
	
	# centerCrop(mat,W,H,normalized,isClip): do a center crop by resizing a square. normalized and isclip are optional.
	# size(224,224) is available for Resnet50 model. 
	val imageCent = centerCrop(imageMat, 224, 224)    
	
	# matToNCHWAndArray(mat): convert OpenCVMat to NCHW[N,channel,height,width] and Array.
	val imageArray = matToNCHWAndArray(imageCent)
	
	# Converet Aarry to [JList[JList[JTensor]]]
	val input = new JTensor(imageArray, Array(1, 224, 224, 3))
	List(util.Arrays.asList(input)).asJava
    })
```

## Defining an Analytics Zoo InferenceModel

Analytics Zoo provides Inference Model package for speeding up prediction with deep learning models of Analytics Zoo, Caffe, Tensorflow and OpenVINO Intermediate Representation(IR). You may see [here](https://github.com/intel-analytics/analytics-zoo/blob/b98d97a7eb2f8c88bcaa34ee135077da0aac091d/zoo/src/main/scala/com/intel/analytics/zoo/pipeline/inference/InferenceModel.scala) for more details of Inference Model APIs.

Define a class extended Analytics Zoo `InferenceModel`. We use the pre-trained model  Restnet50 and load it as OpenVINO IR in this example. You may look up the parameter value to convert a ResNet50 model  [here](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html).

Before that, let's define the input parameters of the class:

- **concurrentNum**-the number of requests a model can accept concurrently.
- **modelType**-the pre-trained model type.
- **modelBytes**-the bytes of the model.
- **inputShape**-array which contains the dimensions of the tensor. It should be fed to an input node(s) of the model.
- **ifReverseInputChannels**-the boolean value of if need reverse input channels. Switch the input channels order from RGB to BGR (or vice versa).
- **meanValues**- All input values coming from original network inputs will be divided by this value. The TensorFlow*-Slim Models were trained with normalized input data. Inference Engine classification sample does not perform normalization. It is necessary to pass mean and scale values to the Model Optimizer so they are embedded into the generated IR in order to get correct classification results.
- **scale**-to be used for the input image per channel.

```scala
# concurrentNum
var concurrentNum = 1

# modelType
var modelType = "resnet_v1_50"

# ifReverseInputChannels
var ifReverseInputChannels = true

# inputShape
var inputShape = Array(1, 224, 224, 3)

# meanValues to convert resnet_v1_50 model
var meanValues = Array(123.68f, 116.78f, 103.94f)

# scale to convert resnet_v1_50 model
var scale = 1.0f

# To obtain model bytes, there are several steps
# abstract path name of the model checkpoint file
var checkpointPath: String = "/path/to/models/resnet_v1_50.ckpt"

# length of the file in bytes
val fileSize = new File(checkpointPath).length()

# Create a file inputStream to read streams of bytes
val inputStream = new FileInputStream(checkpointPath)

#  convert fileSize to Array[Byte]
val modelBytes = new Array[Byte](fileSize.toInt)
inputStream.read(modelBytes)
```

Let's define a `Resnet50InferenceModel` class to extend analytics zoo `InferenceModel`.

```scala
class Resnet50InferenceModel(var concurrentNum: Int = 1, modelType: String, modelBytes: Array[Byte], inputShape: Array[Int], ifReverseInputChannels: Boolean, meanValues: Array[Float], scale: Float)
extends InferenceModel(concurrentNum) with Serializable {

  # load the TF model as OpenVINO IR
  doLoadTF(null, modelType, modelBytes, inputShape, ifReverseInputChannels, meanValues, scale)

  }
}
```

## Getting started the Flink program

We will do the following steps in order:

[1. Obtain an execution environment](#1-obtain-an-execution-environment)  
[2. Create and transform DataStreams](#2-create-and-transform-datastreams)  
[3. Specify transformation functions](#3-specify-transformation-functions)  
[4. Trigger the program execution](#4-trigger-the-program-execution)    
[5. Collect final results](#5-collect-final-results)   
[6. Run the example on a local machine or a cluster](#6-run-the-example-on-a-local-machine-or-a-cluster)

### 1. Obtain an execution environment

The first step is to create an execution environment. The `StreamExecutionEnvironment` is the context in which a streaming program is executed. `getExecutionEnvironment` is the typical function creating an environment to execute your program when the program is invoked on your local machine or a cluster.

```scala
val env: StreamExecutionEnvironment = StreamExecutionEnvironment.getExecutionEnvironment
```

### 2. Create and transform DataStreams

`StreamExecutionEnvironment` provides several stream sources function. As we use `List` to hold the inputs, we can create a DataStream from a collection using `fromCollection()` method.

```scala
# dataStream
val dataStream: DataStream[JList[JList[JTensor]]] = env.fromCollection(inputs)
```

### 3. Specify transformation functions

Define a class extends `RichMapFunction`. Three main methods of rich function in this example are open, close and map. `open()` is initialization method. `close()` is called after the last call to the main working methods. `map()` is the user-defined function, mapping an element from the input data set and to one exact element, ie, `JList[JList[JTensor]]`.

```scala
class ModelPredictionMapFunction(modelType: String, modelBytes: Array[Byte], inputShape: Array[Int], ifReverseInputChannels: Boolean, meanValues: Array[Float], scale: Float)
extends RichMapFunction[JList[JList[JTensor]], Int] {
  var resnet50InferenceModel: Resnet50InferenceModel = _

  # open
  override def open(parameters: Configuration): Unit = {
    resnet50InferenceModel = new Resnet50InferenceModel(1, modelType, modelBytes, inputShape, ifReverseInputChannels, meanValues, scale)
  }

  # close
  override def close(): Unit = {
    resnet50InferenceModel.release()
  }

  # map
    override def map(in: JList[JList[JTensor]]): (Int) = {
    val outputData = resnet50InferenceModel.doPredict(in).get(0).get(0).getData
    val index = outputData.indexOf(outputData.max)
    (index)
  }
}
```

Pass the `RichMapFunctionn` function to a `map` transformation.

```scala
val resultStream = dataStream.map(new ModelPredictionMapFunction(modelType, modelBytes, inputShape, ifReverseInputChannels, meanValues, scale))
```

### 4. Trigger the program execution

The program is actually executed only when calling `execute()` on the `StreamExecutionEnvironment`. Whether the program is executed locally or submitted on a cluster depends on the type of execution environment.

```scala
env.execute()
```

### 5. Collect final results

Finally, create an iterator to iterate over the elements of the DataStream.

```scala
val results = DataStreamUtils.collect(resultStream.javaStream).asScala
results.foreach((i) => println(labels(i)))
```

Out:

```
matchstick
typewriter keyboard
malamute, malemute, Alaskan malamute
daisy
espresso maker
whiskey jug
cardoon
seat belt, seatbelt
lens cap, lens cover
```

At this step, we complete the whole program. Let's start how to run the example on a cluster.

### 6. Run the example on a local machine or a cluster

- ##### Build the project

Build the project using Maven because we need the jar file for running on the cluster. Go to the root directory of your inference flink project and execute the mvn clean package command, which prepares the jar file for your model inference flink program:

```scala
mvn clean package
```

The resulting jar file will be in the target subfolder: target/model-inference-flink-0.1.0-SNAPSHOT-jar-with-dependencies.jar. We’ll use this later.

- ##### Start and stop Flink

You may start a flink cluster if there is no running one. Go to the location where you installed fink :

```scala
./bin/start-cluster.sh
```

Check the Dispatcher's web frontend at [http://localhost:8081](http://localhost:8081/) and make sure everything is up and running. To stop Flink when you're done type:

```scala
./bin/stop-cluster.sh
```

- ##### Run the Example

Additionally, in this example, make sure the python requirements of OpenVINO for each flink node.

```shell
sudo apt install python3-pip
pip3 install numpy networkx tensorflow
```

All are ready! Let's run the following command with arguments to submit the Flink program. Change parameter settings as you need.

```shell
/path/to/FLINK_HOME/bin/flink run \
    -m localhost:8081 -p 2 \
    -c YourImageClassificationStreaming  \
    /path/to/your-inference-flink-project/target/model-inference-flink-0.1.0-SNAPSHOT-jar-with-dependencies.jar  
```

The output of that command should look similar to this, if everything went according to plan.  It shows the prediction result.

```
matchstick
typewriter keyboard
malamute, malemute, Alaskan malamute
daisy
espresso maker
whiskey jug
cardoon
seat belt, seatbelt
lens cap, lens cover
Program execution finished
Job with JobID f0bedd54bd81db640833c283a9283289 has finished.
Job Runtime: 10830 ms
```

#### Wrapping up

we have reached the end of the tutorial. In this tutorial, you have learned how to use Analytics Zoo and Flink for image classification on streaming. You also know about creating the `InferenceModel` class for loading and prediction with a deep learning model. With that, you defined your own `RichMapFunction` and started with the prediction on Flink streaming.

What goes for next? You could take practice. Load the data and model you need to see what speedup you get.
