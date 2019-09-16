# Image classification streaming using Flink and Analytics Zoo

Now, we will use an example to introduce how to use Analytics Zoo with Resnet50 model to accelerate prediction on Flink streaming. See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/apps/model-inference-examples/model-inference-flink/src/main/scala/com/intel/analytics/zoo/apps/model/inference/flink/Resnet50ImageClassification) for the whole program.

There are three main sections in this tutorial.

[Data](#data)

[Defining an Analytics Zoo InferenceModel](#Defining-an-Analytics-Zoo-InferenceModel)

[Getting started the Flink program](#Getting-started-the-Flink-program)

### Data

In this tutorial, we will use the **ImageNet** dataset. It has 1000 classes. The images in ImageNet are various sizes. Let us show some of the predicting images.

![](https://i.loli.net/2019/09/16/4h3qdjXe6wroOlW.png)

Let us extract image from the image folder.

```
# Load image 
val imageFolder = new File("/path/to/imageFolder")
```

Then, you may pre-process data as you need. In this sample, `trait ImageProcessing`  is prepared to provide approaches to convert format, resize and normalize. The methods are defined [here](https://github.com/Le-Zheng/analytics-zoo/blob/test/apps/model-inference-examples/model-inference-flink/src/main/scala/com/intel/analytics/zoo/apps/model/inference/flink/Resnet50ImageClassification/ImageProcessing.scala).

The input image is supposed to be converted as below, and we use a `ListBuffer` to iterate and hold all the input image:

```
# Create data list containing all image Array
var inputs = new ListBuffer[Array[Float]]()

 for (image <- imageFolder.listFiles.toList) {
 	# Three steps. The first step is read each image. The second is preprocessing image. 
 	# At last, add image array to list iterately.
	# Read image to Array[byte]
	val imageBytes = FileUtils.readFileToByteArray(image)

	# apply methods from trait ImageProcessing
	# byteArrayToMat(Array[Byte]): convert Array[byte] to OpenCVMat
	val imageMat = byteArrayToMat(imageBytes)
	
	# centerCrop(mat,W,H,normalized,isClip): do a center crop by resizing a square. normalized and isclip are optional.
	val imageCent = centerCrop(imageMat, 224, 224)
	
	# matToNCHWAndRGBTensor(mat): convert OpenCVMat to Tensor[Float]
	val imageTensor = matToNCHWAndRGBTensor(imageCent)
	
	# An alternative way to add each image array to ListBuffer
	# Convert tensor to Array[Float]
	val input = new Array[Float](imageTensor.nElement())
	inputs += input
}	

# Convert ListBuffer to list
val inputsList = inputs.toList
println(inputs)
```
Out:

```
List([F@6404f418, [F@729d991e, [F@32502377, [F@66f57048, [F@36916eb0, ... [F@194bcebf)
```

### Defining an Analytics Zoo InferenceModel

Analytics Zoo provides Inference Model package for speeding up prediction with deep learning models of Analytics Zoo, Caffe, Tensorflow and OpenVINO Intermediate Representation(IR). 

Define a class extended Analytics Zoo `InferenceModel`.  It can load the pre-trained model easily. The pre-trained model ResNet50 can be obtained from [here](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz). 

Before that, let's define the input parameters of the class:

- **concurrentNum**-the number of requests a model can accept concurrently.
- **modelType**-the pre-trained model type.
- **modelBytes**-the bytes of the model.
- **inputShape**-array which contains the dimensions of the tensor. It should be fed to an input node(s) of the model.
- **ifReverseInputChannels**-the boolean value of if need reverse input channels. Switch the input channels order from RGB to BGR (or vice versa).
- **meanValues(optional)**- mean values only required when converting models.  All input values coming from original network inputs will be divided by this value. 
- **scale(optional)**-to be used for the input image per channel.

Additionally, we prepare to use OpenVINO backend for speeding up in this example, it allows converting by loading `scale` and `meanValues` [parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html). Certainly, using OpenVINO is optional. You may practice in a simple way. 

```
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

Let's define a `Resnet50InferenceModel` class to extend Analytics Zoo `InferenceModel`. 

```
class Resnet50InferenceModel(var concurrentNum: Int = 1, modelType: String, modelBytes: Array[Byte], inputShape: Array[Int], ifReverseInputChannels: Boolean, meanValues: Array[Float], scale: Float)
extends InferenceModel(concurrentNum) with Serializable {

  # load the TF model as OpenVINO IR
  doLoadTF(null, modelType, modelBytes, inputShape, ifReverseInputChannels, meanValues, scale)

  }
}
```

### Getting started the Flink program

We will do the following steps in order:
[1. Obtain an execution environment](#1.-Obtain-an-execution-environment)
[2. Create and transform DataStreams](#2.-Create-and-transform-DataStreams)
[3. Specify Transformation Functions](#Specify-Transformation-Functions)
[4. Trigger the program execution](#4.-Trigger-the-program-execution)
[5. Collect final results](#5.-Collect-final-results)
[6. Run the example on a local machine or a cluster](#6.-Run-the-example-on-a-local-machine-or-a-cluster)

#### 1. Obtain an execution environment

The first step is to create an execution environment. The `StreamExecutionEnvironment` is the context in which a streaming program is executed. `getExecutionEnvironment` is the typical function creating an environment to execute your program when the program is invoked on your local machine or a cluster.

```
val env: StreamExecutionEnvironment = StreamExecutionEnvironment.getExecutionEnvironment
```

#### 2. Create and transform DataStreams

`StreamExecutionEnvironment` provides several stream sources function. As we use `List` to hold the inputs, we can create a DataStream from a collection using `fromCollection()` method.

```
# dataStream
val dataStream: DataStream[Array[Float]] =  env.fromCollection(inputs)

# map to DataStream[JList[JList[JTensor]]]
val tensorStream: DataStream[JList[JList[JTensor]]] = dataStream.map(value => {
  val input = new JTensor(value, Array(1, 224, 224, 3))
  val data = Arrays.asList(input)
  List(data).asJava
})
```

#### 3. Specify transformation functions

Define a class extends `RichMapFunction`. Three main methods of rich function in this example are open, close and map. `open()` is initialization method. `close()` is called after the last call to the main working methods. `map()` is the user-defined function, mapping an element from the input data set and to one exact element, ie, `JList[JList[JTensor]]`.

```
class ModelPredictionMapFunction(modelType: String, modelBytes: Array[Byte], inputShape: Array[Int], ifReverseInputChannels: Boolean, meanValues: Array[Float], scale: Float)
extends RichMapFunction[JList[JList[JTensor]], JList[JList[JTensor]]] {
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
  override def map(in: JList[JList[JTensor]]): JList[JList[JTensor]] = {
    resnet50InferenceModel.doPredict(in)
  }
}
```

Pass the `RichMapFunctionn` function to a `map` transformation.

```
val resultStream = tensorStream.map(new ModelPredictionMapFunction(modelType, modelBytes, inputShape, ifReverseInputChannels, meanValues, scale))
```

#### 4. Trigger the program execution

The program is actually executed only when calling `execute()` on the `StreamExecutionEnvironment`. Whether the program is executed locally or submitted on a cluster depends on the type of execution environment.

```
env.execute()
println(env.getConfig)
```

Out:

```
org.apache.flink.api.common.ExecutionConfig@62f33899
```

#### 5. Collect final results

Finally, create an iterator to iterate over the elements of the DataStream.

```
val results = DataStreamUtils.collect(resultStream.javaStream).asScala
results.foreach(println)
```

Out:

```
[[JTensor{data=[6.978136E-5, 9.844725E-4, 2.3672989E-4, 2.969411E-4, 4.7597036E-4, 1.715969E-4, 1.1608376E-4, 1.8288662E-4, 7.620713E-5, ...], shape=[1, 1000]}]]
...
```

At this step, we complete the whole program. Let's start how to run the whole example.

#### 6. Run the example on a local machine or a cluster

- ##### Build the project

Build the project using Maven because we need the jar file for running on the cluster. Go to the root directory of model-inference-flink and execute the mvn clean package command, which prepares the jar file for model-inference-flink:

```
mvn clean package
```

The resulting jar file will be in the target subfolder: target/model-inference-flink-0.1.0-SNAPSHOT-jar-with-dependencies.jar. We’ll use this later.

- ##### Start and stop Flink

You may start a flink cluster if there is no runing one. Go to the location where you installed Flink :

```
./bin/start-cluster.sh
```

Check the Dispatcher's web frontend at http://localhost:8081 and make sure everything is up and running.
To stop Flink when you're done type:

```
./bin/stop-cluster.sh
```

- ##### Run the Example

  - Run `export FLINK_HOME=the root directory of flink`.
  - Download [resnet_v1_50 model](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz). Run `export MODEL_PATH = path to the downloaded model`.
  - Edit flink-conf.yaml to set heap size or the number of task slots as you need, ie, `jobmanager.heap.size: 10g`
  - Additionally, make sure python requirements for each flink node.

```
sudo apt install python3-pip
pip3 install numpy
pip3 install networkx
pip3 install tensorflow
```

All are ready! Let's run the following command with arguments to submit the Flink program. Change parameter settings as you need.

```bash
${FLINK_HOME}/bin/flink run \
    -m localhost:8081 -p 2 \
    -c ImageClassificationStreaming  \
    /path/to/your/model-inference-flink/target/model-inference-flink-0.1.0-SNAPSHOT-jar-with-dependencies.jar  \
    --modelType resnet_v1_50 --checkpointPath ${MODEL_PATH}  \
    --inputShape "1,224,224,3" --ifReverseInputChannels true --meanValues "123.68,116.78,103.94" --scale 1
```

The output of that command should look similar to this, if everything went according to plan. `JTensor` value shows the prediction of an image in 1000 categories.  And we can see how it accelerates prediction from job runtime here. 

```bash
org.apache.flink.api.common.ExecutionConfig@62f33899
[[JTensor{data=[6.978136E-5, 9.844725E-4, 2.3672989E-4, 2.969411E-4, 4.7597036E-4, 1.715969E-4, 1.1608376E-4, 1.8288662E-4, 7.620713E-5, ...], shape=[1, 1000]}]]
[[JTensor{data=[4.973883E-5, 3.720686E-4, 1.5099304E-5, 2.262065E-5, 1.2184962E-4, 1.618636E-4, 2.8170096E-5, 4.0111943E-5, 6.963265E-4, ...], shape=[1, 1000]}]]
...
Program execution finished
Job with JobID f0bedd54bd81db640833c283a9283289 has finished.
Job Runtime: 14830 ms
```

#### Wrapping up

we have reached the end of the tutorial. In this tutorial, you have learned how to use Analytics Zoo and Flink for image classification on streaming. You also know about creating  the `InferenceModel` class for loading and prediction with a deep learning model. With that, you defined your own `RichMapFunction` and started with the prediction on Flink streaming.

What goes for next? You could take practice. Load the data and model you need to see what speedup you get.
