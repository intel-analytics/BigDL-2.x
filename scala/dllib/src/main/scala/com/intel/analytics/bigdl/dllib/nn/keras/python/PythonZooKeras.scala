/*
 * Copyright 2018 Analytics Zoo Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.zoo.pipeline.api.keras.python

import java.nio.ByteOrder
import java.util
import java.util.{HashMap => JHashMap, List => JList, Map => JMap}

import com.intel.analytics.bigdl.Criterion
import com.intel.analytics.bigdl.dataset.{DataSet, LocalDataSet, MiniBatch}

import scala.collection.JavaConverters._
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.python.api.{EvaluatedResult, JTensor, PythonBigDLKeras, Sample}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.nn.{Container, InitializationMethod}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.Container
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractCriterion, AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.{KerasLayer, KerasModel}
import com.intel.analytics.bigdl.utils.{Engine, Table}
import com.intel.analytics.zoo.feature.image.ImageSet
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.autograd._
import com.intel.analytics.zoo.pipeline.api.keras.layers.{KerasLayerWrapper, _}
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.metrics.{AUC, Accuracy, Top5Accuracy}
import com.intel.analytics.zoo.pipeline.api.keras.models.{KerasNet, Model, Sequential}
import com.intel.analytics.zoo.pipeline.api.keras.objectives._
import com.intel.analytics.zoo.pipeline.api.net._
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.rdd.RDD
import com.intel.analytics.bigdl.dataset.{Sample => JSample}

import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.reflect.ClassTag
import scala.reflect.io.Path

object PythonZooKeras {

  def ofFloat(): PythonZooKeras[Float] = new PythonZooKeras[Float]()

  def ofDouble(): PythonZooKeras[Double] = new PythonZooKeras[Double]()
}

class PythonZooKeras[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDLKeras[T] {

  def createZooKerasModel(
      input: JList[Variable[T]],
      output: JList[Variable[T]]): Model[T] = {
    Model[T](input.asScala.map(_.node).toArray, output.asScala.map(_.node).toArray)
  }

  def createZooKerasSequential(): Sequential[T] = {
    Sequential[T]()
  }

  def createZooKerasIdentity(
    inputShape: JList[Int] = null): Identity[T] = {
    Identity[T](toScalaShape(inputShape))
  }

  def createZooKerasInput(
      inputShape: JList[JList[Int]] = null,
      name : String = null): Variable[T] = {
    new Variable[T](Input[T](name = name, inputShape = toScalaMultiShape(inputShape)), name)
  }

  def createZooKerasInputLayer(
      inputShape: JList[Int] = null): KerasLayer[Activity, Activity, T] = {
    InputLayer(inputShape = toScalaShape(inputShape))
  }

  def zooCompile(
      module: KerasNet[T],
      optimizer: OptimMethod[T],
      loss: Criterion[T],
      metrics: JList[ValidationMethod[T]] = null): Unit = {
    module.compile(optimizer, loss,
      if (metrics == null) null else metrics.asScala.toList)
  }

  def zooFit(
      module: KerasNet[T],
      x: JavaRDD[Sample],
      batchSize: Int = 32,
      nbEpoch: Int = 10,
      validationData: JavaRDD[Sample] = null): Unit = {
    module.fit(toJSample(x), batchSize, nbEpoch,
      if (validationData == null) null else toJSample(validationData))
  }

  def zooFit(
      module: KerasNet[T],
      x: ImageSet,
      batchSize: Int,
      nbEpoch: Int,
      validationData: ImageSet): Unit = {
    module.fit(x, batchSize, nbEpoch, validationData)
  }

  def zooFit(
      module: KerasNet[T],
      xTrain: JList[JTensor],
      yTrain: JTensor,
      batchSize: Int,
      nbEpoch: Int,
      xVal: JList[JTensor],
      yVal: JTensor): Unit = {
    val trainArray = toSampleArray(xTrain.asScala.toList.map{f => toTensor(f)}, toTensor(yTrain))
    val trainData = batching(DataSet.array(trainArray), batchSize)
      .asInstanceOf[LocalDataSet[MiniBatch[T]]]
    val valData = if (xVal != null && yVal != null) {
      val valArray = toSampleArray(xVal.asScala.toList.map{f => toTensor(f)}, toTensor(yVal))
      batching(DataSet.array(valArray), batchSize)
    } else null
    module.fit(trainData, nbEpoch, valData)
  }

  def zooPredict(
      module: AbstractModule[Activity, Activity, T],
      x: JavaRDD[Sample],
      batchPerThread: Int): JavaRDD[JList[Object]] = {
    val resRDD = module match {
      case net: KerasNet[T] =>
        net.predict(x.rdd.map(toJSample), batchPerThread)
      case _ =>
        module.predict(x.rdd.map(toJSample), batchPerThread * x.getNumPartitions)
    }

    resRDD.map(activityToList).toJavaRDD()
  }

  def zooForward(model: AbstractModule[Activity, Activity, T],
                   input: JList[JTensor],
                   inputIsTable: Boolean): JList[Object] = {
    val inputActivity = jTensorsToActivity(input, inputIsTable)
    val outputActivity = model.forward(inputActivity)
    activityToList(outputActivity)
  }

  def activityToList(outputActivity: Activity): JList[Object] = {
    if (outputActivity.isInstanceOf[Tensor[T]]) {
      val list = new util.ArrayList[Object]()
      list.add(toJTensor(outputActivity.toTensor))
      list
    } else {
      table2JList(outputActivity.toTable)
    }
  }

  private def table2JList(t: Table): JList[Object] = {
    var i = 1
    val list = new util.ArrayList[Object]()
    while (i <= t.length()) {
      val item = t[Object](i)
      if (item.isInstanceOf[Tensor[T]]) {
        list.add(toJTensor(item.asInstanceOf[Tensor[T]]))
      } else if (item.isInstanceOf[Table]) {
        list.add(table2JList(item.asInstanceOf[Table]))
      } else {
        throw new IllegalArgumentException(s"Table contains unrecognizable objects $item")
      }
      i += 1
    }
    list
  }

  def zooPredict(
      module: KerasNet[T],
      x: JList[JTensor],
      batchPerThread: Int): JList[JList[Object]] = {
    val sampleArray = toSampleArray(x.asScala.toList.map{f => toTensor(f)})
    val localPredictor = LocalPredictor(module,
      batchPerCore = batchPerThread)
    val result = localPredictor.predict(sampleArray)
    result.map(activityToList).toList.asJava
  }

  def zooPredict(
      module: KerasNet[T],
      x: ImageSet,
      batchPerThread: Int): ImageSet = {
    module.predict(x, batchPerThread)
  }

  def zooEvaluate(
      module: KerasNet[T],
      x: JavaRDD[Sample],
      batchSize: Int = 32): JList[EvaluatedResult] = {
    val resultArray = module.evaluate(toJSample(x), batchSize)
    val testResultArray = resultArray.map { result =>
      EvaluatedResult(result._1.result()._1, result._1.result()._2,
        result._2.toString())
    }
    testResultArray.toList.asJava
  }

  def zooEvaluate(
      module: KerasNet[T],
      x: ImageSet,
      batchSize: Int): JList[EvaluatedResult] = {
    val resultArray = module.evaluate(x, batchSize)
    val testResultArray = resultArray.map { result =>
      EvaluatedResult(result._1.result()._1, result._1.result()._2,
        result._2.toString())
    }
    testResultArray.toList.asJava
  }

  def zooSetTensorBoard(
      module: KerasNet[T],
      logDir: String,
      appName: String): Unit = {
    module.setTensorBoard(logDir, appName)
  }

  def zooClearGradientClipping(module: KerasNet[T]): Unit = {
    module.clearGradientClipping()
  }

  def zooSetConstantGradientClipping(
      module: KerasNet[T],
      min: Float,
      max: Float): Unit = {
    module.setConstantGradientClipping(min, max)
  }

  def zooSetGradientClippingByL2Norm(
      module: KerasNet[T],
      clipNorm: Float): Unit = {
    module.setGradientClippingByL2Norm(clipNorm)
  }

  def zooSetCheckpoint(
      module: KerasNet[T],
      path: String,
      overWrite: Boolean = true): Unit = {
    module.setCheckpoint(path, overWrite)
  }

  def zooSaveGraphTopology(
      module: Model[T],
      logPath: String,
      backward: Boolean = false): Model[T] = {
    module.saveGraphTopology(logPath, backward)
  }

  def zooPredictClasses(
      module: KerasNet[T],
      x: JavaRDD[Sample],
      batchPerThread: Int,
      zeroBasedLabel: Boolean = true): JavaRDD[Int] = {
    module.predictClasses(toJSample(x), batchPerThread, zeroBasedLabel).toJavaRDD()
  }

  def newGraph(model: NetUtils[T, _],
               outputs: JList[String]): NetUtils[T, _] = {
    model.newGraph(outputs.asScala).asInstanceOf[NetUtils[T, _]]
  }

  def freezeUpTo(model: NetUtils[T, _], names: JList[String]): Unit = {
    model.freezeUpTo(names.asScala: _*)
  }

  def netLoadBigDL(
          modulePath: String,
          weightPath : String): AbstractModule[Activity, Activity, T] = {
    Net.loadBigDL[T](modulePath, weightPath)
  }

  def netLoadCaffe(
                    defPath: String,
                    modelPath : String): AbstractModule[Activity, Activity, T] = {
    Net.loadCaffe[T](defPath, modelPath)
  }

  def netLoad(
               modulePath: String,
               weightPath : String): AbstractModule[Activity, Activity, T] = {
    Net.load[T](modulePath, weightPath)
  }

  def netLoadTorch(
               path: String): AbstractModule[Activity, Activity, T] = {
    Net.loadTorch[T](path)
  }

  def netLoadTF(path: String, inputs: JList[String], outputs: JList[String],
             byteOrder: String, binFile: String = null): AbstractModule[Activity, Activity, T] = {
    val order = byteOrder match {
      case "little_endian" => ByteOrder.LITTLE_ENDIAN
      case "big_endian" => ByteOrder.BIG_ENDIAN
      case _ => throw new IllegalArgumentException(s"No support byte order $byteOrder")
    }
    Net.loadTF[T](path, inputs.asScala, outputs.asScala, order, Option(binFile))
  }

  def netLoadTF(folder: String): AbstractModule[Activity, Activity, T] = {
    Net.loadTF[T](folder)
  }

  def kerasNetToModel(value: KerasNet[T]): Model[T] = {
    value.toModel()
  }

  def netToKeras(value: NetUtils[T, _]): KerasLayer[Activity, Activity, T] = {
    value.toKeras()
  }

  def zooKerasNetSummary(
      model: KerasNet[T],
      lineLength: Int = 120,
      positions: JList[Double]): Unit = {
    model.summary(lineLength, positions.asScala.toArray)
  }

  def createZooKerasDense(
      outputDim: Int,
      init: String = "glorot_uniform",
      activation: String = null,
      wRegularizer: Regularizer[T] = null,
      bRegularizer: Regularizer[T] = null,
      bias: Boolean = true,
      inputShape: JList[Int] = null): Dense[T] = {
    Dense(outputDim, init, activation, wRegularizer,
      bRegularizer, bias, toScalaShape(inputShape))
  }

  def createZooKerasEmbedding(
      inputDim: Int,
      outputDim: Int,
      init: String = "uniform",
      wRegularizer: Regularizer[T] = null,
      inputShape: JList[Int] = null): Embedding[T] = {
    Embedding[T](inputDim, outputDim, init, wRegularizer, toScalaShape(inputShape))
  }

  def createZooKerasBatchNormalization(
      epsilon: Double = 0.001,
      momentum: Double = 0.99,
      betaInit: String = "zero",
      gammaInit: String = "one",
      dimOrdering: String = "th",
      inputShape: JList[Int] = null): BatchNormalization[T] = {
    BatchNormalization[T](epsilon, momentum, betaInit,
      gammaInit, dimOrdering, toScalaShape(inputShape))
  }

  def createZooKerasConvolution2D(
      nbFilter: Int,
      nbRow: Int,
      nbCol: Int,
      init: String = "glorot_uniform",
      activation: String = null,
      borderMode: String = "valid",
      subsample: JList[Int],
      dimOrdering: String = "th",
      wRegularizer: Regularizer[T] = null,
      bRegularizer: Regularizer[T] = null,
      bias: Boolean = true,
      inputShape: JList[Int] = null,
      pads: JList[Int]): Convolution2D[T] = {
    new Convolution2D(nbFilter, nbRow, nbCol, KerasUtils.getInitMethod(init),
      KerasUtils.getKerasActivation(activation), borderMode,
      toScalaArray(subsample), KerasUtils.toBigDLFormat(dimOrdering),
      wRegularizer, bRegularizer, bias, toScalaShape(inputShape),
      if (pads != null) {
        pads.asScala.toArray
      } else {
        null
      })
  }

  def createZooKerasMaxPooling2D(
      poolSize: JList[Int],
      strides: JList[Int],
      borderMode: String = "valid",
      dimOrdering: String = "th",
      inputShape: JList[Int] = null,
      pads: JList[Int]): MaxPooling2D[T] = {
    new MaxPooling2D[T](toScalaArray(poolSize), toScalaArray(strides),
      borderMode, KerasUtils.toBigDLFormat(dimOrdering), toScalaShape(inputShape),
      if (pads != null) {
        pads.asScala.toArray
      } else {
        null
      })
  }

  def createZooKerasActivation(
      activation: String,
      inputShape: JList[Int] = null): Activation[T] = {
    Activation(activation, toScalaShape(inputShape))
  }

  def createZooKerasReshape(
      targetShape: JList[Int],
      inputShape: JList[Int] = null): Reshape[T] = {
    Reshape(toScalaArray(targetShape), toScalaShape(inputShape))
  }

  def createZooKerasDropout(
      p: Double,
      inputShape: JList[Int] = null): Dropout[T] = {
    Dropout(p, toScalaShape(inputShape))
  }

  def createZooKerasFlatten(
      inputShape: JList[Int] = null): Flatten[T] = {
    Flatten(toScalaShape(inputShape))
  }

  def createZooKerasMerge(
      layers: JList[KerasLayer[Activity, Activity, T]] = null,
      mode: String = "sum",
      concatAxis: Int = -1,
      inputShape: JList[JList[Int]]): Merge[T] = {
    val layersList = if (layers != null) layers.asScala.toList else null
    Merge[T](layersList, mode, concatAxis, toScalaMultiShape(inputShape))
  }

  def createZooKerasConvolution1D(
      nbFilter: Int,
      filterLength: Int,
      init: String = "glorot_uniform",
      activation: String = null,
      borderMode: String = "valid",
      subsampleLength: Int = 1,
      wRegularizer: Regularizer[T] = null,
      bRegularizer: Regularizer[T] = null,
      bias: Boolean = true,
      inputShape: JList[Int] = null): Convolution1D[T] = {
    Convolution1D(nbFilter, filterLength, init, activation, borderMode,
      subsampleLength, wRegularizer, bRegularizer, bias, toScalaShape(inputShape))
  }

  def createZooKerasSimpleRNN(
      outputDim: Int,
      activation: String = "tanh",
      returnSequences: Boolean = false,
      goBackwards: Boolean = false,
      wRegularizer: Regularizer[T] = null,
      uRegularizer: Regularizer[T] = null,
      bRegularizer: Regularizer[T] = null,
      inputShape: JList[Int] = null): SimpleRNN[T] = {
    SimpleRNN(outputDim, activation, returnSequences, goBackwards,
      wRegularizer, uRegularizer, bRegularizer, toScalaShape(inputShape))
  }

  def createZooKerasLSTM(
      outputDim: Int,
      activation: String = "tanh",
      innerActivation: String = "hard_sigmoid",
      returnSequences: Boolean = false,
      goBackwards: Boolean = false,
      wRegularizer: Regularizer[T] = null,
      uRegularizer: Regularizer[T] = null,
      bRegularizer: Regularizer[T] = null,
      inputShape: JList[Int] = null): LSTM[T] = {
    LSTM(outputDim, activation, innerActivation, returnSequences,
      goBackwards, wRegularizer, uRegularizer, bRegularizer, toScalaShape(inputShape))
  }

  def createZooKerasGRU(
      outputDim: Int,
      activation: String = "tanh",
      innerActivation: String = "hard_sigmoid",
      returnSequences: Boolean = false,
      goBackwards: Boolean = false,
      wRegularizer: Regularizer[T] = null,
      uRegularizer: Regularizer[T] = null,
      bRegularizer: Regularizer[T] = null,
      inputShape: JList[Int] = null): GRU[T] = {
    GRU(outputDim, activation, innerActivation, returnSequences,
      goBackwards, wRegularizer, uRegularizer, bRegularizer, toScalaShape(inputShape))
  }

  def createZooKerasHighway(
      activation: String = null,
      wRegularizer: Regularizer[T] = null,
      bRegularizer: Regularizer[T] = null,
      bias: Boolean = true,
      inputShape: JList[Int] = null): Highway[T] = {
    Highway(activation, wRegularizer, bRegularizer, bias, toScalaShape(inputShape))
  }

  def createZooKerasZeroPadding1D(
      padding: JList[Int],
      inputShape: JList[Int] = null): ZeroPadding1D[T] = {
    new ZeroPadding1D(toScalaArray(padding), toScalaShape(inputShape))
  }

  def createZooKerasZeroPadding2D(
      padding: JList[Int],
      dimOrdering: String = "th",
      inputShape: JList[Int] = null): ZeroPadding2D[T] = {
    new ZeroPadding2D(toScalaArray(padding),
      KerasUtils.toBigDLFormat(dimOrdering), toScalaShape(inputShape))
  }

  def createZooKerasUpSampling1D(
      length: Int = 2,
      inputShape: JList[Int] = null): UpSampling1D[T] = {
    UpSampling1D(length, toScalaShape(inputShape))
  }

  def createZooKerasUpSampling2D(
      size: JList[Int],
      dimOrdering: String = "th",
      inputShape: JList[Int] = null): UpSampling2D[T] = {
    new UpSampling2D(toScalaArray(size), KerasUtils.toBigDLFormat(dimOrdering),
      toScalaShape(inputShape))
  }

  def createZooKerasUpSampling3D(
      size: JList[Int],
      dimOrdering: String = "th",
      inputShape: JList[Int] = null): UpSampling3D[T] = {
    new UpSampling3D(toScalaArray(size), KerasUtils.toBigDLFormat5D(dimOrdering),
      toScalaShape(inputShape))
  }

  def createZooKerasMaxoutDense(
      outputDim: Int,
      nbFeature: Int = 4,
      wRegularizer: Regularizer[T] = null,
      bRegularizer: Regularizer[T] = null,
      bias: Boolean = true,
      inputShape: JList[Int] = null): MaxoutDense[T] = {
    MaxoutDense(outputDim, nbFeature, wRegularizer,
      bRegularizer, bias, toScalaShape(inputShape))
  }

  def createZooKerasConvolution3D(
      nbFilter: Int,
      kernelDim1: Int,
      kernelDim2: Int,
      kernelDim3: Int,
      init: String = "glorot_uniform",
      activation: String = null,
      borderMode: String = "valid",
      subsample: JList[Int],
      dimOrdering: String = "th",
      wRegularizer: Regularizer[T] = null,
      bRegularizer: Regularizer[T] = null,
      bias: Boolean = true,
      inputShape: JList[Int] = null): Convolution3D[T] = {
    new Convolution3D(nbFilter, kernelDim1, kernelDim2, kernelDim3,
      KerasUtils.getInitMethod(init), KerasUtils.getKerasActivation(activation),
      borderMode, toScalaArray(subsample), KerasUtils.toBigDLFormat5D(dimOrdering),
      wRegularizer, bRegularizer, bias, toScalaShape(inputShape))
  }

  def createZooKerasMaxPooling1D(
      poolLength: Int = 2,
      stride: Int = -1,
      borderMode: String = "valid",
      inputShape: JList[Int] = null): MaxPooling1D[T] = {
    MaxPooling1D(poolLength, stride, borderMode, toScalaShape(inputShape))
  }

  def createZooKerasMaxPooling3D(
      poolSize: JList[Int],
      strides: JList[Int],
      dimOrdering: String = "th",
      inputShape: JList[Int] = null): MaxPooling3D[T] = {
    new MaxPooling3D(toScalaArray(poolSize), toScalaArray(strides),
      KerasUtils.toBigDLFormat5D(dimOrdering), toScalaShape(inputShape))
  }

  def createZooKerasAveragePooling1D(
      poolLength: Int = 2,
      stride: Int = -1,
      borderMode: String = "valid",
      inputShape: JList[Int] = null): AveragePooling1D[T] = {
    AveragePooling1D(poolLength, stride, borderMode, toScalaShape(inputShape))
  }

  def createZooKerasAveragePooling2D(
      poolSize: JList[Int],
      strides: JList[Int],
      borderMode: String = "valid",
      dimOrdering: String = "th",
      inputShape: JList[Int] = null,
      pads: JList[Int],
      count_include_pad: Boolean = false): AveragePooling2D[T] = {
    new AveragePooling2D(toScalaArray(poolSize), toScalaArray(strides),
      borderMode, KerasUtils.toBigDLFormat(dimOrdering), toScalaShape(inputShape),
      if (pads != null) {
        pads.asScala.toArray
      } else {
        null
      }, count_include_pad)
  }

  def createZooKerasAveragePooling3D(
      poolSize: JList[Int],
      strides: JList[Int],
      dimOrdering: String = "th",
      inputShape: JList[Int] = null): AveragePooling3D[T] = {
    new AveragePooling3D(toScalaArray(poolSize), toScalaArray(strides),
      KerasUtils.toBigDLFormat5D(dimOrdering), toScalaShape(inputShape))
  }

  def createZooKerasGlobalAveragePooling2D(
      dimOrdering: String = "th",
      inputShape: JList[Int] = null): GlobalAveragePooling2D[T] = {
    GlobalAveragePooling2D(dimOrdering, toScalaShape(inputShape))
  }

  def createZooKerasGlobalMaxPooling2D(
      dimOrdering: String = "th",
      inputShape: JList[Int] = null): GlobalMaxPooling2D[T] = {
    GlobalMaxPooling2D(dimOrdering, toScalaShape(inputShape))
  }

  def createZooKerasRepeatVector(
      n: Int,
      inputShape: JList[Int] = null): RepeatVector[T] = {
    RepeatVector(n, toScalaShape(inputShape))
  }

  def createZooKerasPermute(
      dims: JList[Int],
      inputShape: JList[Int] = null): Permute[T] = {
    Permute(toScalaArray(dims), toScalaShape(inputShape))
  }

  def createZooKerasCropping1D(
      cropping: JList[Int],
      inputShape: JList[Int] = null): Cropping1D[T] = {
    new Cropping1D(toScalaArray(cropping), toScalaShape(inputShape))
  }

  def createZooKerasCropping2D(
      heightCrop: JList[Int],
      widthCrop: JList[Int],
      dimOrdering: String = "th",
      inputShape: JList[Int] = null): Cropping2D[T] = {
    new Cropping2D(toScalaArray(heightCrop), toScalaArray(widthCrop),
      KerasUtils.toBigDLFormat(dimOrdering), toScalaShape(inputShape))
  }

  def createZooKerasCropping3D(
      dim1Crop: JList[Int],
      dim2Crop: JList[Int],
      dim3Crop: JList[Int],
      dimOrdering: String = "th",
      inputShape: JList[Int] = null): Cropping3D[T] = {
    new Cropping3D(toScalaArray(dim1Crop), toScalaArray(dim2Crop), toScalaArray(dim3Crop),
      KerasUtils.toBigDLFormat5D(dimOrdering), toScalaShape(inputShape))
  }

  def createZooKerasAtrousConvolution1D(
      nbFilter: Int,
      filterLength: Int,
      init: String = "glorot_uniform",
      activation: String = null,
      subsampleLength: Int = 1,
      atrousRate: Int = 1,
      wRegularizer: Regularizer[T] = null,
      bRegularizer: Regularizer[T] = null,
      inputShape: JList[Int] = null): AtrousConvolution1D[T] = {
    AtrousConvolution1D(nbFilter, filterLength, init, activation,
      subsampleLength, atrousRate, wRegularizer, bRegularizer, toScalaShape(inputShape))
  }

  def createZooKerasAtrousConvolution2D(
      nbFilter: Int,
      nbRow: Int,
      nbCol: Int,
      init: String = "glorot_uniform",
      activation: String = null,
      subsample: JList[Int],
      atrousRate: JList[Int],
      dimOrdering: String = "th",
      wRegularizer: Regularizer[T] = null,
      bRegularizer: Regularizer[T] = null,
      inputShape: JList[Int] = null): AtrousConvolution2D[T] = {
    new AtrousConvolution2D(nbFilter, nbRow, nbCol, KerasUtils.getInitMethod(init),
      KerasUtils.getKerasActivation(activation), toScalaArray(subsample),
      toScalaArray(atrousRate), KerasUtils.toBigDLFormat(dimOrdering),
      wRegularizer, bRegularizer, toScalaShape(inputShape))
  }

  def createZooKerasDeconvolution2D(
      nbFilter: Int,
      nbRow: Int,
      nbCol: Int,
      init: String = "glorot_uniform",
      activation: String = null,
      subsample: JList[Int],
      dimOrdering: String = "th",
      wRegularizer: Regularizer[T] = null,
      bRegularizer: Regularizer[T] = null,
      bias: Boolean = true,
      inputShape: JList[Int] = null): Deconvolution2D[T] = {
    new Deconvolution2D(nbFilter, nbRow, nbCol, KerasUtils.getInitMethod(init),
      KerasUtils.getKerasActivation(activation), toScalaArray(subsample),
      KerasUtils.toBigDLFormat(dimOrdering), wRegularizer, bRegularizer,
      bias, toScalaShape(inputShape))
  }

  def createZooKerasConvLSTM2D(
      nbFilter: Int,
      nbKernel: Int,
      activation: String = "tanh",
      innerActivation: String = "hard_sigmoid",
      dimOrdering: String = "th",
      subsample: Int = 1,
      wRegularizer: Regularizer[T] = null,
      uRegularizer: Regularizer[T] = null,
      bRegularizer: Regularizer[T] = null,
      returnSequences: Boolean = false,
      goBackwards: Boolean = false,
      inputShape: JList[Int] = null): ConvLSTM2D[T] = {
    ConvLSTM2D(nbFilter, nbKernel, activation, innerActivation,
      dimOrdering, subsample, wRegularizer, uRegularizer, bRegularizer,
      returnSequences, goBackwards, toScalaShape(inputShape))
  }

  def createZooKerasLocallyConnected1D(
      nbFilter: Int,
      filterLength: Int,
      activation: String = null,
      subsampleLength: Int = 1,
      wRegularizer: Regularizer[T] = null,
      bRegularizer: Regularizer[T] = null,
      bias: Boolean = true,
      inputShape: JList[Int] = null): LocallyConnected1D[T] = {
    LocallyConnected1D(nbFilter, filterLength, activation, subsampleLength,
      wRegularizer, bRegularizer, bias, toScalaShape(inputShape))
  }

  def createZooKerasLocallyConnected2D(
      nbFilter: Int,
      nbRow: Int,
      nbCol: Int,
      activation: String = null,
      borderMode: String = "valid",
      subsample: JList[Int],
      dimOrdering: String = "th",
      wRegularizer: Regularizer[T] = null,
      bRegularizer: Regularizer[T] = null,
      bias: Boolean = true,
      inputShape: JList[Int] = null): LocallyConnected2D[T] = {
    new LocallyConnected2D(nbFilter, nbRow, nbCol, KerasUtils.getKerasActivation(activation),
      borderMode, toScalaArray(subsample), KerasUtils.toBigDLFormat(dimOrdering),
      wRegularizer, bRegularizer, bias, toScalaShape(inputShape))
  }

  def createZooKerasSeparableConvolution2D(
      nbFilter: Int,
      nbRow: Int,
      nbCol: Int,
      init: String = "glorot_uniform",
      activation: String = null,
      borderMode: String = "valid",
      subsample: JList[Int],
      depthMultiplier: Int = 1,
      dimOrdering: String = "th",
      depthwiseRegularizer: Regularizer[T] = null,
      pointwiseRegularizer: Regularizer[T] = null,
      bRegularizer: Regularizer[T] = null,
      bias: Boolean = true,
      inputShape: JList[Int] = null): SeparableConvolution2D[T] = {
    new SeparableConvolution2D(nbFilter, nbRow, nbCol, KerasUtils.getInitMethod(init),
      KerasUtils.getKerasActivation(activation), borderMode, toScalaArray(subsample),
      depthMultiplier, KerasUtils.toBigDLFormat(dimOrdering),
      depthwiseRegularizer, pointwiseRegularizer, bRegularizer, bias, toScalaShape(inputShape))
  }

  def createZooKerasZeroPadding3D(
      padding: JList[Int],
      dimOrdering: String = "th",
      inputShape: JList[Int] = null): ZeroPadding3D[T] = {
    new ZeroPadding3D(toScalaArray(padding), KerasUtils.toBigDLFormat5D(dimOrdering),
      toScalaShape(inputShape))
  }

  def createZooKerasGlobalAveragePooling1D(
      inputShape: JList[Int] = null): GlobalAveragePooling1D[T] = {
    GlobalAveragePooling1D(toScalaShape(inputShape))
  }

  def createZooKerasGlobalMaxPooling1D(
      inputShape: JList[Int] = null): GlobalMaxPooling1D[T] = {
    GlobalMaxPooling1D(toScalaShape(inputShape))
  }

  def createZooKerasGlobalMaxPooling3D(
      dimOrdering: String = "th",
      inputShape: JList[Int] = null): GlobalMaxPooling3D[T] = {
    GlobalMaxPooling3D(dimOrdering, toScalaShape(inputShape))
  }

  def createZooKerasGlobalAveragePooling3D(
      dimOrdering: String = "th",
      inputShape: JList[Int] = null): GlobalAveragePooling3D[T] = {
    GlobalAveragePooling3D(dimOrdering, toScalaShape(inputShape))
  }

  def createZooKerasSpatialDropout1D(
      p: Double = 0.5,
      inputShape: JList[Int] = null): SpatialDropout1D[T] = {
    SpatialDropout1D(p, toScalaShape(inputShape))
  }

  def createZooKerasSpatialDropout2D(
      p: Double = 0.5,
      dimOrdering: String = "th",
      inputShape: JList[Int] = null): SpatialDropout2D[T] = {
    SpatialDropout2D(p, dimOrdering, toScalaShape(inputShape))
  }

  def createZooKerasSpatialDropout3D(
      p: Double = 0.5,
      dimOrdering: String = "th",
      inputShape: JList[Int] = null): SpatialDropout3D[T] = {
    SpatialDropout3D(p, dimOrdering, toScalaShape(inputShape))
  }

  def createZooKerasGaussianDropout(
      p: Double,
      inputShape: JList[Int] = null): GaussianDropout[T] = {
    GaussianDropout(p, toScalaShape(inputShape))
  }

  def createZooKerasGaussianNoise(
      sigma: Double,
      inputShape: JList[Int] = null): GaussianNoise[T] = {
    GaussianNoise(sigma, toScalaShape(inputShape))
  }

  def createZooKerasMasking(
      maskValue: Double = 0.0,
      inputShape: JList[Int] = null): Masking[T] = {
    Masking(maskValue, toScalaShape(inputShape))
  }

  def createZooKerasSReLU(
      tLeftInit: String = "zero",
      aLeftInit: String = "glorot_uniform",
      tRightInit: String = "glorot_uniform",
      aRightInit: String = "one",
      sharedAxes: JList[Int] = null,
      inputShape: JList[Int] = null): SReLU[T] = {
    SReLU(tLeftInit, aLeftInit, tRightInit, aRightInit,
      toScalaArray(sharedAxes), toScalaShape(inputShape))
  }

  def createZooKerasELU(
      alpha: Double = 1.0,
      inputShape: JList[Int] = null): ELU[T] = {
    ELU(alpha, toScalaShape(inputShape))
  }

  def createZooKerasLeakyReLU(
      alpha: Double = 0.01,
      inputShape: JList[Int] = null): LeakyReLU[T] = {
    LeakyReLU(alpha, toScalaShape(inputShape))
  }

  def createZooKerasThresholdedReLU(
      theta: Double = 1.0,
      inputShape: JList[Int] = null): ThresholdedReLU[T] = {
    ThresholdedReLU(theta, toScalaShape(inputShape))
  }

  def createZooKerasTimeDistributed(
      layer: KerasLayer[Tensor[T], Tensor[T], T],
      inputShape: JList[Int] = null): TimeDistributed[T] = {
    TimeDistributed(layer, toScalaShape(inputShape))
  }

  def createZooKerasBidirectional(
      layer: com.intel.analytics.bigdl.nn.keras.Recurrent[T],
      mergeMode: String = "concat",
      inputShape: JList[Int] = null): Bidirectional[T] = {
    Bidirectional(layer, mergeMode, toScalaShape(inputShape))
  }

  def createZooKerasKerasLayerWrapper(
       torchLayer: AbstractModule[Activity, Activity, T],
       inputShape: JList[Int] = null): KerasLayerWrapper[T] = {
    new KerasLayerWrapper(torchLayer, toScalaShape(inputShape))
  }


  // ================================= Torch layers in Keras style =================================

  def createZooKerasSelect(
      dim: Int,
      index: Int,
      inputShape: JList[Int] = null): Select[T] = {
    Select(dim, index, toScalaShape(inputShape))
  }

  def createZooKerasSparseDense(
      outputDim: Int,
      init: String = "glorot_uniform",
      activation: String = null,
      wRegularizer: Regularizer[T] = null,
      bRegularizer: Regularizer[T] = null,
      backwardStart: Int = -1,
      backwardLength: Int = -1,
      initWeight: JTensor = null,
      initBias: JTensor = null,
      initGradWeight: JTensor = null,
      initGradBias: JTensor = null,
      bias: Boolean = true,
      inputShape: JList[Int] = null): SparseDense[T] = {
    SparseDense(outputDim, init, activation, wRegularizer,
      bRegularizer, backwardStart, backwardLength, toTensor(initWeight),
      toTensor(initBias), toTensor(initGradWeight), toTensor(initGradBias),
      bias, toScalaShape(inputShape))
  }

  def createZooKerasSparseEmbedding(
       inputDim: Int,
       outputDim: Int,
       combiner: String = "sum",
       maxNorm: Double = -1,
       init: String = "uniform",
       wRegularizer: Regularizer[T] = null,
       inputShape: JList[Int] = null): SparseEmbedding[T] = {
    SparseEmbedding(inputDim, outputDim, combiner.toLowerCase,
      maxNorm, init, wRegularizer, toScalaShape(inputShape))
  }

  def createZooKerasNarrow(
      dim: Int,
      offset: Int,
      length: Int = 1,
      inputShape: JList[Int] = null): Narrow[T] = {
    Narrow(dim, offset, length, toScalaShape(inputShape))
  }

  def createZooKerasSqueeze(
      dims: JList[Int],
      inputShape: JList[Int] = null): Squeeze[T] = {
    Squeeze(toScalaArray(dims), toScalaShape(inputShape))
  }

  def createZooKerasAddConstant(
      constant: Double,
      inputShape: JList[Int] = null): AddConstant[T] = {
    AddConstant(constant, toScalaShape(inputShape))
  }

  def createZooKerasMulConstant(
      constant: Double,
      inputShape: JList[Int] = null): MulConstant[T] = {
    MulConstant(constant, toScalaShape(inputShape))
  }

  def createZooKerasLRN2D(
      alpha: Double = 1e-4,
      k: Double = 1.0,
      beta: Double = 0.75,
      n: Int = 5,
      dimOrdering: String = "th",
      inputShape: JList[Int] = null): LRN2D[T] = {
    LRN2D(alpha, k, beta, n, dimOrdering, toScalaShape(inputShape))
  }

  def createZooKerasShareConvolution2D(
      nbFilter: Int,
      nbRow: Int,
      nbCol: Int,
      init: String = "glorot_uniform",
      activation: String = null,
      subsample: JList[Int],
      padH: Int = 0,
      padW: Int = 0,
      propagateBack: Boolean = true,
      dimOrdering: String = "th",
      wRegularizer: Regularizer[T] = null,
      bRegularizer: Regularizer[T] = null,
      bias: Boolean = true,
      inputShape: JList[Int] = null): ShareConvolution2D[T] = {
    new ShareConvolution2D(nbFilter, nbRow, nbCol, KerasUtils.getInitMethod(init),
      KerasUtils.getKerasActivation(activation), toScalaArray(subsample),
      padH, padW, propagateBack, KerasUtils.toBigDLFormat(dimOrdering),
      wRegularizer, bRegularizer, bias, toScalaShape(inputShape))
  }

  def createZooKerasCAdd(
      size: JList[Int],
      bRegularizer: Regularizer[T] = null,
      inputShape: JList[Int] = null): CAdd[T] = {
    CAdd(toScalaArray(size), bRegularizer, toScalaShape(inputShape))
  }

  def createZooKerasCMul(
      size: JList[Int],
      wRegularizer: Regularizer[T] = null,
      inputShape: JList[Int] = null): CMul[T] = {
    CMul(toScalaArray(size), wRegularizer, toScalaShape(inputShape))
  }

  def createZooKerasExp(
      inputShape: JList[Int] = null): Exp[T] = {
    Exp(toScalaShape(inputShape))
  }

  def createZooKerasLog(
      inputShape: JList[Int] = null): Log[T] = {
    Log(toScalaShape(inputShape))
  }

  def createZooKerasMul(
      inputShape: JList[Int] = null): Mul[T] = {
    Mul(toScalaShape(inputShape))
  }

  def createZooKerasPower(
      power: Double,
      scale: Double = 1,
      shift: Double = 0,
      inputShape: JList[Int] = null): Power[T] = {
    Power(power, scale, shift, toScalaShape(inputShape))
  }

  def createZooKerasScale(
      size: JList[Int],
      inputShape: JList[Int] = null): Scale[T] = {
    Scale(toScalaArray(size), toScalaShape(inputShape))
  }

  def createZooKerasSqrt(
      inputShape: JList[Int] = null): Sqrt[T] = {
    Sqrt(toScalaShape(inputShape))
  }

  def createZooKerasSquare(
      inputShape: JList[Int] = null): Square[T] = {
    Square(toScalaShape(inputShape))
  }

  def createAUC(thresholdNum: Int): ValidationMethod[T] = {
    new AUC[T](thresholdNum)
  }

  def createZooKerasHardShrink(
      value: Double = 0.5,
      inputShape: JList[Int] = null): HardShrink[T] = {
    HardShrink(value, toScalaShape(inputShape))
  }

  def createZooKerasHardTanh(
      minValue: Double = -1,
      maxValue: Double = 1,
      inputShape: JList[Int] = null): HardTanh[T] = {
    HardTanh(minValue, maxValue, toScalaShape(inputShape))
  }

  def createZooKerasNegative(
      inputShape: JList[Int] = null): Negative[T] = {
    Negative(toScalaShape(inputShape))
  }

  def createZooKerasPReLU(
      nOutputPlane: Int = 0,
      inputShape: JList[Int] = null): PReLU[T] = {
    PReLU(nOutputPlane, toScalaShape(inputShape))
  }

  def createZooKerasRReLU(
      lower: Double = 1.0/8,
      upper: Double = 1.0/3,
      inputShape: JList[Int] = null): RReLU[T] = {
    RReLU(lower, upper, toScalaShape(inputShape))
  }

  def createZooKerasSoftShrink(
      value: Double = 0.5,
      inputShape: JList[Int] = null): SoftShrink[T] = {
    SoftShrink(value, toScalaShape(inputShape))
  }

  def createZooKerasWithinChannelLRN2D(
      size: Int = 5,
      alpha: Double = 1.0,
      beta: Double = 0.75,
      inputShape: JList[Int] = null): WithinChannelLRN2D[T] = {
    WithinChannelLRN2D(size, alpha, beta, toScalaShape(inputShape))
  }

  def createZooKerasBinaryThreshold(
      th: Double = 1e-6,
      inputShape: JList[Int] = null): BinaryThreshold[T] = {
    BinaryThreshold(th, toScalaShape(inputShape))
  }

  def createZooKerasThreshold(
       th: Double = 1e-6,
       v: Double = 0.0,
       inputShape: JList[Int] = null): Threshold[T] = {
    Threshold(th, v, toScalaShape(inputShape))
  }

  def getSubModules(module: AbstractModule[Activity, Activity, T]):
  JList[AbstractModule[Activity, Activity, T]] = {
    module match {
      case m: KerasNet[T] =>
        m.getSubModules().asJava
      case m: GraphNet[T] =>
        m.getSubModules().asJava
      case m: Container[Activity, Activity, T] =>
        m.modules.asJava
      case _ =>
        throw new IllegalArgumentException(s"module $module does not have submodules")
    }
  }

  def getFlattenSubModules(module: AbstractModule[Activity, Activity, T],
                        includeContainer: Boolean)
  : JList[AbstractModule[Activity, Activity, T]] = {
    val result = ArrayBuffer[AbstractModule[Activity, Activity, T]]()
    doGetFlattenModules(module, includeContainer, result)
    result.toList.asJava
  }

  // TODO: refactor Container and KerasLayer to simplify this logic
  private def hasSubModules(module: AbstractModule[Activity, Activity, T]) = {
    module match {
      case km: KerasModel[T] => true
      case c: Container[_, _, _] => true
      case k: KerasNet[T] => true
      case _ => false
    }
  }

  private def doGetFlattenModules(module: AbstractModule[Activity, Activity, T],
       includeContainer: Boolean,
       result: ArrayBuffer[AbstractModule[Activity, Activity, T]]): Unit = {
    getSubModules(module).asScala.foreach {m =>
      if (hasSubModules(m)) {
        doGetFlattenModules(m.asInstanceOf[Container[Activity, Activity, T]],
          includeContainer,
          result)
      } else {
        result.append(m)
      }
    }
    if (includeContainer) {
      result.append(module)
    }
  }

  def createZooKerasGaussianSampler(
      inputShape: JList[Int] = null): GaussianSampler[T] = {
    GaussianSampler(toScalaShape(inputShape))
  }

  def createZooKerasResizeBilinear(
      outputHeight: Int,
      outputWidth: Int,
      alignCorners: Boolean,
      dimOrdering: String = "th",
      inputShape: JList[Int] = null): ResizeBilinear[T] = {
    ResizeBilinear(outputHeight, outputWidth, alignCorners, dimOrdering)
  }

  def createTFNet(
      path: String,
      inputNames: JList[String],
      outputNames: JList[String]): TFNet = {
    TFNet(path, inputNames.asScala.toArray, outputNames.asScala.toArray)
  }

  def createTFNet(path: String): TFNet = {
    TFNet(path)
  }

  def createTFTrainingHelper(modelPath: String): TFTrainingHelper = {
    TFTrainingHelper(modelPath)
  }

  def createIdentityCriterion(): IdentityCriterion = {
    new IdentityCriterion()
  }

  def createTFValidationMethod(validationMethod: ValidationMethod[Float],
                               outputLength: Int, targetLength: Int): TFValidationMethod = {
    new TFValidationMethod(validationMethod, outputLength, targetLength)
  }

  def connectInputs(module: AbstractModule[Activity, Activity, T],
      x: JList[Variable[T]]): Variable[T] = {
    require(!x.isEmpty, "We don't accept empty inputs")
    Variable(module.inputs(x.asScala.map(_.node): _*))
  }

  def createZooKerasSparseCategoricalCrossEntropy(
      logProbAsInput: Boolean = false,
      zeroBasedLabel: Boolean = true,
      weights: JTensor = null,
      sizeAverage: Boolean = true,
      paddingValue: Int = -1): SparseCategoricalCrossEntropy[T] = {
    SparseCategoricalCrossEntropy(logProbAsInput, zeroBasedLabel,
      if (weights == null) null else toTensor(weights),
      sizeAverage, paddingValue)
  }

  def createZooKerasMeanAbsoluteError(
      sizeAverage: Boolean = true): MeanAbsoluteError[T] = {
    MeanAbsoluteError[T](sizeAverage)
  }

  def createZooKerasMeanSquaredError(
      sizeAverage: Boolean = true): MeanSquaredError[T] = {
    MeanSquaredError[T](sizeAverage)
  }

  def createZooKerasCategoricalCrossEntropy(): CategoricalCrossEntropy[T] = {
    CategoricalCrossEntropy[T]()
  }

  def createZooKerasKullbackLeiblerDivergence(): KullbackLeiblerDivergence[T] = {
    KullbackLeiblerDivergence[T]()
  }

  def createZooKerasPoisson(): Poisson[T] = {
    Poisson[T]()
  }

  def createZooKerasMeanAbsolutePercentageError(): MeanAbsolutePercentageError[T] = {
    MeanAbsolutePercentageError[T]()
  }

  def createZooKerasMeanSquaredLogarithmicError(): MeanSquaredLogarithmicError[T] = {
    MeanSquaredLogarithmicError[T]()
  }

  def createZooKerasCosineProximity(): CosineProximity[T] = {
    CosineProximity[T]()
  }

  def createZooKerasSquaredHinge(
    margin : Double = 1.0, sizeAverage : Boolean = true):
      SquaredHinge[T] = {SquaredHinge[T](margin, sizeAverage)
  }

  def createZooKerasHinge(
    margin: Double = 1.0, sizeAverage: Boolean = true):
      Hinge[T] = {Hinge[T](margin, sizeAverage)
  }

  def createZooKerasBinaryCrossEntropy(
      weights: JTensor = null,
      sizeAverage: Boolean = true
      ): BinaryCrossEntropy[T] = {
    BinaryCrossEntropy[T](
      if (weights == null) null else toTensor(weights),
      sizeAverage)
  }

  def createZooKerasAccuracy(
      zeroBasedLabel: Boolean = true): ValidationMethod[T] = {
    new Accuracy[T](zeroBasedLabel)
  }

  def createZooKerasTop5Accuracy(
      zeroBasedLabel: Boolean = true): ValidationMethod[T] = {
    new Top5Accuracy[T](zeroBasedLabel)
  }

  def createZooKerasWordEmbedding(
      embeddingFile: String,
      wordIndex: JMap[String, Int] = null,
      trainable: Boolean = false,
      inputShape: JList[Int] = null): WordEmbedding[T] = {
    WordEmbedding[T](embeddingFile, if (wordIndex!= null) wordIndex.asScala.toMap else null,
      trainable, inputShape.get(0))
  }

  def wordEmbeddingGetWordIndex(
      embeddingFile: String): JMap[String, Int] = {
    WordEmbedding.getWordIndex(embeddingFile).asJava
  }

  def zooGetWeightsShape(model: AbstractModule[Activity, Activity, T]): JList[JList[Int]] = {
    val weights = model.getWeightsBias()
    if (weights != null) {
      weights.map{w => w.size().toList.asJava}.toList.asJava
    } else {
      null
    }
  }

  def zooSetWeights(model: AbstractModule[Activity, Activity, T],
      weights: JList[JTensor]): Unit = {
    super.setWeights(model, weights)
  }

  def createTFOptimizer(modelPath: String,
                        optimMethod: OptimMethod[Float],
                        x: JavaRDD[Sample],
                        batchSize: Int = 32): TFOptimizer = {
    new TFOptimizer(modelPath, optimMethod,
      toJSample(x).asInstanceOf[RDD[JSample[Float]]], batchSize)
  }

  def trainTFNet(modelPath: String,
                 optimMethod: OptimMethod[Float],
                 x: JavaRDD[Sample],
                 batchSize: Int = 32,
                 endTrigger: Trigger = Trigger.maxEpoch(1)): JList[JTensor] = {
    val (model, meta) = NetUtils.processTFFolder(modelPath)

    val folderPath = Path(modelPath)
    val trainingMetaPath = folderPath / Path("training_meta.json")

    val jsonStr = Source.fromFile(trainingMetaPath.jfile).getLines().mkString
    import org.json4s._
    import org.json4s.jackson.JsonMethods._
    implicit val formats = DefaultFormats

    val trainingMeta = parse(jsonStr).camelizeKeys.extract[TrainMeta]

    val newMeta = Meta(
      (meta.inputNames.toSeq ++: trainingMeta.variables.toSeq).toArray,
      meta.outputNames)
    val graphDef = TFNet.parseGraph(model)
    val tfnet = TFNet(graphDef, model, newMeta, TFNet.defaultSessionConfig.toByteArray())


    val trainer = new TFTrainingHelper(tfnet,
      trainingMeta.inputNames,
      trainingMeta.outputNames,
      trainingMeta.variables,
      trainingMeta.gradVariables)



    import scala.collection.JavaConverters._
    val optimizer = Optimizer(trainer,
      toJSample(x).asInstanceOf[RDD[JSample[Float]]], new IdentityCriterion(), batchSize)

    optimizer.setOptimMethod(optimMethod)
    optimizer.setEndWhen(endTrigger)
    optimizer.optimize()

    trainer.parameters()._1
      .map(t => toJTensor(t.asInstanceOf[Tensor[T]])).toVector.asJava
  }


  def createZooKerasParameter(inputShape: JList[Int],
      initMethod: InitializationMethod, initWeight: JTensor, trainable: Boolean): Parameter[T] = {
    Parameter[T](toScalaShape(inputShape), initMethod, toTensor(initWeight), trainable)
  }

  def getParameterWeight(parameter: Parameter[T]): JTensor = {
    toJTensor(parameter.getWeight())
  }

  def setParameterWeight(parameter: Parameter[T], value: JTensor): Unit = {
    parameter.setWeight(toTensor(value))
  }
}
