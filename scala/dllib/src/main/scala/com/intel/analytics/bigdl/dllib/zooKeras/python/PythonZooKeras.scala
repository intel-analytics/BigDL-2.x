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

import java.util.{List => JList}

import scala.collection.JavaConverters._
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.python.api.PythonBigDLKeras
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.models.{Model, Sequential}

import scala.reflect.ClassTag

object PythonZooKeras {

  def ofFloat(): PythonZooKeras[Float] = new PythonZooKeras[Float]()

  def ofDouble(): PythonZooKeras[Double] = new PythonZooKeras[Double]()
}

class PythonZooKeras[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDLKeras[T] {

  def createZooKerasModel(
      input: JList[ModuleNode[T]],
      output: JList[ModuleNode[T]]): Model[T] = {
    Model[T](input.asScala.toArray, output.asScala.toArray)
  }

  def createZooKerasSequential(): Sequential[T] = {
    Sequential[T]()
  }

  def createZooKerasInput(
      name : String = null,
      inputShape: JList[Int] = null): ModuleNode[T] = {
    Input(name = name, inputShape = toScalaShape(inputShape))
  }

  def createZooKerasInputLayer(
      inputShape: JList[Int] = null): KerasLayer[Activity, Activity, T] = {
    InputLayer(inputShape = toScalaShape(inputShape))
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
      inputShape: JList[Int] = null): Convolution2D[T] = {
    new Convolution2D(nbFilter, nbRow, nbCol, KerasUtils.getInitMethod(init),
      KerasUtils.getKerasActivation(activation), borderMode,
      toScalaArray(subsample), KerasUtils.toBigDLFormat(dimOrdering),
      wRegularizer, bRegularizer, bias, toScalaShape(inputShape))
  }

  def createZooKerasMaxPooling2D(
      poolSize: JList[Int],
      strides: JList[Int],
      borderMode: String = "valid",
      dimOrdering: String = "th",
      inputShape: JList[Int] = null): MaxPooling2D[T] = {
    new MaxPooling2D[T](toScalaArray(poolSize), toScalaArray(strides),
      borderMode, KerasUtils.toBigDLFormat(dimOrdering), toScalaShape(inputShape))
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
      inputShape: JList[Int] = null): AveragePooling2D[T] = {
    new AveragePooling2D(toScalaArray(poolSize), toScalaArray(strides),
      borderMode, KerasUtils.toBigDLFormat(dimOrdering), toScalaShape(inputShape))
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


  // ================================= Torch layers in Keras style =================================

  def createZooKerasSelect(
      dim: Int,
      index: Int,
      inputShape: JList[Int] = null): Select[T] = {
    Select(dim, index, toScalaShape(inputShape))
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

}
