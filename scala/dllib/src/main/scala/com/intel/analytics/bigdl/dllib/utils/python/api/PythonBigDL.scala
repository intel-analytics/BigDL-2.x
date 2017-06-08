/*
 * Copyright 2016 The BigDL Authors.
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

package com.intel.analytics.bigdl.python.api

import java.util.{ArrayList => JArrayList, HashMap => JHashMap, List => JList, Map => JMap}

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{Identity => DIdentity, Sample => JSample, _}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn._
import com.intel.analytics.bigdl.numeric._
import com.intel.analytics.bigdl.optim.{Optimizer, _}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Table, _}
import com.intel.analytics.bigdl.visualization.{Summary, TrainSummary, ValidationSummary}
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.rdd.RDD
import java.lang.{Integer, Boolean => JBoolean}

import com.intel.analytics.bigdl.nn.Graph._

import scala.collection.JavaConverters._
import scala.language.existentials
import scala.reflect.ClassTag

/**
 * [[com.intel.analytics.bigdl.dataset.Sample]] for python.
 * @param features features
 * @param label labels
 * @param featuresShape feature size
 * @param labelShape label size
 * @param bigdlType bigdl numeric type
 */
case class Sample(features: JList[Any],
                  label: JList[Any],
                  featuresShape: JList[Int],
                  labelShape: JList[Int],
                  bigdlType: String)

case class JTensor(storage: JList[Any], shape: JList[Int], bigdlType: String)

/**
 * [[ValidationResult]] for python
 * @param result result
 * @param totalNum total number
 * @param method method name
 */
case class TestResult(val result: Float, totalNum: Int, val method: String)


object PythonBigDL {

  def ofFloat(): PythonBigDL[Float] = new PythonBigDL[Float]()

  def ofDouble(): PythonBigDL[Double] = new PythonBigDL[Double]()

  def getInitMethod(initMethod: String): InitializationMethod = {
    initMethod.toLowerCase() match {
      case "xavier" => Xavier
      case "randomUniform" => RandomUniform
      case "bilinearfiller" => BilinearFiller
      case m: String => throw new IllegalArgumentException(s"Not supported init method: ${m}")
    }
  }
}

/**
 * Implementation of Python API for BigDL
 */
class PythonBigDL[T: ClassTag](implicit ev: TensorNumeric[T]) extends Serializable {

  private val typeName = {
    val cls = implicitly[ClassTag[T]].runtimeClass
    cls.getSimpleName
  }

  def jTensorsToActivity(input: JList[JTensor]): Activity = {
    val inputActivity = input.size() match {
      case 0 => throw new IllegalArgumentException("Invalid input")
      case 1 => toTensor(input.iterator().next())
      case _ =>
        input.asScala.foldLeft(new Table())((t, jtensor) => t.insert(toTensor(jtensor)))
    }
    inputActivity
  }

  def activityToJTensors(outputActivity: Activity): JList[JTensor] = {
    if (outputActivity.isInstanceOf[Tensor[T]]) {
      List(toJTensor(outputActivity.toTensor)).asJava
    } else {
      outputActivity.toTable.getState().toList.map {
        pair => (pair._1.asInstanceOf[Int], toJTensor(pair._2.asInstanceOf[Tensor[T]]))
      }.sortWith(_._1 < _._1).map(pair => pair._2).asJava
    }
  }


  private def toValidationMethod(vMethods: JList[String]): Array[ValidationMethod[T]] = {
    vMethods.toArray.map {
      case "Top1Accuracy" => new Top1Accuracy[T]()
      case "Top5Accuracy" => new Top5Accuracy[T]()
      case "Loss" => new Loss[T]()
      case m: String => throw new RuntimeException(s"not supported validation method: $m")
    }
  }

  private def validationMethodToStr(method: ValidationMethod[T]): String = {
    method match {
      case _: Top1Accuracy[T] => "Top1Accuracy"
      case _: Top5Accuracy[T] => "Top5Accuracy"
      case _: Loss[T] => "loss"
      case _ => throw new RuntimeException(s"not supported validation method: $method")
    }
  }

  def toPySample(sample: JSample[T]): Sample = {
    val featureList = sample.feature().contiguous().storage().toArray[T].toList.asJava
    val labelList = sample.label().contiguous().storage().toArray[T].toList.asJava
    val cls = implicitly[ClassTag[T]].runtimeClass
    Sample(featureList.asInstanceOf[JList[Any]],
      labelList.asInstanceOf[JList[Any]],
      sample.feature().size().toList.asJava,
      sample.label().size().toList.asJava,
      cls.getSimpleName)
  }

  def toTensor(jTensor: JTensor): Tensor[T] = {
    Tensor(jTensor.storage.asScala.map(_.asInstanceOf[T]).toArray,
      jTensor.shape.asScala.toArray)
  }

  def toJTensor(tensor: Tensor[T]): JTensor = {
    // clone here in case the the size of storage larger then the size of tensor.
    val cloneTensor = tensor.clone()
    JTensor(cloneTensor.storage().toList.map(_.asInstanceOf[Any]).asJava,
      cloneTensor.size().toList.asJava, typeName)
  }

  def testTensor(jTensor: JTensor): JTensor = {
    val tensor = toTensor(jTensor)
    toJTensor(tensor)
  }

  def toSample(record: Sample): JSample[T] = {
    require(record.bigdlType == this.typeName,
      s"record.bigdlType: ${record.bigdlType} == this.typeName: ${this.typeName}")
    val sample = this.typeName match {
      case "float" =>
        val feature = Tensor[Float](Storage[Float](
          record.features.asInstanceOf[JList[Double]].asScala.map(_.toFloat).toArray[Float]),
          1,
          record.featuresShape.asScala.toArray[Int])
        val label = Tensor[Float](
          Storage(record.label.asInstanceOf[JList[Double]].asScala.map(_.toFloat).toArray[Float]),
          1,
          record.labelShape.asScala.toArray[Int])
        JSample[Float](feature, label)
      case "double" =>
        val feature = Tensor[Double](Storage[Double](
          record.features.asInstanceOf[JList[Double]].asScala.toArray[Double]),
          1,
          record.featuresShape.asScala.toArray[Int])
        val label = Tensor[Double](
          Storage(record.label.asInstanceOf[JList[Double]].asScala.toArray[Double]),
          1,
          record.labelShape.asScala.toArray[Int])
        JSample[Double](feature, label)
      case t: String =>
        throw new IllegalArgumentException(s"Not supported type: ${t}")
    }
    sample.asInstanceOf[JSample[T]]
  }

  private def batching(rdd: RDD[Sample], batchSize: Int)
  : DistributedDataSet[MiniBatch[T]] = {
    val recordRDD = rdd.map(toSample(_))
    (DataSet.rdd(recordRDD) -> new SampleToBatch[T](batchSize))
      .asInstanceOf[DistributedDataSet[MiniBatch[T]]]
  }

  def createSequential(): Sequential[T] = {
    Sequential[T]()
  }

  def createLinear(inputSize: Int, outputSize: Int,
                   initMethod: String, withBias: Boolean): Linear[T] = {
    val l = Linear[T](inputSize, outputSize, withBias)
    if (initMethod != "default") {
      l.setInitMethod(weightInitMethod = PythonBigDL.getInitMethod(initMethod),
      biasInitMethod = Zeros)
    }
    l
  }

  def createReLU(ip: Boolean = false): ReLU[T] = {
    ReLU[T](ip)
  }

  def createTanh(): Tanh[T] = {
    Tanh[T]()
  }

  def createTimeDistributed(layer: TensorModule[T]): TimeDistributed[T] = {
    TimeDistributed[T](layer)
  }

  def createRnnCell(inputSize: Int,
                    hiddenSize: Int,
                    activation: TensorModule[T]): RnnCell[T] = {
    RnnCell[T](inputSize, hiddenSize, activation)
  }

  def createTimeDistributedCriterion(critrn: TensorCriterion[T],
                                     sizeAverage: Boolean = false): TimeDistributedCriterion[T] = {
    TimeDistributedCriterion[T](critrn, sizeAverage)
  }

  def createGRU(
    inputSize: Int,
    outputSize: Int,
    p: Double = 0): GRU[T] = {
    GRU[T](inputSize, outputSize, p)
  }

  def createLSTM(
    inputSize: Int,
    hiddenSize: Int,
    p: Double = 0): LSTM[T] = {
    LSTM[T](inputSize, hiddenSize, p)
  }

  def createLSTMPeephole(
    inputSize: Int,
    hiddenSize: Int,
    p: Double = 0): LSTMPeephole[T] = {
    LSTMPeephole[T](inputSize, hiddenSize, p)
  }

  def createRecurrent(): Recurrent[T] = {
    Recurrent[T]()
  }

  def createEcho(): Echo[T] = {
    Echo[T]()
  }

  def createLogSoftMax(): LogSoftMax[T] = {
    LogSoftMax[T]()
  }

  def createSpatialMaxPooling(kW: Int,
                              kH: Int,
                              dW: Int,
                              dH: Int,
                              padW: Int = 0,
                              padH: Int = 0,
                              ceilMode: Boolean = false)
  : SpatialMaxPooling[T] = {
    val maxpooling = SpatialMaxPooling[T](kW,
      kH,
      dW,
      dH,
      padW,
      padH)
    if (ceilMode) maxpooling.ceil()
    else maxpooling
  }

  def createSpatialConvolution(nInputPlane: Int,
                               nOutputPlane: Int,
                               kernelW: Int,
                               kernelH: Int,
                               strideW: Int = 1,
                               strideH: Int = 1,
                               padW: Int = 0,
                               padH: Int = 0,
                               nGroup: Int = 1,
                               propagateBack: Boolean = true,
                               initMethod: String = "default")
  : SpatialConvolution[T] = {
    val s = SpatialConvolution[T](nInputPlane,
      nOutputPlane,
      kernelW,
      kernelH,
      strideW,
      strideH,
      padW,
      padH,
      nGroup,
      propagateBack)
    if (initMethod != "default") {
      s.setInitMethod(weightInitMethod = PythonBigDL.getInitMethod(initMethod),
        biasInitMethod = Zeros)
    }
    s
  }

  def createReshape(size: JList[Int], batchMode: JBoolean = null): Reshape[T] = {
    val mappedBatchMode = batchMode match {
      case JBoolean.TRUE => Some(true)
      case JBoolean.FALSE => Some(false)
      case _ => None
    }
    Reshape(size.asScala.toArray, mappedBatchMode)
  }

  def createConcat(dimension: Int): Concat[T] = {
    Concat[T](dimension)
  }

  def createSpatialAveragePooling(kW: Int,
                                  kH: Int,
                                  dW: Int = 1,
                                  dH: Int = 1,
                                  padW: Int = 0,
                                  padH: Int = 0,
                                  ceilMode: Boolean = false,
                                  countIncludePad: Boolean = true,
                                  divide: Boolean = true)
  : SpatialAveragePooling[T] = {
    SpatialAveragePooling[T](kW, kH, dW, dH, padW, padH, ceilMode, countIncludePad, divide)
  }

  def createSpatialBatchNormalization(nOutput: Int,
                                      eps: Double = 1e-5,
                                      momentum: Double = 0.1,
                                      affine: Boolean = true)
  : SpatialBatchNormalization[T] = {
    SpatialBatchNormalization[T](nOutput, eps, momentum, affine)
  }

  def createSpatialCrossMapLRN(size: Int = 5,
                               alpha: Double = 1.0,
                               beta: Double = 0.75,
                               k: Double = 1.0)
  : SpatialCrossMapLRN[T] = {
    SpatialCrossMapLRN[T](size, alpha, beta, k)
  }

  def createDropout(initP: Double = 0.5,
                    inplace: Boolean = false,
                    scale: Boolean = true)
  : Dropout[T] = {
    Dropout[T](initP, inplace, scale)
  }

  def createView(sizes: JList[Int], num_input_dims: Int = 0): View[T] = {
    View[T](sizes.asScala.toArray).setNumInputDims(num_input_dims)
  }

  def createAbs()
  : Abs[T] = {
    Abs[T]()
  }

  def createAdd(inputSize: Int)
  : Add[T] = {
    Add[T](inputSize)
  }

  def createAddConstant(constant_scalar: Double,
                        inplace: Boolean = false)
  : AddConstant[T] = {
    AddConstant[T](constant_scalar,
      inplace)
  }


  def createBatchNormalization(nOutput: Int,
                               eps: Double = 1e-5,
                               momentum: Double = 0.1,
                               affine: Boolean = true)
  : BatchNormalization[T] = {
    BatchNormalization[T](nOutput,
      eps,
      momentum,
      affine)
  }

  def createBilinear(inputSize1: Int,
                     inputSize2: Int,
                     outputSize: Int,
                     biasRes: Boolean = true)
  : Bilinear[T] = {
    Bilinear[T](inputSize1,
      inputSize2,
      outputSize,
      biasRes)
  }

  def createBottle(module: AbstractModule[Activity, Activity, T],
                   nInputDim: Int = 2,
                   nOutputDim1: Int = Int.MaxValue)
  : Bottle[T] = {
    Bottle[T](module,
      nInputDim,
      nOutputDim1)
  }

  def createCAdd(size: JList[Int])
  : CAdd[T] = {
    CAdd[T](size.asScala.toArray)
  }

  def createCAddTable(inplace: Boolean = false)
  : CAddTable[T] = {
    CAddTable[T](inplace)
  }

  def createCDivTable()
  : CDivTable[T] = {
    CDivTable[T]()
  }

  def createCMaxTable()
  : CMaxTable[T] = {
    CMaxTable[T]()
  }

  def createCMinTable()
  : CMinTable[T] = {
    CMinTable[T]()
  }

  def createCMul(size: JList[Int])
  : CMul[T] = {
    CMul[T](size.asScala.toArray)
  }

  def createCMulTable()
  : CMulTable[T] = {
    CMulTable[T]()
  }

  def createCSubTable()
  : CSubTable[T] = {
    CSubTable[T]()
  }

  def createClamp(min: Int,
                  max: Int)
  : Clamp[T] = {
    Clamp[T](min,
      max)
  }

  def createContiguous()
  : Contiguous[T] = {
    Contiguous[T]()
  }

  def createCosine(inputSize: Int,
                   outputSize: Int)
  : Cosine[T] = {
    Cosine[T](inputSize,
      outputSize)
  }

  def createCosineDistance()
  : CosineDistance[T] = {
    CosineDistance[T]()
  }

  def createDiceCoefficientCriterion(sizeAverage: Boolean = true,
                                     epsilon: Float = 1.0f)
  : DiceCoefficientCriterion[T] = {
    DiceCoefficientCriterion[T](sizeAverage, epsilon)
  }

  def createDotProduct()
  : DotProduct[T] = {
    DotProduct[T]()
  }

  def createELU(alpha: Double = 1.0,
                inplace: Boolean = false)
  : ELU[T] = {
    ELU[T](alpha,
      inplace)
  }

  def createEuclidean(inputSize: Int,
                      outputSize: Int,
                      fastBackward: Boolean = true)
  : Euclidean[T] = {
    Euclidean[T](inputSize,
      outputSize,
      fastBackward)
  }

  def createExp()
  : Exp[T] = {
    Exp[T]()
  }

  def createFlattenTable()
  : FlattenTable[T] = {
    FlattenTable[T]()
  }

  def createGradientReversal(lambda: Double = 1)
  : GradientReversal[T] = {
    GradientReversal[T](lambda)
  }

  def createHardShrink(lambda: Double = 0.5)
  : HardShrink[T] = {
    HardShrink[T](lambda)
  }

  def createHardTanh(minValue: Double = -1,
                     maxValue: Double = 1,
                     inplace: Boolean = false)
  : HardTanh[T] = {
    HardTanh[T](minValue,
      maxValue,
      inplace)
  }

  def createIndex(dimension: Int)
  : Index[T] = {
    Index[T](dimension)
  }

  def createInferReshape(size: JList[Int], batchMode: Boolean = false)
  : InferReshape[T] = {
    InferReshape[T](size.asScala.toArray,
      batchMode)
  }

  def createJoinTable(dimension: Int,
                      nInputDims: Int)
  : JoinTable[T] = {
    JoinTable[T](dimension,
      nInputDims)
  }

  def createL1Cost()
  : L1Cost[T] = {
    L1Cost[T]()
  }

  def createL1Penalty(l1weight: Int,
                      sizeAverage: Boolean = false,
                      provideOutput: Boolean = true)
  : L1Penalty[T] = {
    L1Penalty[T](l1weight,
      sizeAverage,
      provideOutput)
  }

  def createLeakyReLU(negval: Double = 0.01,
                      inplace: Boolean = false)
  : LeakyReLU[T] = {
    LeakyReLU[T](negval,
      inplace)
  }

  def createLog()
  : Log[T] = {
    Log[T]()
  }

  def createLogSigmoid()
  : LogSigmoid[T] = {
    LogSigmoid[T]()
  }

  def createLookupTable(nIndex: Int, nOutput: Int,
                        paddingValue: Double = 0, maxNorm: Double = Double.MaxValue,
                        normType: Double = 2.0, shouldScaleGradByFreq: Boolean = false)
  : LookupTable[T] = {
    LookupTable[T](nIndex,
      nOutput,
      paddingValue,
      maxNorm,
      normType,
      shouldScaleGradByFreq)
  }

  def createMM(transA: Boolean = false,
               transB: Boolean = false)
  : MM[T] = {
    MM[T](transA,
      transB)
  }

  def createMV(trans: Boolean = false)
  : MV[T] = {
    MV[T](trans)
  }

  def createMapTable(module: AbstractModule[Activity, Activity, T] = null)
  : MapTable[T] = {
    MapTable[T](module)
  }

  def createMaskedSelect()
  : MaskedSelect[T] = {
    MaskedSelect[T]()
  }

  def createMax(dim: Int = 1,
                numInputDims: Int = Int.MinValue)
  : Max[T] = {
    Max[T](dim,
      numInputDims)
  }

  def createMean(dimension: Int = 1,
                 nInputDims: Int = -1)
  : Mean[T] = {
    Mean[T](dimension,
      nInputDims)
  }

  def createMin(dim: Int = 1,
                numInputDims: Int = Int.MinValue)
  : Min[T] = {
    Min[T](dim,
      numInputDims)
  }

  def createMixtureTable(dim: Int = Int.MaxValue)
  : MixtureTable[T] = {
    MixtureTable[T](dim)
  }

  def createMul()
  : Mul[T] = {
    Mul[T]()
  }

  def createMulConstant(scalar: Double,
                        inplace: Boolean = false)
  : MulConstant[T] = {
    MulConstant[T](scalar,
      inplace)
  }

  def createNarrow(dimension: Int,
                   offset: Int,
                   length: Int = 1)
  : Narrow[T] = {
    Narrow[T](dimension,
      offset,
      length)
  }

  def createNarrowTable(offset: Int,
                        length: Int = 1)
  : NarrowTable[T] = {
    NarrowTable[T](offset,
      length)
  }

  def createNormalize(p: Double,
                      eps: Double = 1e-10)
  : Normalize[T] = {
    Normalize[T](p,
      eps)
  }

  def createPReLU(nOutputPlane: Int = 0)
  : PReLU[T] = {
    PReLU[T](nOutputPlane)
  }

  def createPadding(dim: Int,
                    pad: Int,
                    nInputDim: Int,
                    value: Double = 0.0,
                    nIndex: Int = 1)
  : Padding[T] = {
    Padding[T](dim,
      pad,
      nInputDim,
      value,
      nIndex)
  }

  def createPairwiseDistance(norm: Int = 2)
  : PairwiseDistance[T] = {
    PairwiseDistance[T](norm)
  }

  def createParallelTable()
  : ParallelTable[T] = {
    ParallelTable[T]()
  }

  def createPower(power: Double,
                  scale: Double = 1,
                  shift: Double = 0)
  : Power[T] = {
    Power[T](power,
      scale,
      shift)
  }

  def createRReLU(lower: Double = 1.0 / 8,
                  upper: Double = 1.0 / 3,
                  inplace: Boolean = false)
  : RReLU[T] = {
    RReLU[T](lower,
      upper,
      inplace)
  }

  def createReLU6(inplace: Boolean = false)
  : ReLU6[T] = {
    ReLU6[T](inplace)
  }

  def createReplicate(nFeatures: Int,
                      dim: Int = 1,
                      nDim: Int = Int.MaxValue)
  : Replicate[T] = {
    Replicate[T](nFeatures,
      dim,
      nDim)
  }

  def createRoiPooling(pooled_w: Int, pooled_h: Int, spatial_scale: T)
  : RoiPooling[T] = {
    RoiPooling[T](pooled_w,
      pooled_h,
      spatial_scale)
  }

  def createScale(size: JList[Int])
  : Scale[T] = {
    Scale[T](size.asScala.toArray)
  }

  def createSelect(dimension: Int,
                   index: Int)
  : Select[T] = {
    Select[T](dimension,
      index)
  }

  def createSelectTable(dimension: Int)
  : SelectTable[T] = {
    SelectTable[T](dimension)
  }

  def createSigmoid()
  : Sigmoid[T] = {
    Sigmoid[T]()
  }

  def createSoftMax()
  : SoftMax[T] = {
    SoftMax[T]()
  }

  def createSoftMin()
  : SoftMin[T] = {
    SoftMin[T]()
  }

  def createSoftPlus(beta: Double = 1.0)
  : SoftPlus[T] = {
    SoftPlus[T](beta)
  }

  def createSoftShrink(lambda: Double = 0.5)
  : SoftShrink[T] = {
    SoftShrink[T](lambda)
  }

  def createSoftSign()
  : SoftSign[T] = {
    SoftSign[T]()
  }

  def createSpatialDilatedConvolution(nInputPlane: Int,
                                      nOutputPlane: Int,
                                      kW: Int,
                                      kH: Int,
                                      dW: Int = 1,
                                      dH: Int = 1,
                                      padW: Int = 0,
                                      padH: Int = 0,
                                      dilationW: Int = 1,
                                      dilationH: Int = 1,
                                      initMethod: String = "default")
  : SpatialDilatedConvolution[T] = {
    val s = SpatialDilatedConvolution[T](nInputPlane,
      nOutputPlane,
      kW,
      kH,
      dW,
      dH,
      padW,
      padH,
      dilationW,
      dilationH)
    if (initMethod != "default") {
      s.setInitMethod(weightInitMethod = PythonBigDL.getInitMethod(initMethod),
        biasInitMethod = Zeros)
    }
    s
  }

  def createSpatialFullConvolution(nInputPlane: Int,
                                   nOutputPlane: Int,
                                   kW: Int,
                                   kH: Int,
                                   dW: Int = 1,
                                   dH: Int = 1,
                                   padW: Int = 0,
                                   padH: Int = 0,
                                   adjW: Int = 0,
                                   adjH: Int = 0,
                                   nGroup: Int = 1,
                                   noBias: Boolean = false,
                                   initMethod: String = "default")
  : SpatialFullConvolution[Activity, T] = {
    val s = SpatialFullConvolution[Activity, T](nInputPlane,
      nOutputPlane,
      kW,
      kH,
      dW,
      dH,
      padW,
      padH,
      adjW,
      adjH,
      nGroup,
      noBias)
    if (initMethod != "default") {
      s.setInitMethod(weightInitMethod = PythonBigDL.getInitMethod(initMethod),
        biasInitMethod = Zeros)
    }
    s
  }

  def createSpatialShareConvolution(nInputPlane: Int,
                                    nOutputPlane: Int,
                                    kernelW: Int,
                                    kernelH: Int,
                                    strideW: Int = 1,
                                    strideH: Int = 1,
                                    padW: Int = 0,
                                    padH: Int = 0,
                                    nGroup: Int = 1,
                                    propagateBack: Boolean = true,
                                    initMethod: String = "default")
  : SpatialShareConvolution[T] = {
    val s = SpatialShareConvolution[T](nInputPlane,
      nOutputPlane,
      kernelW,
      kernelH,
      strideW,
      strideH,
      padW,
      padH,
      nGroup,
      propagateBack)
    if (initMethod != "default") {
      s.setInitMethod(weightInitMethod = PythonBigDL.getInitMethod(initMethod),
        biasInitMethod = Zeros)
    }
    s
  }

  def createSpatialZeroPadding(padLeft: Int,
                               padRight: Int,
                               padTop: Int,
                               padBottom: Int)
  : SpatialZeroPadding[T] = {
    SpatialZeroPadding[T](padLeft,
      padRight,
      padTop,
      padBottom)
  }

  def createSplitTable(dimension: Int,
                       nInputDims: Int = -1)
  : SplitTable[T] = {
    SplitTable[T](dimension,
      nInputDims)
  }

  def createSqrt()
  : Sqrt[T] = {
    Sqrt[T]()
  }

  def createSquare()
  : Square[T] = {
    Square[T]()
  }

  def createSqueeze(dim: Int = Int.MinValue,
                    numInputDims: Int = Int.MinValue)
  : Squeeze[T] = {
    Squeeze[T](dim,
      numInputDims)
  }

  def createSum(dimension: Int = 1,
                nInputDims: Int = -1,
                sizeAverage: Boolean = false)
  : Sum[T] = {
    Sum[T](dimension,
      nInputDims,
      sizeAverage)
  }

  def createTanhShrink()
  : TanhShrink[T] = {
    TanhShrink[T]()
  }

  def createThreshold(th: Double = 1e-6,
                      v: Double = 0.0,
                      ip: Boolean = false)
  : Threshold[T] = {
    Threshold[T](th,
      v,
      ip)
  }

  def createUnsqueeze(pos: Int,
                      numInputDims: Int = Int.MinValue)
  : Unsqueeze[T] = {
    Unsqueeze[T](pos,
      numInputDims)
  }

  def createBCECriterion(weights: JTensor = null,
                         sizeAverage: Boolean = true)
  : BCECriterion[T] = {
    BCECriterion[T](if (weights == null) null else toTensor(weights),
      sizeAverage)
  }

  def createBiRecurrent(merge: AbstractModule[Table, Tensor[T], T] = null)
  : BiRecurrent[T] = {
    BiRecurrent[T](merge)
  }

  def createConcatTable()
  : ConcatTable[T] = {
    ConcatTable[Activity, T]()
  }

  def createIdentity()
  : Identity[T] = {
    Identity[T]()
  }

  def createMultiLabelSoftMarginCriterion(weights: JTensor = null,
                                          sizeAverage: Boolean = true)
  : MultiLabelSoftMarginCriterion[T] = {
    MultiLabelSoftMarginCriterion[T](if (weights == null) null else toTensor(weights),
      sizeAverage)
  }

  def createMultiMarginCriterion(p: Int = 1,
                                 weights: JTensor = null,
                                 margin: Double = 1.0,
                                 sizeAverage: Boolean = true)
  : MultiMarginCriterion[T] = {
    MultiMarginCriterion[T](p,
      if (weights == null) null else toTensor(weights),
      margin,
      sizeAverage)
  }

  def createReverse(dimension: Int = 1)
  : Reverse[T] = {
    Reverse[T](dimension)
  }

  def createTranspose(permutations: JList[JList[Int]])
  : Transpose[T] = {
    Transpose[T](permutations.asScala.toArray.map { item =>
      val itemArray = item.asScala.toArray
      (itemArray(0), itemArray(1))
    })
  }

  def createSpatialContrastiveNormalization(nInputPlane: Int = 1,
                                            kernel: JTensor = null,
                                            threshold: Double = 1e-4,
                                            thresval: Double = 1e-4)
  : SpatialContrastiveNormalization[T] = {
    SpatialContrastiveNormalization[T](nInputPlane,
      if (kernel == null) null else toTensor(kernel),
      threshold,
      thresval)
  }

  def createSpatialConvolutionMap(connTable: JTensor,
                                  kW: Int,
                                  kH: Int,
                                  dW: Int = 1,
                                  dH: Int = 1,
                                  padW: Int = 0,
                                  padH: Int = 0)
  : SpatialConvolutionMap[T] = {
    SpatialConvolutionMap[T](if (connTable == null) null else toTensor(connTable),
      kW,
      kH,
      dW,
      dH,
      padW,
      padH)
  }

  def createVolumetricConvolution(nInputPlane: Int,
                                  nOutputPlane: Int,
                                  kT: Int,
                                  kW: Int,
                                  kH: Int,
                                  dT: Int = 1,
                                  dW: Int = 1,
                                  dH: Int = 1,
                                  padT: Int = 0,
                                  padW: Int = 0,
                                  padH: Int = 0,
                                  withBias: Boolean = true,
                                  initMethod: String = "default")
  : VolumetricConvolution[T] = {
    val v = VolumetricConvolution[T](nInputPlane,
      nOutputPlane,
      kT,
      kW,
      kH,
      dT,
      dW,
      dH,
      padT,
      padW,
      padH,
      withBias)
    if (initMethod != "default") {
      v.setInitMethod(weightInitMethod = PythonBigDL.getInitMethod(initMethod),
        biasInitMethod = Zeros)
    }
    v
  }

  def createVolumetricMaxPooling(kT: Int,
    kW: Int,
    kH: Int,
    dT: Int,
    dW: Int,
    dH: Int,
    padT: Int = 0,
    padW: Int = 0,
    padH: Int = 0): VolumetricMaxPooling[T] = {
    VolumetricMaxPooling[T](kT, kW, kH, dT, dW, dH, padT, padW, padH)
  }

  def createSpatialDivisiveNormalization(nInputPlane: Int = 1,
                                         kernel: JTensor = null,
                                         threshold: Double = 1e-4,
                                         thresval: Double = 1e-4)
  : SpatialDivisiveNormalization[T] = {
    SpatialDivisiveNormalization[T](nInputPlane,
      if (kernel == null) null else toTensor(kernel),
      threshold,
      thresval)
  }

  def createSpatialSubtractiveNormalization(nInputPlane: Int = 1,
                                            kernel: JTensor = null)
  : SpatialSubtractiveNormalization[T] = {
    SpatialSubtractiveNormalization[T](nInputPlane,
      if (kernel == null) null else toTensor(kernel))
  }

  def createSoftMarginCriterion(sizeAverage: Boolean = true)
  : SoftMarginCriterion[T] = {
    SoftMarginCriterion[T](sizeAverage)
  }

  //   Optimizer
  def createPoly(power: Double, maxIteration: Int): SGD.Poly = {
    SGD.Poly(power, maxIteration)
  }

  def createStep(stepSize: Int, gamma: Double): SGD.Step = {
    SGD.Step(stepSize, gamma)
  }

  def createMultiStep(stepSizes: JList[Int], gamma: Double): SGD.MultiStep = {
    SGD.MultiStep(stepSizes.asScala.toArray, gamma)
  }

  def createDefault(): SGD.Default = {
    SGD.Default()
  }

  def createClassNLLCriterion(weights: JTensor = null,
                              sizeAverage: Boolean = true)
  : ClassNLLCriterion[T] = {
    ClassNLLCriterion[T](if (weights == null) null else toTensor(weights),
      sizeAverage)
  }

  def createMSECriterion: MSECriterion[T] = {
    MSECriterion[T]()
  }

  def createAbsCriterion(sizeAverage: Boolean = true)
  : AbsCriterion[T] = {
    AbsCriterion[T](sizeAverage)
  }

  def createClassSimplexCriterion(nClasses: Int)
  : ClassSimplexCriterion[T] = {
    ClassSimplexCriterion[T](nClasses)
  }

  def createCrossEntropyCriterion(weights: JTensor = null,
                                  sizeAverage: Boolean = true): CrossEntropyCriterion[T] = {
    new CrossEntropyCriterion[T](if (null == weights) null else toTensor(weights), sizeAverage)
  }


  def createCosineEmbeddingCriterion(margin: Double = 0.0,
                                     sizeAverage: Boolean = true)
  : CosineEmbeddingCriterion[T] = {
    CosineEmbeddingCriterion[T](margin,
      sizeAverage)
  }

  def createDistKLDivCriterion(sizeAverage: Boolean = true)
  : DistKLDivCriterion[T] = {
    DistKLDivCriterion[T](sizeAverage)
  }

  def createHingeEmbeddingCriterion(margin: Double = 1,
                                    sizeAverage: Boolean = true)
  : HingeEmbeddingCriterion[T] = {
    HingeEmbeddingCriterion[T](margin,
      sizeAverage)
  }

  def createL1HingeEmbeddingCriterion(margin: Double = 1)
  : L1HingeEmbeddingCriterion[T] = {
    L1HingeEmbeddingCriterion[T](margin)
  }

  def createMarginCriterion(margin: Double = 1.0,
                            sizeAverage: Boolean = true)
  : MarginCriterion[T] = {
    MarginCriterion[T](margin,
      sizeAverage)
  }

  def createMarginRankingCriterion(margin: Double = 1.0,
                                   sizeAverage: Boolean = true)
  : MarginRankingCriterion[T] = {
    MarginRankingCriterion[T](margin,
      sizeAverage)
  }

  def createMultiCriterion()
  : MultiCriterion[T] = {
    MultiCriterion[T]()
  }

  def createMultiLabelMarginCriterion(sizeAverage: Boolean = true)
  : MultiLabelMarginCriterion[T] = {
    MultiLabelMarginCriterion[T](sizeAverage)
  }

  def createParallelCriterion(repeatTarget: Boolean = false)
  : ParallelCriterion[T] = {
    ParallelCriterion[T](repeatTarget)
  }

  def createSmoothL1Criterion(sizeAverage: Boolean = true)
  : SmoothL1Criterion[T] = {
    SmoothL1Criterion[T](sizeAverage)
  }

  def createSmoothL1CriterionWithWeights(sigma: Double, num: Int = 0)
  : SmoothL1CriterionWithWeights[T] = {
    SmoothL1CriterionWithWeights[T](sigma,
      num)
  }

  def createSoftmaxWithCriterion(ignoreLabel: Integer = null,
                                 normalizeMode: String = "VALID")
  : SoftmaxWithCriterion[T] = {
    val normM = normalizeMode match {
      case "FULL" => NormMode.FULL
      case "VALID" => NormMode.VALID
      case "BATCH_SIZE" => NormMode.BATCH_SIZE
      case "NONE" => NormMode.NONE
      case n: String =>
        throw new IllegalArgumentException(s"Only support 'FULL', " +
          s"'VALID', 'BATCH_SIZE' and 'NONE': $n")
    }
    val labelToIgnore = ignoreLabel match {
      case i: Integer => Some(i.toInt)
      case null => None
    }
    SoftmaxWithCriterion[T](labelToIgnore, normM)
  }

  def createConst(value: JTensor): Const[T] = {
    Const(toTensor(value))
  }

  def createFill(value: Double): Fill[T] = {
    Fill(value)
  }

  def createPack(dimension: Int): Pack[T] = {
    Pack(dimension)
  }

  def createShape(): Shape[T] = {
    Shape()
  }

  def createSplitAndSelect(dimension: Int, index: Int, numSplit: Int): SplitAndSelect[T] = {
    SplitAndSelect(dimension, index, numSplit)
  }

  def setModelSeed(seed: Long): Unit = {
    RandomGenerator.RNG.setSeed(seed)
  }

  def modelTest(model: AbstractModule[Activity, Activity, T],
                valRDD: JavaRDD[Sample],
                batchSize: Int,
                valMethods: JList[String])
  : JList[TestResult] = {
    val resultArray = model.evaluate(valRDD.rdd.map(toSample(_)),
      toValidationMethod(valMethods), Some(batchSize))
    val testResultArray = resultArray.map { result =>
      TestResult(result._1.result()._1, result._1.result()._2,
        validationMethodToStr(result._2))
    }
    testResultArray.toList.asJava
  }

  def loadBigDL(path: String): AbstractModule[Activity, Activity, T] = {
    Module.load[T](path)
  }

  def loadTorch(path: String): AbstractModule[Activity, Activity, T] = {
    Module.loadTorch[T](path)
  }

  def loadCaffe(model: AbstractModule[Activity, Activity, T],
                defPath: String,
                modelPath: String,
                matchAll: Boolean = true): AbstractModule[Activity, Activity, T] = {
    Module.loadCaffe[T](model, defPath, modelPath, matchAll)
  }

  def modelPredictRDD(model: AbstractModule[Activity, Activity, T],
                      dataRdd: JavaRDD[Sample]): JavaRDD[JTensor] = {
    val tensorRDD = model.predict(dataRdd.rdd.map(toSample(_)))
    val listRDD = tensorRDD.map{res =>
      val tensor = res.asInstanceOf[Tensor[T]]
      val cloneTensor = tensor.clone()
      toJTensor(cloneTensor)

    }
    new JavaRDD[JTensor](listRDD)
  }

  def modelForward(model: AbstractModule[Activity, Activity, T],
                   input: JList[JTensor]): JList[JTensor] = {
    forward(input, model.forward)
  }

  def modelBackward(model: AbstractModule[Activity, Activity, T],
                    input: JList[JTensor], gradOutput: JList[JTensor]): JList[JTensor] = {
    backward(input, gradOutput, model.backward)
  }


  def graphForward(graph: Graph[T],
                   input: JList[JTensor]): JList[JTensor] = {
    forward(input, graph.forward)
  }

  def graphBackward(graph: Graph[T],
                    input: JList[JTensor], gradOutput: JList[JTensor]): JList[JTensor] = {
    backward(input, gradOutput, graph.backward)
  }

  private def forward(input: JList[JTensor], forward: (Activity) => Activity): JList[JTensor] = {
    val inputActivity = jTensorsToActivity(input)
    val outputActivity = forward(inputActivity)
    activityToJTensors(outputActivity)
  }

  private def backward(input: JList[JTensor], gradOutput: JList[JTensor],
                       backward: (Activity, Activity) => Activity): JList[JTensor] = {
    val inputActivity = jTensorsToActivity(input)
    val gradOutputActivity = jTensorsToActivity(gradOutput)
    val outputActivity = backward(inputActivity, gradOutputActivity)
    activityToJTensors(outputActivity)
  }

  def criterionForward(criterion: AbstractCriterion[Activity, Activity, T],
                       input: JList[JTensor], target: JList[JTensor]): T = {
    val inputActivity = jTensorsToActivity(input)
    val targetActivity = jTensorsToActivity(target)
    return criterion.forward(inputActivity, targetActivity)
  }

  def criterionBackward(criterion: AbstractCriterion[Activity, Activity, T],
                        input: JList[JTensor], target: JList[JTensor]): JList[JTensor] = {
    val inputActivity = jTensorsToActivity(input)
    val targetActivity = jTensorsToActivity(target)
    val outputActivity = criterion.backward(inputActivity, targetActivity)
    activityToJTensors(outputActivity)
  }

  def modelGetParameters(model: AbstractModule[Activity, Activity, T])
  : JMap[Any, JMap[Any, JList[JList[Any]]]] = {
    model.getParametersTable().getState().mapValues {
      case name2Values: Table =>
        name2Values.getState().mapValues {
          case t: Tensor[T] =>
            val tensorClone = t.clone()
            val item = List(tensorClone.storage().toList.asJava.asInstanceOf[JList[Any]],
              tensorClone.size().toList.asJava.asInstanceOf[JList[Any]]).asJava
            item
        }.asJava
    }.asJava
  }

  def createMaxEpoch(max: Int): Trigger = {
    Trigger.maxEpoch(max)
  }

  def createEveryEpoch(): Trigger = {
    Trigger.everyEpoch
  }

  def createSeveralIteration(interval: Int): Trigger = {
    Trigger.severalIteration(interval)
  }

  def createMaxIteration(max: Int): Trigger = {
    Trigger.maxIteration(max)
  }

  def createSGD(learningRate: Double = 1e-3,
    learningRateDecay: Double = 0.0,
    weightDecay: Double = 0.0,
    momentum: Double = 0.0,
    dampening: Double = Double.MaxValue,
    nesterov: Boolean = false,
    leaningRateSchedule: SGD.LearningRateSchedule = SGD.Default(),
    learningRates: JTensor = null,
    weightDecays: JTensor = null): SGD[T] = {
    val p1 = if (learningRates == null) null else toTensor(learningRates)
    val p2 = if (weightDecays == null) null else toTensor(weightDecays)
    new SGD[T](learningRate, learningRateDecay, weightDecay, momentum, dampening,
      nesterov, leaningRateSchedule, p1, p2)
  }

  def createAdagrad(learningRate: Double = 1e-3,
    learningRateDecay: Double = 0.0,
    weightDecay: Double = 0.0): Adagrad[T] = {
    new Adagrad[T](learningRate, learningRateDecay, weightDecay)
  }

  def createLBFGS(maxIter: Int = 20,
    maxEval: Double = Double.MaxValue,
    tolFun: Double = 1e-5,
    tolX: Double = 1e-9,
    nCorrection: Int = 100,
    learningRate: Double = 1.0,
    verbose: Boolean = false,
    lineSearch: LineSearch[T] = null,
    lineSearchOptions: JMap[Any, Any] = null): LBFGS[T] = {
    val p1 = if (lineSearch == null) None else Option(lineSearch)
    val p2 = if (lineSearchOptions == null) None else Option(T(lineSearchOptions))
    new LBFGS[T](maxIter, maxEval, tolFun, tolX, nCorrection, learningRate, verbose, p1, p2)
  }

  def createAdadelta(decayRate: Double = 0.9, Epsilon: Double = 1e-10): Adadelta[T] = {
    new Adadelta[T](decayRate, Epsilon)
  }

  def createAdam(
    learningRate: Double = 1e-3,
    learningRateDecay: Double = 0.0,
    beta1: Double = 0.9,
    beta2: Double = 0.999,
    Epsilon: Double = 1e-8): Adam[T] = {
    new  Adam[T](learningRate, learningRateDecay, beta1, beta2, Epsilon)
  }

  def createAdamax(
    learningRate: Double = 0.002,
    beta1: Double = 0.9,
    beta2: Double = 0.999,
    Epsilon: Double = 1e-38): Adamax[T] = {
    new Adamax(learningRate, beta1, beta2, Epsilon)
  }

  def createRMSprop(
    learningRate: Double = 1e-2,
    learningRateDecay: Double = 0.0,
    decayRate: Double = 0.99,
    Epsilon: Double = 1e-8): RMSprop[T] = {
    new  RMSprop[T](learningRate, learningRateDecay, decayRate, Epsilon)
  }

  def createOptimizer(model: AbstractModule[Activity, Activity, T],
                      trainingRdd: JavaRDD[Sample],
                      criterion: Criterion[T],
                      optimMethod: OptimMethod[T],
                      endTrigger: Trigger,
                      batchSize: Int): Optimizer[T, MiniBatch[T]] = {
    val optimizer = new DistriOptimizer(
      _model = model,
      dataset = batching(trainingRdd, batchSize),
      criterion = criterion
    ).asInstanceOf[Optimizer[T, MiniBatch[T]]]
    // TODO: we should provide a more convenient way to create Table

    optimizer.setEndWhen(endTrigger)

    optimizer.setOptimMethod(optimMethod)

    // TODO: remove this
    optimizer.disableCheckSingleton()

    optimizer
  }

  def setValidation(optimizer: Optimizer[T, MiniBatch[T]],
                    batchSize: Int,
                    trigger: Trigger,
                    valRdd: JavaRDD[Sample],
                    vMethods: JList[String]): Unit = {
    optimizer.setValidation(trigger, batching(valRdd, batchSize.toInt),
      toValidationMethod(vMethods))
  }

  def setCheckPoint(optimizer: Optimizer[T, MiniBatch[T]],
                    trigger: Trigger,
                    checkPointPath: String,
                    isOverwrite: Boolean): Unit = {
    optimizer.setCheckpoint(checkPointPath, trigger)
    if (isOverwrite) {
      optimizer.overWriteCheckpoint()
    }
  }

  def setTrainSummary(optimizer: Optimizer[T, MiniBatch[T]], summary: TrainSummary): Unit = {
    optimizer.setTrainSummary(summary)
  }

  def setValSummary(optimizer: Optimizer[T, MiniBatch[T]], summary: ValidationSummary): Unit = {
    optimizer.setValidationSummary(summary)
  }

  def summaryReadScalar(summary: Summary, tag: String): JList[JList[Any]] = {
    val result = summary.readScalar(tag)
    result.toList.map { item =>
      List(item._1, item._2, item._3).asJava.asInstanceOf[JList[Any]]
    }.asJava
  }

  def summarySetTrigger(
                         summary: TrainSummary,
                         summaryName: String,
                         trigger: Trigger): TrainSummary = {
    summary.setSummaryTrigger(summaryName, trigger)
    summary
  }

  def createTrainSummary(logDir: String,
                         appName: String): TrainSummary = {
    new TrainSummary(logDir, appName)
  }

  def createValidationSummary(logDir: String,
                              appName: String): ValidationSummary = {
    new ValidationSummary(logDir, appName)
  }

  def createGraph(input: JList[ModuleNode[T]], output: JList[ModuleNode[T]]): Graph[T] = {
    Graph(input.asScala.toArray, output.asScala.toArray)
  }

  def createNode(module: AbstractModule[Activity, Activity, T],
                 x: JList[ModuleNode[T]]): ModuleNode[T] = {
    if (null == x || x.isEmpty) {
      module.apply()
    } else {
      module.apply(x.asScala : _*)
    }
  }

  def createInput(): ModuleNode[T] = {
    Input()
  }

  def initEngine(): Unit = {
    Engine.init
  }


  def setWeights(model: AbstractModule[Activity, Activity, T], weights: JList[JTensor]): Unit = {
    val weightTensor = weights.asScala.toArray.map(toTensor(_))
    model.setWeightsBias(weightTensor)
  }

  def getWeights(model: AbstractModule[Activity, Activity, T]): JList[JTensor] = {
    val weights = model.getWeightsBias()
    if (weights != null) {
      weights.map(toJTensor(_)).toList.asJava
    } else {
      null
    }
  }

  def uniform(a: Double, b: Double, size: JList[Int]): JTensor = {
    val result = Tensor[T]().resize(size.asScala.toArray)
    result.apply1(i => ev.fromType(RandomGenerator.RNG.uniform(a, b)))
    toJTensor(result)
  }
}



