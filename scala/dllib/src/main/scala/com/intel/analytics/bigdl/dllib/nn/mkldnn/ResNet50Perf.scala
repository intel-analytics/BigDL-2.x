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

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.mkl.{Memory, MklDnn}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.nn.mkldnn.ResNet.DatasetType.ImageNet
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{Engine, T, Table}
import org.apache.log4j.Logger
import scopt.OptionParser

import scala.reflect.ClassTag

object ResNet50Perf {

  val logger = Logger.getLogger(getClass)

  val parser = new OptionParser[ResNet50PerfParams]("BigDL Local ResNet-50 Performance Test") {
    opt[Int]('b', "batchSize")
      .text("Batch size of input data")
      .action((v, p) => p.copy(batchSize = v))
    opt[Int]('i', "iteration")
      .text("Iteration of perf test. The result will be average of each iteration time cost")
      .action((v, p) => p.copy(iteration = v))
    opt[Boolean]('t', "training")
      .text(s"Perf test training or testing")
      .action((v, p) => p.copy(training = v))
  }

  def main(argv: Array[String]): Unit = {
    System.setProperty("bigdl.mkldnn.fusion.convbn", "true")
    System.setProperty("bigdl.mkldnn.fusion.bnrelu", "true")
    System.setProperty("bigdl.mkldnn.fusion.convrelu", "true")
    System.setProperty("bigdl.mkldnn.fusion.convsum", "true")

    val coreNumber: Int = Runtime.getRuntime.availableProcessors() / 2
    System.setProperty("bigdl.mklNumThreads", s"$coreNumber")
    Engine.setCoreNumber(1)
    MklDnn.setNumThreads(coreNumber)

    parser.parse(argv, new ResNet50PerfParams()).foreach { params =>
      val batchSize = params.batchSize
      val training = params.training
      val iterations = params.iteration

      val classNum = 1000

      val inputFormat = Memory.Format.nchw
      val inputShape = Array(batchSize, 3, 224, 224)
      val input = Tensor(inputShape).rand()
      val label = Tensor(batchSize).apply1(_ => Math.floor(RNG.uniform(0, 1) * 1000).toFloat)

      val model = ResNet(batchSize, classNum, T("depth" -> 50, "dataSet" -> ImageNet))
      val criterion = CrossEntropyCriterion()

      if (training) {
        model.compile(TrainingPhase, Array(HeapData(inputShape, inputFormat)))
        model.training()
      } else {
        model.compile(InferencePhase, Array(HeapData(inputShape, inputFormat)))
        model.evaluate()
      }

      var iteration = 0
      while (iteration < iterations) {
        val start = System.nanoTime()
        val output = model.forward(input)

        if (training) {
          val _loss = criterion.forward(output, label)
          val errors = criterion.backward(output, label).toTensor
          model.backward(input, errors)
        }

        val takes = System.nanoTime() - start

        val throughput = "%.2f".format(batchSize.toFloat / (takes / 1e9))
        logger.info(s"Iteration $iteration, takes $takes s, throughput is $throughput imgs/sec")

        iteration += 1
      }
    }
  }
}

case class ResNet50PerfParams (
    batchSize: Int = 16,
    iteration: Int = 50,
    training: Boolean = true
)

object ResNet {
  def modelInit(model: Module[Float]): Unit = {
    def initModules(model: Module[Float]): Unit = {
      model match {
        case container: Container[Activity, Activity, Float]
        => container.modules.foreach(m => initModules(m))
        case conv: SpatialConvolution =>
          val n: Float = conv.kernelW * conv.kernelW * conv.nOutputPlane
          val weight = Tensor[Float].resize(conv.weight.size()).apply1 { _ =>
            RNG.normal(0, Math.sqrt(2.0f / n)).toFloat
          }
          val bias = Tensor[Float].resize(conv.bias.size()).apply1(_ => 0.0f)
          conv.weight.copy(weight)
          conv.bias.copy(bias)
        case bn: SpatialBatchNormalization =>
          val runningMean = Tensor[Float].resize(bn.runningMean.size()).fill(0)
          val runningVairance = Tensor[Float].resize(bn.runningVariance.size()).fill(1)
          bn.runningMean.copy(runningMean)
          bn.runningVariance.copy(runningVairance)
        case linear: Linear =>
          val bias = Tensor[Float](linear.bias.size()).apply1(_ => 0.0f)
          linear.bias.copy(bias)
        case _ => Unit
      }
    }
    initModules(model)
  }

  var iChannels = 0
  def apply(batchSize: Int, classNum: Int, opt: Table): Sequential = {

    val depth = opt.get("depth").getOrElse(18)
    val shortCutType = opt.get("shortcutType")
    val shortcutType = shortCutType.getOrElse(ShortcutType.B).asInstanceOf[ShortcutType]
    val dataSet = opt.getOrElse[DatasetType]("dataSet", DatasetType.CIFAR10)
    val optnet = opt.get("optnet").getOrElse(true)

    def shortcut(nInputPlane: Int, nOutputPlane: Int, stride: Int, name: String): Module[Float] = {
      val useConv = shortcutType == ShortcutType.C ||
        (shortcutType == ShortcutType.B && nInputPlane != nOutputPlane)

      if (useConv) {
        Sequential()
          .add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride, optnet = optnet)
            .setName(s"res${name}_branch1"))
          .add(SbnDnn(nOutputPlane).setName(s"bn${name}_branch1"))
      } else if (nInputPlane != nOutputPlane) {
        throw new IllegalArgumentException(s"useConv false")
      } else {
        Identity()
      }
    }

    def bottleneck(n: Int, stride: Int, name: String = ""): Module[Float] = {
      val nInputPlane = iChannels
      iChannels = n * 4

      val s = Sequential()
      s.add(Convolution(nInputPlane, n, 1, 1, 1, 1, 0, 0, optnet = optnet).setName(
          s"res${name}_branch2a"))
        .add(SbnDnn(n).setName(s"bn${name}_branch2a"))
        .add(ReLU().setName(s"res${name}_branch2a_relu"))
        .add(Convolution(n, n, 3, 3, stride, stride, 1, 1, optnet = optnet).setName(
          s"res${name}_branch2b"))
        .add(SbnDnn(n).setName(s"bn${name}_branch2b"))
        .add(ReLU().setName(s"res${name}_branch2b_relu"))
        .add(Convolution(n, n*4, 1, 1, 1, 1, 0, 0, optnet = optnet).setName(
          s"res${name}_branch2c"))
        .add(SbnDnn(n * 4).setInitMethod(Zeros, Zeros).setName(s"bn${name}_branch2c"))

      val model = Sequential()
        .add(ConcatTable().
          add(s).
          add(shortcut(nInputPlane, n*4, stride, name)).setName(s"$name/concatTable"))
        .add(CAddTable().setName(s"res$name"))
        .add(ReLU().setName(s"res${name}_relu"))
      model
    }

    def getName(i: Int, name: String): String = {
      val name1 = i match {
        case 1 => name + "a"
        case 2 => name + "b"
        case 3 => name + "c"
        case 4 => name + "d"
        case 5 => name + "e"
        case 6 => name + "f"
      }
      return name1
    }

    def layer(block: (Int, Int, String) => Module[Float], features: Int,
      count: Int, stride: Int = 1, name : String): Module[Float] = {
      val s = Sequential()
      for (i <- 1 to count) {
        s.add(block(features, if (i == 1) stride else 1, getName(i, name)))
      }
      s
    }

    val model = Sequential()
    if (dataSet == DatasetType.ImageNet) {
      val cfg = Map(
        50 -> ((3, 4, 6, 3), 2048, bottleneck: (Int, Int, String) => Module[Float])
      )

      require(cfg.keySet.contains(depth), s"Invalid depth ${depth}")

      val (loopConfig, nFeatures, block) = cfg.get(depth).get
      iChannels = 64

      model.add(ReorderMemory(HeapData(Array(batchSize, 3, 224, 224), Memory.Format.nchw)))
        .add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, propagateBack = false)
        .setName("conv1").setReLU(true))
        .add(SbnDnn(64).setName("bn_conv1"))
        .add(ReLU().setName("conv1_relu"))
        .add(MaxPooling(3, 3, 2, 2).setName("pool1"))
        .add(layer(block, 64, loopConfig._1, name = "2"))
        .add(layer(block, 128, loopConfig._2, 2, name = "3"))
        .add(layer(block, 256, loopConfig._3, 2, name = "4"))
        .add(layer(block, 512, loopConfig._4, 2, name = "5"))
        .add(AvgPooling(7, 7, 1, 1).setName("pool5"))
        .add(Linear(nFeatures, classNum).setInitMethod(RandomNormal(0.0, 0.01), Zeros).setName(
          "fc1000"))
        .add(ReorderMemory(HeapData(Array(batchSize, classNum), Memory.Format.nc)))
    } else {
      throw new IllegalArgumentException(s"Invalid dataset ${dataSet}")
    }

    modelInit(model)
    model
  }

  /**
   * dataset type
   * @param typeId type id
   */
  sealed abstract class DatasetType(typeId: Int)
    extends Serializable

  /**
   *  define some dataset type
   */
  object DatasetType {
    case object CIFAR10 extends DatasetType(0)
    case object ImageNet extends DatasetType(1)
  }

  /**
   * ShortcutType
   * @param typeId type id
   */
  sealed abstract class ShortcutType(typeId: Int)
    extends Serializable

  /**
   * ShortcutType-A is used for Cifar-10, ShortcutType-B is used for ImageNet.
   * ShortcutType-C is used for others.
   */
  object ShortcutType{
    case object A extends ShortcutType(0)
    case object B extends ShortcutType(1)
    case object C extends ShortcutType(2)
  }
}

object Convolution {
  def apply(
    nInputPlane: Int,
    nOutputPlane: Int,
    kernelW: Int,
    kernelH: Int,
    strideW: Int = 1,
    strideH: Int = 1,
    padW: Int = 0,
    padH: Int = 0,
    nGroup: Int = 1,
    propagateBack: Boolean = true,
    optnet: Boolean = true,
    weightDecay: Double = 1e-4): SpatialConvolution = {
    val conv = SpatialConvolution(nInputPlane, nOutputPlane, kernelW, kernelH,
      strideW, strideH, padW, padH, nGroup, propagateBack)
    conv.setInitMethod(MsraFiller(false), Zeros)
    conv
  }
}

object SbnDnn {
  def apply[@specialized(Float, Double) T: ClassTag](
    nOutput: Int,
    eps: Double = 1e-3,
    momentum: Double = 0.9,
    affine: Boolean = true)
    (implicit ev: TensorNumeric[T]): SpatialBatchNormalization = {
    SpatialBatchNormalization(nOutput, eps, momentum, affine).setInitMethod(Ones, Zeros)
  }
}
