package com.intel.analytics.zoo.utils

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.models.resnet.Convolution
import com.intel.analytics.bigdl.models.resnet.ResNet.DatasetType.ImageNet
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.L2Regularizer
import org.apache.log4j.Logger

import scala.reflect.ClassTag

object ModelConvertor {
  def convert[T: ClassTag](args: Object*)(
    implicit ev: TensorNumeric[T]): Module[T] = {
    val obj = "com.intel.analytics.bigdl.utils.intermediate.ConversionUtils"
    val methodName = "convert"
    val clazz = Class.forName(obj)
    val argsWithTag = args ++ Seq(implicitly[reflect.ClassTag[T]], ev)
    val method =
      try {
        clazz.getMethod(methodName, argsWithTag.map(_.getClass): _*)
      } catch {
        case t: Throwable =>
          val methods = clazz.getMethods().filter(_.getName() == methodName)
            .filter(_.getParameterCount == argsWithTag.size)
          require(methods.length == 1,
            s"We should only found one result, but got ${methodName}: ${methods.length}")
          methods(0)
      }
    method.invoke(obj, argsWithTag: _*).asInstanceOf[Module[T]]
  }

  def caffe2zoo(model: Module[Float]): Module[Float] = {
    val newModel =
      resnet50(2, T("depth" -> 50, "shortcutType" -> ShortcutType.B, "dataSet" -> ImageNet,
        "optnet" -> false))

    val pt = model.getParametersTable()
    val newPt = newModel.getParametersTable()

    // copy parameter without scale
    newPt.keySet.map(_.toString).foreach{ key =>
      pt[Table](key).keySet.foreach{key2 =>
        newPt[Table](key).apply[Tensor[Float]](key2).copy(
          pt[Table](key).apply[Tensor[Float]](key2)
        )
      }
    }

    // copy parameter from scale to bn
    pt.keySet.map(_.toString).filter(_.contains("scale")).foreach{key =>
      val bnkey = key.replace("scale", "bn")
      pt[Table](key).keySet.foreach{k =>
        newPt[Table](bnkey)[Tensor[Float]](k).copy(
          pt[Table](key)[Tensor[Float]](k)
        )
      }
    }
    newModel
  }

  var iChannels = 0
  def resnet50(classNum: Int, opt: Table): Module[Float] = {
    val depth = opt.get("depth").getOrElse(18)
    val shortCutType = opt.get("shortcutType")
    val shortcutType = shortCutType.getOrElse(ShortcutType.B).asInstanceOf[ShortcutType]
    val dataSet = opt.getOrElse[DatasetType]("dataSet", DatasetType.CIFAR10)
    val optnet = opt.get("optnet").getOrElse(true)

    def shortcut(nInputPlane: Int, nOutputPlane: Int, stride: Int,
                 name: String): Module[Float] = {
      val useConv = shortcutType == ShortcutType.C ||
        (shortcutType == ShortcutType.B && nInputPlane != nOutputPlane)

      if (useConv) {
        Sequential()
          .add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride, optnet = optnet)
            .setName(s"res${name}_branch1"))
          .add(sbn(nOutputPlane).setName(s"bn${name}_branch1"))
      } else if (nInputPlane != nOutputPlane) {
        Sequential()
          .add(SpatialAveragePooling(1, 1, stride, stride))
          .add(Concat(2)
            .add(Identity())
            .add(MulConstant(0f)))
      } else {
        Identity()
      }
    }

    def bottleneck(n: Int, stride: Int,
                   name: String): Module[Float] = {
      val nInputPlane = iChannels
      iChannels = n * 4

      val s = Sequential()
      s.add(Convolution(nInputPlane, n, 1, 1, 1, 1, 0, 0, optnet = optnet).setName(s"res${name}_branch2a"))
        .add(sbn(n).setName(s"bn${name}_branch2a"))
        .add(ReLU(true).setName(s"res${name}_branch2a_relu"))
        .add(Convolution(n, n, 3, 3, stride, stride, 1, 1, optnet = optnet).setName(s"res${name}_branch2b"))
        .add(sbn(n).setName(s"bn${name}_branch2b"))
        .add(ReLU(true).setName(s"res${name}_branch2b_relu"))
        .add(Convolution(n, n*4, 1, 1, 1, 1, 0, 0, optnet = optnet).setName(s"res${name}_branch2c"))
        .add(sbn(n * 4).setInitMethod(Zeros, Zeros).setName(s"bn${name}_branch2c"))
      Sequential()
        .add(ConcatTable()
          .add(s)
          .add(shortcut(nInputPlane, n*4, stride, name)))
        .add(CAddTable(true).setName(s"res${name}"))
        .add(ReLU(true).setName(s"res${name}_relu"))
    }

    def layer(block: (Int, Int, String) => Module[Float], features: Int,
              count: Int, id: Int, stride: Int = 1): Module[Float] = {
      val s = Sequential()
      for (i <- 0 until count) {
        s.add(block(features, if (i == 0) stride else 1, s"${id}${('a' + i).toChar}"))
      }
      s
    }

    val model = Sequential()
    if (dataSet == DatasetType.ImageNet) {
      val cfg = Map(
        50 -> ((3, 4, 6, 3), 2048,
          bottleneck: (Int, Int, String) => Module[Float]) //,
        //        101 -> ((3, 4, 23, 3), 2048,
        //          bottleneck: (Int, Int, String) => Module[Float]),
        //        152 -> ((3, 8, 36, 3), 2048,
        //          bottleneck: (Int, Int, String) => Module[Float]),
        //        200 -> ((3, 24, 36, 3), 2048,
        //          bottleneck: (Int, Int, String) => Module[Float])
      )

      require(cfg.keySet.contains(depth), s"Invalid depth ${depth}")

      val (loopConfig, nFeatures, block) = cfg.get(depth).get
      iChannels = 64
      logger.info(" | ResNet-" + depth + " ImageNet")

      model.add(Convolution(3, 64, 7, 7, 2, 2, 3, 3, optnet = optnet, propagateBack = false)
        .setName("conv1"))
        .add(sbn(64).setName("bn_conv1"))
        .add(ReLU(true).setName("conv1_relu"))
        .add(SpatialMaxPooling(3, 3, 2, 2, 0, 0).ceil().setName("pool1"))
        .add(layer(block, 64, loopConfig._1, 2))
        .add(layer(block, 128, loopConfig._2,3,  2))
        .add(layer(block, 256, loopConfig._3,4,  2))
        .add(layer(block, 512, loopConfig._4,5,  2))
        .add(SpatialAveragePooling(7, 7, 1, 1).setName("pool5"))
        .add(View(nFeatures).setNumInputDims(3))
        .add(Linear(nFeatures, classNum, true, L2Regularizer(1e-4), L2Regularizer(1e-4))
          .setName(s"fc1000")
          .setInitMethod(RandomNormal(0.0, 0.01), Zeros))
        .add(SoftMax().setName("probt"))
    } else {
      throw new IllegalArgumentException(s"Invalid dataset ${dataSet}")
    }
    model
  }

  def sbn[@specialized(Float, Double) T: ClassTag](
                                                    nOutput: Int,
                                                    eps: Double = 1e-5, // 1e-5 in caffe
                                                    momentum: Double = 0.1,
                                                    affine: Boolean = true)(implicit ev: TensorNumeric[T]): SpatialBatchNormalization[T] = {
    SpatialBatchNormalization[T](nOutput, eps, momentum, affine).setInitMethod(Ones, Zeros)
  }

  val logger = Logger.getLogger(this.getClass)
}
