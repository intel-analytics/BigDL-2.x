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

package com.intel.analytics.zoo.pipeline.ssd.model

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{L2Regularizer, Regularizer}
import com.intel.analytics.zoo.pipeline.common.nn._
import com.intel.analytics.bigdl.tensor.Storage
import org.apache.log4j.Logger


case class PreProcessParam(batchSize: Int = 4,
  resolution: Int = 300,
  pixelMeanRGB: (Float, Float, Float),
  hasLabel: Boolean, nPartition: Int
)

object SSDGraph {

  def apply(numClasses: Int, resolution: Int, input: ModuleNode[Float],
    basePart1: ModuleNode[Float], basePart2: ModuleNode[Float],
    params: Map[String, ComponetParam],
    isLastPool: Boolean, normScale: Float, param: PostProcessParam,
    wRegularizer: Regularizer[Float] = L2Regularizer(0.0005),
    bRegularizer: Regularizer[Float] = null)
  : Module[Float] = {

    val conv4_3_norm = NormalizeScale(2, scale = normScale,
      size = Array(1, params("conv4_3_norm").nInput, 1, 1),
      wRegularizer = L2Regularizer(0.0005))
      .setName("conv4_3_norm").inputs(basePart1)


    val base2 = basePart2
    val fc6 = SpatialDilatedConvolution(params("fc7").nInput,
      1024, 3, 3, 1, 1, 6, 6, 6, 6)
      .setInitMethod(weightInitMethod = Xavier, biasInitMethod = Zeros)
      .setName("fc6").inputs(base2)

    val relu6 = ReLU(true).setName("relu6").inputs(fc6)
    val fc7 = SpatialConvolution(1024, 1024, 1, 1)
      .setInitMethod(weightInitMethod = Xavier, biasInitMethod = Zeros).setName("fc7").inputs(relu6)
    val relu7 = ReLU(true).setName("relu7").inputs(fc7)

    val norm4_3Out = getConcatOutput(conv4_3_norm, "conv4_3_norm", params,
      params("conv4_3_norm").nInput, numClasses)
    val fc7Out = getConcatOutput(relu7, "fc7", params, 1024, numClasses)


    val relu6_1 = addConvRelu(relu7, (1024, 256, 1, 1, 0), "6_1")
    val relu6_2 = addConvRelu(relu6_1, (256, 512, 3, 2, 1), "6_2")
    val c6Out = getConcatOutput(relu6_2, "conv6_2", params, params("conv6_2").nInput, numClasses)

    val relu7_1 = addConvRelu(relu6_2, (512, 128, 1, 1, 0), "7_1")
    val relu7_2 = addConvRelu(relu7_1, (128, 256, 3, 2, 1), "7_2")
    val c7Out = getConcatOutput(relu7_2, "conv7_2", params, params("conv7_2").nInput, numClasses)

    val relu8_1 = addConvRelu(relu7_2, (256, 128, 1, 1, 0), "8_1")
    val relu8_2 = if (isLastPool || resolution == 512) {
      addConvRelu(relu8_1, (128, 256, 3, 2, 1), "8_2")
    } else {
      addConvRelu(relu8_1, (128, 256, 3, 1, 0), "8_2")
    }

    val c8Out = getConcatOutput(relu8_2, "conv8_2", params, params("conv8_2").nInput, numClasses)
    val (c9Out, relu9_2) = if (isLastPool) {
      //      addFeatureComponentPool6(com8, params, numClasses)
      val pool6 = SpatialAveragePooling(3, 3).setName("pool6").inputs(relu8_2)
      (getConcatOutput(pool6, "pool6", params, params("pool6").nInput, numClasses), pool6)
    } else {
      val relu9_1 = addConvRelu(relu8_2, (256, 128, 1, 1, 0), "9_1")
      val relu9_2 = if (resolution == 512) {
        addConvRelu(relu9_1, (128, 256, 3, 2, 1), "9_2")
      } else {
        addConvRelu(relu9_1, (128, 256, 3, 1, 0), "9_2")
      }
      (getConcatOutput(relu9_2, "conv9_2", params, params("conv9_2").nInput, numClasses), relu9_2)
    }

    val c10Out = if (resolution == 512) {
      val relu10_1 = addConvRelu(relu9_2, (256, 128, 1, 1, 0), "10_1")
      val relu10_2 = addConvRelu(relu10_1, (128, 256, 4, 1, 1), "10_2")
      getConcatOutput(relu10_2, "conv10_2", params, params("conv10_2").nInput, numClasses)
    } else {
      null
    }

   val (conf, loc, priors) = if (resolution == 300) {
      val conf = JoinTable(1, 1)
        .inputs(norm4_3Out._1, fc7Out._1, c6Out._1, c7Out._1, c8Out._1, c9Out._1)
      val loc = JoinTable(1, 1)
        .inputs(norm4_3Out._2, fc7Out._2, c6Out._2, c7Out._2, c8Out._2, c9Out._2)
      val priors = JoinTable(2, 2)
        .inputs(norm4_3Out._3, fc7Out._3, c6Out._3, c7Out._3, c8Out._3, c9Out._3)
      (conf, loc, priors)
    } else {
      val conf = JoinTable(1, 1)
        .inputs(norm4_3Out._1, fc7Out._1, c6Out._1, c7Out._1, c8Out._1, c9Out._1, c10Out._1)
      val loc = JoinTable(1, 1)
        .inputs(norm4_3Out._2, fc7Out._2, c6Out._2, c7Out._2, c8Out._2, c9Out._2, c10Out._2)
      val priors = JoinTable(2, 2)
        .inputs(norm4_3Out._3, fc7Out._3, c6Out._3, c7Out._3, c8Out._3, c9Out._3, c10Out._3)
      (conf, loc, priors)
    }

    val model = Graph(input, Array(loc, conf, priors))
    model.setScaleB(2)
    stopGradient(model)
    val ssd = Sequential()
    ssd.add(model)
    ssd.add(DetectionOutput(param))
    setRegularizer(model, wRegularizer, bRegularizer)
    ssd
  }


  private def setRegularizer(module: AbstractModule[Activity, Activity, Float],
    wRegularizer: Regularizer[Float], bRegularizer: Regularizer[Float])
  : Unit = {
    if (module.isInstanceOf[Container[Activity, Activity, Float]]) {
      val ms = module.asInstanceOf[Container[Activity, Activity, Float]].modules
      ms.foreach(m => setRegularizer(m, wRegularizer, bRegularizer))
    } else if (module.isInstanceOf[SpatialConvolution[Float]]) {
      val m = module.asInstanceOf[SpatialConvolution[Float]]
      m.wRegularizer = wRegularizer
      m.bRegularizer = bRegularizer
    } else if (module.isInstanceOf[SpatialDilatedConvolution[Float]]) {
      val m = module.asInstanceOf[SpatialDilatedConvolution[Float]]
      m.wRegularizer = wRegularizer
      m.bRegularizer = bRegularizer
    } else if (module.isInstanceOf[NormalizeScale[Float]]) {
      val m = module.asInstanceOf[NormalizeScale[Float]]
      m.wRegularizer = wRegularizer
    }
  }

  private def stopGradient(model: Graph[Float]): Unit = {
    val priorboxNames = model.modules
      .filter(_.getClass.getName.toLowerCase().endsWith("priorbox"))
      .map(_.getName()).toArray
    model.stopGradient(priorboxNames)
  }


  /**
   * share the storage of SpatialConvolution fInput
   * note that this sharing only works for Inference only
   * @param model model to share
   */
  def shareMemory(model: Module[Float]): Unit = {
    logger.info("Share memory in ssd")
    val shareFinputStorage = Storage[Float]()
    shareModules(model, shareFinputStorage)
  }

  private def shareModules(module: Module[Float], shareFinputStorage: Storage[Float]): Unit = {
    Utils.getNamedModules(module)
    module match {
      case m: Container[_, _, Float] =>
        for (m <- module.asInstanceOf[Container[_, _, Float]].modules) {
          shareModules(m, shareFinputStorage)
        }
      case _ =>
        if (module.getClass.getName.endsWith("SpatialConvolution")) {
          module.asInstanceOf[SpatialConvolution[Float]].fInput.set(shareFinputStorage)
        }
    }
  }

  /**
   * select results (confs || locs || priorboxes), use JoinTable to concat them into one tensor
   * @param start
   * @param dim
   * @param nInputDims
   * @param name
   * @return
   */
  private def selectResults(start: Int, dim: Int, nInputDims: Int, numComponents: Int,
    name: String): Module[Float] = {
    val con = ConcatTable().setName(s"select results $name")
    var i = start
    while (i <= numComponents * 3) {
      con.add(SelectTable(i).setName(s"select $name $i"))
      i += 3
    }
    Sequential().add(con).add(JoinTable(dim, nInputDims).setName(name))
  }


  private def getConcatOutput(conv: ModuleNode[Float],
    name: String, params: Map[String, ComponetParam], nInput: Int, numClasses: Int)
  : (ModuleNode[Float], ModuleNode[Float], ModuleNode[Float]) = {
    val conFlat = getLocConfComponent(conv, name, nInput, params(name).nPriors * numClasses, "conf")
    val locFlat = getLocConfComponent(conv, name, nInput, params(name).nPriors * 4, "loc")
    val priorBox = getPriorBox(conv, name, params)
    (conFlat, locFlat, priorBox)
  }

  private def getLocConfComponent(prev: ModuleNode[Float], name: String, nInput: Int,
    nOutput: Int, typeName: String): ModuleNode[Float] = {
    val conv = SpatialConvolution(nInput, nOutput, 3, 3, 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier, biasInitMethod = Zeros)
      .setName(s"${ name }_mbox_$typeName").inputs(prev)
    val trans = Transpose(Array((2, 3), (3, 4))).setName(s"${ name }_mbox_${ typeName }_perm")
      .inputs(conv)
    InferReshape(Array(0, -1)).setName(s"${ name }_mbox_${ typeName }_flat").inputs(trans)
  }

  private def getPriorBox(conv: ModuleNode[Float],
    name: String, params: Map[String, ComponetParam]): ModuleNode[Float] = {
    val param = params(name)
    PriorBox[Float](minSizes = param.minSizes, maxSizes = param.maxSizes,
      _aspectRatios = param.aspectRatios, isFlip = param.isFlip, isClip = param.isClip,
      variances = param.variances, step = param.step, offset = 0.5f,
      imgH = param.resolution, imgW = param.resolution)
      .setName(s"${ name }_mbox_priorbox").inputs(conv)
  }


  private val logger = Logger.getLogger(getClass)

  private[pipeline] def addConvRelu(prevNodes: ModuleNode[Float], p: (Int, Int, Int, Int, Int),
    name: String, prefix: String = "conv", nGroup: Int = 1, propogateBack: Boolean = true)
  : ModuleNode[Float] = {
    val conv = SpatialConvolution(p._1, p._2, p._3, p._3, p._4, p._4,
      p._5, p._5, nGroup = nGroup, propagateBack = propogateBack)
      .setInitMethod(weightInitMethod = Xavier, biasInitMethod = Zeros)
      .setName(s"$prefix$name").inputs(prevNodes)
    ReLU(true).setName(s"relu$name").inputs(conv)
  }
}
