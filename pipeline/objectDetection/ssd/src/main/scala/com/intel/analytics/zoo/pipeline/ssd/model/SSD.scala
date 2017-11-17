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
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{L2Regularizer, Regularizer}
import com.intel.analytics.zoo.pipeline.common.nn._
import org.apache.log4j.Logger

/**
 *
 * @param nInput
 * @param nPriors
 * @param minSizes
 * @param maxSizes
 * @param aspectRatios
 * @param isFlip
 * @param isClip
 * @param variances
 * @param step
 */
case class ComponetParam(nInput: Int, nPriors: Int, minSizes: Array[Float],
  maxSizes: Array[Float], aspectRatios: Array[Float], isFlip: Boolean,
  isClip: Boolean, variances: Array[Float], step: Int = 0, resolution: Int = 300)

object SSD {

  def apply(numClasses: Int, resolution: Int,
    basePart1: Sequential[Float], basePart2: Sequential[Float],
    params: Map[String, ComponetParam],
    isLastPool: Boolean, normScale: Float, param: PostProcessParam,
    wRegularizer: Regularizer[Float] = L2Regularizer(0.0005),
    bRegularizer: Regularizer[Float] = null)
  : Module[Float] = {
    val model = Sequential()
    model.add(basePart1)
      .add(ConcatTable()
        .add(NormalizeScale(2, scale = normScale,
          size = Array(1, params("conv4_3_norm").nInput, 1, 1),
          wRegularizer = L2Regularizer(0.0005))
          .setName("conv4_3_norm"))
        .add(basePart2
          .add(SpatialDilatedConvolution(params("fc7").nInput,
            1024, 3, 3, 1, 1, 6, 6, 6, 6)
            .setInitMethod(weightInitMethod = Xavier, biasInitMethod = Zeros).setName("fc6"))
          .add(ReLU(true).setName("relu6"))
          .add(SpatialConvolution(1024, 1024, 1, 1)
            .setInitMethod(weightInitMethod = Xavier, biasInitMethod = Zeros).setName("fc7"))
          .add(ReLU(true).setName("relu7"))))

    val conv4_3NormMboxPriorbox =
      Sequential()
        .add(SelectTable(1).setName("conv4_3_norm_out"))
        .add(getPriorBox("conv4_3_norm", params))

    val conv4_3data = SelectTable(1)


    def fc7data = SelectTable(2)
    val fc7MboxPriorbox =
      Sequential()
        .add(fc7data)
        .add(getPriorBox("fc7", params))



    val featureComponents = ConcatTable()
      .add(getConcatOutput("conv4_3_norm", params("conv4_3_norm").nInput,
        params("conv4_3_norm").nPriors, conv4_3NormMboxPriorbox, numClasses, conv4_3data))
      .add(getConcatOutput("fc7", 1024, params("fc7").nPriors,
        fc7MboxPriorbox, numClasses, fc7data))

    model.add(featureComponents)

    val com6 = addFeatureComponent6(featureComponents, params, numClasses)
    val com7 = addFeatureComponent7(com6, params, numClasses)
    val com8 = addFeatureComponent8(com7, params, numClasses, isLastPool, resolution)
    if (isLastPool) {
      addFeatureComponentPool6(com8, params, numClasses)
    } else {
      val com9 = addFeatureComponent9(com8, params, numClasses, resolution)
      if (resolution == 512) {
        addFeatureComponent10(com9, params, numClasses)
      }
    }
    val module = concatResults(model, numClasses, params.size)
    module.add(DetectionOutput(param))
    module.setScaleB(2)
    setRegularizer(module, wRegularizer, bRegularizer)
    module
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

  private def addComponet(params: Map[String, ComponetParam],
    component: Sequential[Float], name: String, numClasses: Int)
  : ConcatTable[Float] = {
    val param = params(name)
    // connect neighbor feature components
    val connection = ConcatTable()
      .add(getConcatOutput(name, param.nInput, param.nPriors,
        getPriorBox(name, params), numClasses))

    component
      .add(connection)
    connection
  }


  private def getConcatOutput(name: String, nInput: Int, numPreds: Int,
    priorBox: Module[Float], numClasses: Int,
    input: Module[Float] = null): ConcatTable[Float] = {
    ConcatTable()
      .add(getLocConfComponent(name, nInput, numPreds * numClasses, "conf", input))
      .add(getLocConfComponent(name, nInput, numPreds * 4, "loc", input))
      .add(priorBox)
  }

  private def getLocConfComponent(name: String, nInput: Int,
    nOutput: Int, typeName: String, input: Module[Float] = null): Sequential[Float] = {
    val out = Sequential()
    if (input != null) out.add(input)
    out.add(SpatialConvolution(nInput, nOutput, 3, 3, 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier, biasInitMethod = Zeros)
      .setName(s"${ name }_mbox_$typeName"))
      .add(Transpose(Array((2, 3), (3, 4))).setName(s"${ name }_mbox_${ typeName }_perm"))
      .add(InferReshape(Array(0, -1)).setName(s"${ name }_mbox_${ typeName }_flat"))
  }

  private def concatResults(model: Sequential[Float], numClasses: Int, numComponents: Int)
  : Sequential[Float] = {
    model
      .add(FlattenTable())
      .add(ConcatTable()
        .add(selectResults(2, 1, 1, numComponents, "mbox_loc"))
        .add(selectResults(1, 1, 1, numComponents, "mbox_conf"))
        .add(selectResults(3, 2, 2, numComponents, "mbox_priorbox")))
    model
  }

  private def addFeatureComponent6(connection: ConcatTable[Float],
    params: Map[String, ComponetParam],
    numClasses: Int): ConcatTable[Float] = {
    val component = Sequential()
    connection.add(component)
    component.add(SelectTable(2))
    addConvRelu(component, (1024, 256, 1, 1, 0), "6_1")
    addConvRelu(component, (256, 512, 3, 2, 1), "6_2")
    addComponet(params, component, "conv6_2", numClasses)
  }

  private def addFeatureComponent7(connection: ConcatTable[Float],
    params: Map[String, ComponetParam],
    numClasses: Int)
  : ConcatTable[Float] = {
    val component = Sequential()
    connection.add(component)
    addConvRelu(component, (512, 128, 1, 1, 0), "7_1")
    addConvRelu(component, (128, 256, 3, 2, 1), "7_2")

    addComponet(params, component, "conv7_2", numClasses)
  }

  private def addFeatureComponent8(connection: ConcatTable[Float],
    params: Map[String, ComponetParam], numClasses: Int,
    isLastPool: Boolean, resolution: Int)
  : ConcatTable[Float] = {
    val component = Sequential()
    connection.add(component)
    addConvRelu(component, (256, 128, 1, 1, 0), "8_1")
    if (isLastPool || resolution == 512) {
      addConvRelu(component, (128, 256, 3, 2, 1), "8_2")
    } else {
      addConvRelu(component, (128, 256, 3, 1, 0), "8_2")
    }

    addComponet(params, component, "conv8_2", numClasses)
  }

  private def addFeatureComponent9(connection: ConcatTable[Float],
    params: Map[String, ComponetParam], numClasses: Int,
    resolution: Int): ConcatTable[Float] = {
    val c9 = Sequential()
    connection.add(c9)
    addConvRelu(c9, (256, 128, 1, 1, 0), "9_1")
    if (resolution == 512) {
      addConvRelu(c9, (128, 256, 3, 2, 1), "9_2")
    } else {
      addConvRelu(c9, (128, 256, 3, 1, 0), "9_2")
    }

    val name = "conv9_2"
    if (resolution == 300) {
      val param = params(name)
      c9.add(getConcatOutput(name, param.nInput,
        param.nPriors, getPriorBox(name, params), numClasses))
      null
    } else {
      addComponet(params, c9, name, numClasses)
    }
  }

  private def addFeatureComponent10(connection: ConcatTable[Float],
    params: Map[String, ComponetParam], numClasses: Int): Unit = {
    val component = Sequential()
    connection.add(component)
    addConvRelu(component, (256, 128, 1, 1, 0), "10_1")
    addConvRelu(component, (128, 256, 4, 1, 1), "10_2")

    val name = "conv10_2"
    require(params.contains(name))
    val param = params(name)
    component
      .add(getConcatOutput(name, param.nInput,
        param.nPriors, getPriorBox(name, params), numClasses))
  }

  private def addFeatureComponentPool6(connection: ConcatTable[Float],
    params: Map[String, ComponetParam], numClasses: Int)
  : Unit = {
    val component = Sequential()
    connection.add(component)

    val name = "pool6"
    val pool6priorbox = getPriorBox(name, params)

    val param = params(name)

    component
      .add(SpatialAveragePooling(3, 3).setName(name))
      .add(getConcatOutput(name,
        param.nInput, param.nPriors, pool6priorbox, numClasses))
  }

  private def getPriorBox(name: String, params: Map[String, ComponetParam]): PriorBox[Float] = {
    val param = params(name)
    PriorBox[Float](minSizes = param.minSizes, maxSizes = param.maxSizes,
      _aspectRatios = param.aspectRatios, isFlip = param.isFlip, isClip = param.isClip,
      variances = param.variances, step = param.step, offset = 0.5f,
      imgH = param.resolution, imgW = param.resolution)
      .setName(s"${ name }_mbox_priorbox")
  }


  private val logger = Logger.getLogger(getClass)

  private[pipeline] def addConvRelu(model: Sequential[Float], p: (Int, Int, Int, Int, Int),
    name: String, prefix: String = "conv", nGroup: Int = 1, propogateBack: Boolean = true): Unit = {
    model.add(SpatialConvolution(p._1, p._2, p._3, p._3, p._4, p._4,
      p._5, p._5, nGroup = nGroup, propagateBack = propogateBack)
      .setInitMethod(weightInitMethod = Xavier, biasInitMethod = Zeros)
      .setName(s"$prefix$name"))
    model.add(ReLU(true).setName(s"relu$name"))
  }
}
