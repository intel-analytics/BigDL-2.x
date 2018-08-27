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

package com.intel.analytics.zoo.models.image.objectdetection.ssd

import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer._
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.zoo.models.image.common.ImageModel
import com.intel.analytics.zoo.models.image.objectdetection.common.OBUtils

import scala.reflect.ClassTag
import scala.reflect.runtime._

object SSDVgg extends ModuleSerializable {
  ModuleSerializer.registerModule(
    "com.intel.analytics.zoo.models.image.objectdetection.ssd.SSDVgg",
    SSDVgg)

  def apply[@specialized(Float, Double) T: ClassTag](classNum: Int, resolution: Int = 300,
    dataset: String = "pascal", sizes: Array[Float] = null,
    postProcessParam: DetectionOutputParam = null)
    (implicit ev: TensorNumeric[T]): SSDVgg[T] = {
    new SSDVgg[T](classNum, resolution, dataset, sizes, postProcessParam).build()
  }

  def vgg16[@specialized(Float, Double) T: ClassTag](implicit ev: TensorNumeric[T]):
    (ModuleNode[T], ModuleNode[T], ModuleNode[T]) = {
    val conv1_1 = SpatialConvolution[T](3, 64, 3, 3, 1, 1, 1, 1, propagateBack = false)
      .setInitMethod(weightInitMethod = Xavier, biasInitMethod = Zeros)
      .setName(s"conv1_1").inputs()

    val relu1_1 = ReLU(true).setName(s"relu1_1").inputs(conv1_1)
    val relu1_2 = OBUtils.addConvRelu(relu1_1, (64, 64, 3, 1, 1), "1_2")
    val pool1 = SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool1").inputs(relu1_2)

    val relu2_1 = OBUtils.addConvRelu(pool1, (64, 128, 3, 1, 1), "2_1")
    val relu2_2 = OBUtils.addConvRelu(relu2_1, (128, 128, 3, 1, 1), "2_2")
    val pool2 = SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool2").inputs(relu2_2)

    val relu3_1 = OBUtils.addConvRelu(pool2, (128, 256, 3, 1, 1), "3_1")
    val relu3_2 = OBUtils.addConvRelu(relu3_1, (256, 256, 3, 1, 1), "3_2")
    val relu3_3 = OBUtils.addConvRelu(relu3_2, (256, 256, 3, 1, 1), "3_3")
    val pool3 = SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool3").inputs(relu3_3)

    val relu4_1 = OBUtils.addConvRelu(pool3, (256, 512, 3, 1, 1), "4_1")
    val relu4_2 = OBUtils.addConvRelu(relu4_1, (512, 512, 3, 1, 1), "4_2")
    val relu4_3 = OBUtils.addConvRelu(relu4_2, (512, 512, 3, 1, 1), "4_3")

    val pool4 = SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool4").inputs(relu4_3)
    val relu5_1 = OBUtils.addConvRelu(pool4, (512, 512, 3, 1, 1), "5_1")
    val relu5_2 = OBUtils.addConvRelu(relu5_1, (512, 512, 3, 1, 1), "5_2")
    val relu5_3 = OBUtils.addConvRelu(relu5_2, (512, 512, 3, 1, 1), "5_3")
    val poo5 = SpatialMaxPooling[T](3, 3, 1, 1, 1, 1).ceil().setName("pool5").inputs(relu5_3)
    (conv1_1, relu4_3, poo5)
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
    builder : BigDLModule.Builder)
    (implicit ev: TensorNumeric[T]) : Unit = {
    val model = context.moduleData.module.asInstanceOf[SSDVgg[T]]
    val classNumBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context,
      classNumBuilder, model.classNum, universe.typeOf[Int])
    builder.putAttr("classNum", classNumBuilder.build)

    val resolutionBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context,
      resolutionBuilder, model.resolution, universe.typeOf[Int])
    builder.putAttr("resolution", resolutionBuilder.build)

    val datasetBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context,
      datasetBuilder, model.dataset, universe.typeOf[String])
    builder.putAttr("dataset", datasetBuilder.build)

    val hasSizes = if (model.sizes != null) {
      val sizesBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context,
        sizesBuilder, model.sizes, universe.typeOf[Array[Float]])
      builder.putAttr("sizes", sizesBuilder.build)
      true
    } else false
    val hasSizesBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, hasSizesBuilder,
      hasSizes, universe.typeOf[Boolean])
    builder.putAttr("hasSizes", hasSizesBuilder.build)

    val hasPostProcessParam = if (model.postProcessParam != null) {
      val nClassesBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, nClassesBuilder,
        model.postProcessParam.nClasses, universe.typeOf[Int])
      builder.putAttr("nClasses", nClassesBuilder.build)

      val shareLocationBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, shareLocationBuilder,
        model.postProcessParam.shareLocation, universe.typeOf[Boolean])
      builder.putAttr("shareLocation", shareLocationBuilder.build)

      val bgLabelBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, bgLabelBuilder,
        model.postProcessParam.bgLabel, universe.typeOf[Int])
      builder.putAttr("bgLabel", bgLabelBuilder.build)

      val nmsThreshBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, nmsThreshBuilder,
        model.postProcessParam.nmsThresh, universe.typeOf[Float])
      builder.putAttr("nmsThresh", nmsThreshBuilder.build)

      val nmsTopkBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, nmsTopkBuilder,
        model.postProcessParam.nmsTopk, universe.typeOf[Int])
      builder.putAttr("nmsTopk", nmsTopkBuilder.build)

      val keepTopKBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, keepTopKBuilder,
        model.postProcessParam.keepTopK, universe.typeOf[Int])
      builder.putAttr("keepTopK", keepTopKBuilder.build)

      val confThreshBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, confThreshBuilder,
        model.postProcessParam.confThresh, universe.typeOf[Float])
      builder.putAttr("confThresh", confThreshBuilder.build)

      val varianceEncodedInTargetBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, varianceEncodedInTargetBuilder,
        model.postProcessParam.varianceEncodedInTarget, universe.typeOf[Boolean])
      builder.putAttr("varianceEncodedInTarget", varianceEncodedInTargetBuilder.build)
      true
    } else false

    val hasPostProcessParamBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, hasPostProcessParamBuilder,
      hasPostProcessParam, universe.typeOf[Boolean])
    builder.putAttr("hasPostProcessParam", hasPostProcessParamBuilder.build)
  }

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
    val attrMap = context.bigdlModule.getAttrMap

    val classNum = DataConverter
      .getAttributeValue(context, attrMap.get("classNum"))
      .asInstanceOf[Int]
    val resolution = DataConverter
      .getAttributeValue(context, attrMap.get("resolution"))
      .asInstanceOf[Int]
    val dataset = DataConverter
      .getAttributeValue(context, attrMap.get("dataset"))
      .asInstanceOf[String]
    val hasSizes = DataConverter
      .getAttributeValue(context, attrMap.get("hasSizes"))
      .asInstanceOf[Boolean]
    val sizes = if (hasSizes) {
      DataConverter
        .getAttributeValue(context, attrMap.get("sizes"))
        .asInstanceOf[Array[Float]]
    } else null

    val hasPostProcessParam = DataConverter
      .getAttributeValue(context, attrMap.get("hasPostProcessParam"))
      .asInstanceOf[Boolean]
    val postProcessParam = if (hasPostProcessParam) {
      val nClasses = DataConverter
        .getAttributeValue(context, attrMap.get("nClasses"))
        .asInstanceOf[Int]
      val shareLocation = DataConverter
        .getAttributeValue(context, attrMap.get("shareLocation"))
        .asInstanceOf[Boolean]
      val bgLabel = DataConverter
        .getAttributeValue(context, attrMap.get("bgLabel"))
        .asInstanceOf[Int]
      val nmsThresh = DataConverter
        .getAttributeValue(context, attrMap.get("nmsThresh"))
        .asInstanceOf[Float]
      val nmsTopk = DataConverter
        .getAttributeValue(context, attrMap.get("nmsTopk"))
        .asInstanceOf[Int]
      val keepTopK = DataConverter
        .getAttributeValue(context, attrMap.get("keepTopK"))
        .asInstanceOf[Int]
      val confThresh = DataConverter
        .getAttributeValue(context, attrMap.get("confThresh"))
        .asInstanceOf[Float]
      val varianceEncodedInTarget = DataConverter
        .getAttributeValue(context, attrMap.get("varianceEncodedInTarget"))
        .asInstanceOf[Boolean]
      DetectionOutputParam(nClasses, shareLocation, bgLabel, nmsThresh, nmsTopk,
        keepTopK, confThresh, varianceEncodedInTarget)
    } else null
    SSDVgg(classNum, resolution, dataset, sizes, postProcessParam)
  }
}

class SSDVgg[T: ClassTag] private (val classNum: Int, val resolution: Int = 300,
                                   val dataset: String = "pascal",
                                   var sizes: Array[Float] = null,
                                   var postProcessParam: DetectionOutputParam = null)
                                  (implicit ev: TensorNumeric[T])
  extends ImageModel[T] {

  override def buildModel(): AbstractModule[Activity, Activity, T] = {
    val model = Sequential[T]()
    require(resolution == 300 || resolution == 512, "only support 300*300 or 512*512 as input")
    val isClip = false
    val isFlip = true
    val variances = Array(0.1f, 0.1f, 0.2f, 0.2f)
    var params = Map[String, ComponetParam]()
    val priorBoxSizes = if (sizes != null) {
      if (resolution == 300) {
        require(sizes.length == 7, "the min and max division boundary length should be 7")
      } else {
        require(sizes.length == 8, "the min and max division boundary length should be 8")
      }
      sizes
    } else {
      if (dataset == "pascal") {
        if (resolution == 300) Array[Float](30, 60, 111, 162, 213, 264, 315)
        else Array[Float](35.84f, 76.8f, 153.6f, 230.4f, 307.2f, 384.0f, 460.8f, 537.6f)
      } else if (dataset == "coco") {
        if (resolution == 300) Array[Float](21, 45, 99, 153, 207, 261, 315)
        else Array[Float](20.48f, 51.2f, 133.12f, 215.04f, 296.96f, 378.88f, 460.8f, 542.72f)
      } else {
        throw new NotImplementedError(s"$dataset is not supported," +
          s" please provide your own boundary sizes to parameter sizes")
      }
    }

    val (conv1_1, relu4_3, poo5) = SSDVgg.vgg16[T]

    val postParam = if (postProcessParam == null) DetectionOutputParam(classNum)
    else postProcessParam
    if (resolution == 300) {
      params += "conv4_3_norm" -> ComponetParam(512, 4,
        minSizes = Array(priorBoxSizes(0)), maxSizes = Array(priorBoxSizes(1)),
        aspectRatios = Array(2), isFlip, isClip, variances, 8, 300)
      params += "fc7" -> ComponetParam(512, 6,
        minSizes = Array(priorBoxSizes(1)), maxSizes = Array(priorBoxSizes(2)),
        aspectRatios = Array(2, 3), isFlip, isClip, variances, 16, 300)
      params += "conv6_2" -> ComponetParam(512, 6,
        minSizes = Array(priorBoxSizes(2)), maxSizes = Array(priorBoxSizes(3)),
        aspectRatios = Array(2, 3), isFlip, isClip, variances, 32, 300)
      params += "conv7_2" -> ComponetParam(256, 6,
        minSizes = Array(priorBoxSizes(3)), maxSizes = Array(priorBoxSizes(4)),
        aspectRatios = Array(2, 3), isFlip, isClip, variances, 64, 300)
      params += "conv8_2" -> ComponetParam(256, 4,
        minSizes = Array(priorBoxSizes(4)), maxSizes = Array(priorBoxSizes(5)),
        aspectRatios = Array(2), isFlip, isClip, variances, 100, 300)
      params += "conv9_2" -> ComponetParam(256, 4,
        minSizes = Array(priorBoxSizes(5)), maxSizes = Array(priorBoxSizes(6)),
        aspectRatios = Array(2), isFlip, isClip, variances, 300, 300)
      model.add(SSDGraph[T](classNum, resolution, conv1_1, relu4_3, poo5, params,
        normScale = 20f, isLastPool = false, param = postParam))
    } else {
      params += "conv4_3_norm" -> ComponetParam(512, 4,
        minSizes = Array(priorBoxSizes(0)), maxSizes = Array(priorBoxSizes(1)),
        aspectRatios = Array(2), isFlip, isClip, variances, 8, 512)
      params += "fc7" -> ComponetParam(512, 6,
        minSizes = Array(priorBoxSizes(1)), maxSizes = Array(priorBoxSizes(2)),
        aspectRatios = Array(2, 3), isFlip, isClip, variances, 16, 512)
      params += "conv6_2" -> ComponetParam(512, 6,
        minSizes = Array(priorBoxSizes(2)), maxSizes = Array(priorBoxSizes(3)),
        aspectRatios = Array(2, 3), isFlip, isClip, variances, 32, 512)
      params += "conv7_2" -> ComponetParam(256, 6,
        minSizes = Array(priorBoxSizes(3)), maxSizes = Array(priorBoxSizes(4)),
        aspectRatios = Array(2, 3), isFlip, isClip, variances, 64, 512)
      params += "conv8_2" -> ComponetParam(256, 6,
        minSizes = Array(priorBoxSizes(4)), maxSizes = Array(priorBoxSizes(5)),
        aspectRatios = Array(2, 3), isFlip, isClip, variances, 128, 512)
      params += "conv9_2" -> ComponetParam(256, 4,
        minSizes = Array(priorBoxSizes(5)), maxSizes = Array(priorBoxSizes(6)),
        aspectRatios = Array(2), isFlip, isClip, variances, 256, 512)
      params += "conv10_2" -> ComponetParam(256, 4,
        minSizes = Array(priorBoxSizes(6)), maxSizes = Array(priorBoxSizes(7)),
        aspectRatios = Array(2), isFlip, isClip, variances, 512, 512)
      model.add(SSDGraph[T](classNum, resolution, conv1_1, relu4_3, poo5, params,
        normScale = 20f, isLastPool = false, param = postParam))
    }
    model
  }
}
