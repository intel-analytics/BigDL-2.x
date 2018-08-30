package com.intel.analytics.zoo.models.image.objectdetection.fasterrcnn

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.serializer.{DeserializeContext, ModuleSerializable, ModuleSerializer, SerializeContext}
import com.intel.analytics.zoo.models.image.common.ImageModel
import com.intel.analytics.zoo.models.image.objectdetection.common.OBUtils
import com.intel.analytics.zoo.models.image.objectdetection.common.nn.{AnchorTarget, BboxPred, EvaluateOnly, ProposalTarget}

import scala.reflect.ClassTag
import scala.reflect.runtime._
import com.intel.analytics.bigdl.numeric.NumericFloat

object VggFRcnn {
//  ModuleSerializer.registerModule(
//    "com.intel.analytics.zoo.models.image.objectdetection.fasterrcnn.VggFRcnn", VggFRcnn)

  def vgg16(data: ModuleNode[Float])(implicit ev: TensorNumeric[Float]): ModuleNode[Float] = {
    val conv1_1 = SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1, propagateBack = false)
      .setInitMethod(weightInitMethod = Xavier, biasInitMethod = Zeros)
      .setName(s"conv1_1").inputs(data)
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
    relu5_3
  }

  val rpnPreNmsTopNTest = 6000
  val rpnPostNmsTopNTest = 300
  var debug = false
  val ratios = Array[Float](0.5f, 1.0f, 2.0f)
  val scales = Array[Float](8, 16, 32)
  val anchorNum = ratios.length * scales.length

  def apply(classNum: Int, postProcessParam: PostProcessParam): Module[Float] = {
//    new VggFRcnn(classNum, postProcessParam).build()
val data = Input()
    val imInfo = Input()
    // for training only
    val gt = Input()
    val vgg = vgg16(data)
    // val rpnNet = rpn(vgg, imInfo)
    val rpn_conv_3x3 = SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)
      .setName("rpn_conv/3x3").inputs(vgg)
    val relu3x3 = ReLU(true).setName("rpn_relu/3x3").inputs(rpn_conv_3x3)
    val rpn_cls_score = SpatialConvolution(512, 18, 1, 1, 1, 1)
      .setName("rpn_cls_score").inputs(relu3x3)
    val rpn_cls_score_reshape = InferReshape(Array(0, 2, -1, 0))
      .setName("rpn_cls_score_reshape").inputs(rpn_cls_score)
    val rpn_cls_prob = SoftMax().setName("rpn_cls_prob").inputs(rpn_cls_score_reshape)
    val rpn_cls_prob_reshape = InferReshape(Array(1, 2 * VggFRcnn.anchorNum, -1, 0))
      .setName("rpn_cls_prob_reshape").inputs(rpn_cls_prob)
    val rpn_bbox_pred = SpatialConvolution(512, 36, 1, 1, 1, 1).setName("rpn_bbox_pred")
      .inputs(relu3x3)
    val proposal = Proposal(VggFRcnn.rpnPreNmsTopNTest, VggFRcnn.rpnPostNmsTopNTest,
      VggFRcnn.ratios, VggFRcnn.scales).setName("proposal")
      .inputs(rpn_cls_prob_reshape, rpn_bbox_pred, imInfo)

    val roi_data = ProposalTarget(128, classNum).setName("roi-data").setDebug(VggFRcnn.debug)
      .inputs(proposal, gt)
    val roi = SelectTable(1).setName("roi").inputs(roi_data)
    // val (clsProb, bboxPred) = fastRcnn(vgg, rpnNet)
    val pool = 7
    val roiPooling = RoiPooling(pool, pool, 0.0625f).setName("pool5").inputs(vgg, roi)
    val reshape = InferReshape(Array(-1, 512 * pool * pool))
      .setName("pool5_reshape").inputs(roiPooling)
    val fc6 = Linear(512 * pool * pool, 4096).setName("fc6").inputs(reshape)
    val reLU6 = ReLU().inputs(fc6)
    val dropout6 = Dropout().setName("drop6").inputs(reLU6)
    val fc7 = if (!VggFRcnn.debug) Linear(4096, 4096).setName("fc7").inputs(dropout6)
    else Linear(4096, 4096).setName("fc7").inputs(reLU6)
    val reLU7 = ReLU().inputs(fc7)
    val dropout7 = Dropout().setName("drop7").inputs(reLU7)
    val cls_score = if (!VggFRcnn.debug) Linear(4096, classNum).setName("cls_score").inputs(dropout7)
    else Linear(4096, classNum).setName("cls_score").inputs(reLU7)
    val cls_prob = EvaluateOnly(SoftMax().setName("cls_prob")
      .asInstanceOf[Module[Float]]).inputs(cls_score)
    val bbox_pred = if (!VggFRcnn.debug) BboxPred(4096, classNum * 4, nClass = classNum)
      .setName("bbox_pred").inputs(dropout7)
    else BboxPred(4096, classNum * 4, nClass = classNum).setName("bbox_pred").inputs(reLU7)

    // Training part
    val rpn_data = AnchorTarget(VggFRcnn.ratios, VggFRcnn.scales).setName("rpn-data").setDebug(VggFRcnn.debug)
      .inputs(rpn_cls_score, gt, imInfo, data)

    val detectionOut = DetectionOutputFrcnn(postProcessParam.nmsThresh, postProcessParam.nClasses,
      postProcessParam.bboxVote, postProcessParam.maxPerImage, postProcessParam.thresh).inputs(
      imInfo, roi_data, bbox_pred, cls_prob,
      rpn_cls_score_reshape, rpn_bbox_pred, rpn_data)
    val vggfrcnn = Graph(Array(data, imInfo, gt), detectionOut)
    vggfrcnn.setScaleB(2)
    vggfrcnn.stopGradient(Array("rpn-data", "roi-data", "proposal", "roi", "conv3_1"))
    vggfrcnn
  }

//  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
//    builder : BigDLModule.Builder)
//   (implicit ev: TensorNumeric[T]) : Unit = {
//    val model = context.moduleData.module.asInstanceOf[VggFRcnn]
//    val classNumBuilder = AttrValue.newBuilder
//    DataConverter.setAttributeValue(context,
//      classNumBuilder, model.classNum, universe.typeOf[Int])
//    builder.putAttr("classNum", classNumBuilder.build)
//
//    val nClassesBuilder = AttrValue.newBuilder
//    DataConverter.setAttributeValue(context, nClassesBuilder,
//      model.postProcessParam.nClasses, universe.typeOf[Int])
//    builder.putAttr("nClasses", nClassesBuilder.build)
//
//    val bboxVoteBuilder = AttrValue.newBuilder
//    DataConverter.setAttributeValue(context, bboxVoteBuilder,
//      model.postProcessParam.bboxVote, universe.typeOf[Boolean])
//    builder.putAttr("bboxVote", bboxVoteBuilder.build)
//
//    val maxPerImageBuilder = AttrValue.newBuilder
//    DataConverter.setAttributeValue(context, maxPerImageBuilder,
//      model.postProcessParam.maxPerImage, universe.typeOf[Int])
//    builder.putAttr("maxPerImage", maxPerImageBuilder.build)
//
//    val nmsThreshBuilder = AttrValue.newBuilder
//    DataConverter.setAttributeValue(context, nmsThreshBuilder,
//      model.postProcessParam.nmsThresh, universe.typeOf[Float])
//    builder.putAttr("nmsThresh", nmsThreshBuilder.build)
//
//    val threshBuilder = AttrValue.newBuilder
//    DataConverter.setAttributeValue(context, threshBuilder,
//      model.postProcessParam.thresh, universe.typeOf[Double])
//    builder.putAttr("thresh", threshBuilder.build)
//  }
//
//  override def doLoadModule[T: ClassTag](context: DeserializeContext)
//    (implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {
//    val attrMap = context.bigdlModule.getAttrMap
//
//    val classNum = DataConverter
//      .getAttributeValue(context, attrMap.get("classNum"))
//      .asInstanceOf[Int]
//    val nClasses = DataConverter
//      .getAttributeValue(context, attrMap.get("nClasses"))
//      .asInstanceOf[Int]
//    val bboxVote = DataConverter
//      .getAttributeValue(context, attrMap.get("bboxVote"))
//      .asInstanceOf[Boolean]
//    val maxPerImage = DataConverter
//      .getAttributeValue(context, attrMap.get("maxPerImage"))
//      .asInstanceOf[Int]
//    val nmsThresh = DataConverter
//      .getAttributeValue(context, attrMap.get("nmsThresh"))
//      .asInstanceOf[Float]
//    val thresh = DataConverter
//      .getAttributeValue(context, attrMap.get("thresh"))
//      .asInstanceOf[Double]
//    VggFRcnn(classNum, PostProcessParam(nmsThresh, nClasses,
//      bboxVote, maxPerImage, thresh)).asInstanceOf[AbstractModule[Activity, Activity, T]]
//  }
}

// Cannot use T here as some layer defined in BigDL hard coded Float
//class VggFRcnn private (val classNum: Int, val postProcessParam: PostProcessParam)
//  (implicit ev: TensorNumeric[Float]) extends ImageModel[Float] {
//
//  override def buildModel(): AbstractModule[Activity, Activity, Float] = {
//    val data = Input()
//    val imInfo = Input()
//    // for training only
//    val gt = Input()
//    val vgg = VggFRcnn.vgg16(data)
//    // val rpnNet = rpn(vgg, imInfo)
//    val rpn_conv_3x3 = SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)
//      .setName("rpn_conv/3x3").inputs(vgg)
//    val relu3x3 = ReLU(true).setName("rpn_relu/3x3").inputs(rpn_conv_3x3)
//    val rpn_cls_score = SpatialConvolution(512, 18, 1, 1, 1, 1)
//      .setName("rpn_cls_score").inputs(relu3x3)
//    val rpn_cls_score_reshape = InferReshape(Array(0, 2, -1, 0))
//      .setName("rpn_cls_score_reshape").inputs(rpn_cls_score)
//    val rpn_cls_prob = SoftMax().setName("rpn_cls_prob").inputs(rpn_cls_score_reshape)
//    val rpn_cls_prob_reshape = InferReshape(Array(1, 2 * VggFRcnn.anchorNum, -1, 0))
//      .setName("rpn_cls_prob_reshape").inputs(rpn_cls_prob)
//    val rpn_bbox_pred = SpatialConvolution(512, 36, 1, 1, 1, 1).setName("rpn_bbox_pred")
//      .inputs(relu3x3)
//    val proposal = Proposal(VggFRcnn.rpnPreNmsTopNTest, VggFRcnn.rpnPostNmsTopNTest,
//      VggFRcnn.ratios, VggFRcnn.scales).setName("proposal")
//      .inputs(rpn_cls_prob_reshape, rpn_bbox_pred, imInfo)
//
//
//    val roi_data = ProposalTarget(128, classNum).setName("roi-data").setDebug(VggFRcnn.debug)
//      .inputs(proposal, gt)
//    val roi = SelectTable(1).setName("roi").inputs(roi_data)
//    // val (clsProb, bboxPred) = fastRcnn(vgg, rpnNet)
//    val pool = 7
//    val roiPooling = RoiPooling(pool, pool, 0.0625f).setName("pool5").inputs(vgg, roi)
//    val reshape = InferReshape(Array(-1, 512 * pool * pool))
//      .setName("pool5_reshape").inputs(roiPooling)
//    val fc6 = Linear(512 * pool * pool, 4096).setName("fc6").inputs(reshape)
//    val reLU6 = ReLU().inputs(fc6)
//    val dropout6 = Dropout().setName("drop6").inputs(reLU6)
//    val fc7 = if (!VggFRcnn.debug) Linear(4096, 4096).setName("fc7").inputs(dropout6)
//    else Linear(4096, 4096).setName("fc7").inputs(reLU6)
//    val reLU7 = ReLU().inputs(fc7)
//    val dropout7 = Dropout().setName("drop7").inputs(reLU7)
//    val cls_score = if (!VggFRcnn.debug) Linear(4096, classNum).setName("cls_score").inputs(dropout7)
//    else Linear(4096, classNum).setName("cls_score").inputs(reLU7)
//    val cls_prob = EvaluateOnly(SoftMax().setName("cls_prob")
//      .asInstanceOf[Module[Float]]).inputs(cls_score)
//    val bbox_pred = if (!VggFRcnn.debug) BboxPred(4096, classNum * 4, nClass = classNum)
//      .setName("bbox_pred").inputs(dropout7)
//    else BboxPred(4096, classNum * 4, nClass = classNum).setName("bbox_pred").inputs(reLU7)
//
//    // Training part
//    val rpn_data = AnchorTarget(VggFRcnn.ratios, VggFRcnn.scales).setName("rpn-data").setDebug(VggFRcnn.debug)
//      .inputs(rpn_cls_score, gt, imInfo, data)
//
//    val detectionOut = DetectionOutputFrcnn(postProcessParam.nmsThresh, postProcessParam.nClasses,
//      postProcessParam.bboxVote, postProcessParam.maxPerImage, postProcessParam.thresh).inputs(
//      imInfo, roi_data, bbox_pred, cls_prob,
//      rpn_cls_score_reshape, rpn_bbox_pred, rpn_data)
//    val model = Sequential()
//    val vggfrcnn = Graph(Array(data, imInfo, gt), detectionOut)
//    vggfrcnn.setScaleB(2)
//    vggfrcnn.stopGradient(Array("rpn-data", "roi-data", "proposal", "roi", "conv3_1"))
//    model.add(vggfrcnn)
//
//    model
//  }
//}
