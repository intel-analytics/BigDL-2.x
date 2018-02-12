package com.intel.analytics.zoo.pipeline.model

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.models.resnet.Convolution
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.{DetectionOutputMRcnn, ProposalMaskRcnn, PyramidROIAlign, _}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.transform.vision.image.util.BoundingBox


object MaskRCNN {
  // The strides of each layer of the FPN Pyramid. These values
  // are based on a Resnet101 backbone.
  val BACKBONE_STRIDES = Array(4, 8, 16, 32, 64)
  val IMAGE_MAX_DIM = 1024

  // Input image size
  val IMAGE_SHAPE = Array(IMAGE_MAX_DIM, IMAGE_MAX_DIM, 3)

  // Length of square anchor side in pixels
  val RPN_ANCHOR_SCALES = Array(32, 64, 128, 256, 512)

  // Ratios of anchors at each cell (width/height)
  // A value of 1 represents a square anchor, and 0.5 is a wide anchor
  val RPN_ANCHOR_RATIOS = Array[Float](0.5f, 1, 2)

  // Anchor stride
  // If 1 then anchors are created for each cell in the backbone feature map.
  // If 2, then anchors are created for every other cell, and so on.
  val RPN_ANCHOR_STRIDE = 1

  // Compute backbone size from input image size

  val BACKBONE_SHAPES = BACKBONE_STRIDES.map(stride => {
    ((IMAGE_SHAPE(1) / stride).ceil, (IMAGE_SHAPE(2) / stride).ceil)
  })

  // Pooled ROIs
  val POOL_SIZE = 7
  val MASK_POOL_SIZE = 14

  def identityBlock(nInputPlanes: Int, input: ModuleNode[Float],
    kernelSize: Int, filters: Array[Int],
    stage: Int, block: Any, useBias: Boolean = true): ModuleNode[Float] = {
    val convNameBase = s"res$stage${block}_branch"
    val bnNameBase = s"bn$stage${block}_branch"

    var x = SpatialConvolution(nInputPlanes, filters(0), 1, 1, withBias = useBias)
      .setName(s"${convNameBase}2a").inputs(input)
    x = SpatialBatchNormalization(filters(0), eps = 0.001).setName(bnNameBase + "2a").inputs(x)
    x = ReLU(true).inputs(x)

    x = SpatialConvolution(filters(0), filters(1), kernelSize, kernelSize, padH = -1, padW = -1,
      withBias = useBias).setName(convNameBase + "2b").inputs(x)
    x = SpatialBatchNormalization(filters(1), eps = 0.001).setName(bnNameBase + "2b").inputs(x)
    x = ReLU(true).inputs(x)

    x = SpatialConvolution(filters(1), filters(2), 1, 1, withBias = useBias)
      .setName(convNameBase + "2c").inputs(x)
    x = SpatialBatchNormalization(filters(2), eps = 0.001).setName(bnNameBase + "2c").inputs(x)

    x = CAddTable(true).inputs(x, input)
    x = ReLU(true).setName(s"res$stage${block}_out").inputs(x)

    x
  }

  def convBlock(nInputPlanes: Int, input: ModuleNode[Float],
    kernelSize: Int, filters: Array[Int],
    stage: Int, block: Any, strides: (Int, Int) = (2, 2),
    useBias: Boolean = true): ModuleNode[Float] = {
    val convNameBase = s"res$stage${block}_branch"
    val bnNameBase = s"bn$stage${block}_branch"

    var x = SpatialConvolution(nInputPlanes, filters(0), 1, 1, strides._1, strides._2,
      withBias = useBias).setName(s"${convNameBase}2a").inputs(input)
    x = SpatialBatchNormalization(filters(0), eps = 0.001).setName(bnNameBase + "2a").inputs(x)
    x = ReLU(true).inputs(x)

    x = SpatialConvolution(filters(0), filters(1), kernelSize, kernelSize, padH = -1, padW = -1,
      withBias = useBias).setName(convNameBase + "2b").inputs(x)
    x = SpatialBatchNormalization(filters(1), eps = 0.001).setName(bnNameBase + "2b").inputs(x)
    x = ReLU(true).inputs(x)

    x = SpatialConvolution(filters(1), filters(2), 1, 1, withBias = useBias)
      .setName(convNameBase + "2c").inputs(x)
    x = SpatialBatchNormalization(filters(2), eps = 0.001).setName(bnNameBase + "2c").inputs(x)

    var shortCut = SpatialConvolution(nInputPlanes, filters(2), 1, 1,
      strides._1, strides._2, withBias = useBias).setName(s"${convNameBase}1").inputs(input)
    shortCut = SpatialBatchNormalization(filters(2), eps = 0.001)
      .setName(bnNameBase + "1").inputs(shortCut)

    x = CAddTable(true).inputs(x, shortCut)
    x = ReLU(true).setName(s"res$stage${block}_out").inputs(x)

    x
  }

  def resnetGraph(inputImage: ModuleNode[Float], architecture: String, stage5: Boolean = false,
    optnet: Boolean = true): Array[ModuleNode[Float]] = {
    require(architecture == "resnet50" || architecture == "resnet101")
    // stage1
    var x = SpatialZeroPadding(3, 3, 3, 3).inputs(inputImage)
    x = Convolution(3, 64, 7, 7, 2, 2,
      optnet = optnet, propagateBack = false).setName("conv1").inputs(x)
    x = SpatialBatchNormalization(64, eps = 0.001)
      .setName("bn_conv1").inputs(x)
    x = ReLU(true).inputs(x)
    x = SpatialMaxPooling(3, 3, 2, 2, -1, -1).setName("pool1").inputs(x)
    val C1 = x
    // stage2
    val c21 = convBlock(64, x, 3, Array(64, 64, 256), stage = 2, block = 'a', strides = (1, 1))
    val c22 = identityBlock(256, c21, 3, Array(64, 64, 256), stage = 2, block = 'b')
    x = identityBlock(256, c22, 3, Array(64, 64, 256), stage = 2, block = "c")
    val C2 = x

    // stage3
    x = convBlock(256, x, 3, Array(128, 128, 512), stage = 3, block = "a")
    x = identityBlock(512, x, 3, Array(128, 128, 512), stage = 3, block = "b")
    x = identityBlock(512, x, 3, Array(128, 128, 512), stage = 3, block = "c")
    x = identityBlock(512, x, 3, Array(128, 128, 512), stage = 3, block = "d")
    val C3 = x
    // Stage 4
    x = convBlock(512, x, 3, Array(256, 256, 1024), stage = 4, block = 'a')
    val blockCount = architecture match {
      case "resnet50" => 5
      case "resnet101" => 22
    }
    (0 until blockCount).foreach(i => {
      x = identityBlock(1024, x, 3, Array(256, 256, 1024), stage = 4, block = (98 + i).toChar)
    })
    val C4 = x
    // Stage 5
    val C5 = if (stage5) {
      x = convBlock(1024, x, 3, Array(512, 512, 2048), stage = 5, block = 'a')
      x = identityBlock(2048, x, 3, Array(512, 512, 2048), stage = 5, block = 'b')
      x = identityBlock(2048, x, 3, Array(512, 512, 2048), stage = 5, block = 'c')
      x
    }
    else {
      null
    }
    Array(C1, C2, C3, C4, C5)
  }

  /**
   * Builds the Region Proposal Network.
   * It wraps the RPN graph so it can be used multiple times with shared weights.
   *
   * @param anchorStride Controls the density of anchors. Typically 1 (anchors for
   * every pixel in the feature map), or 2 (every other pixel).
   * @param anchorsPerLocation number of anchors per pixel in the feature map
   * @param depth Depth of the backbone feature map.
   */
  def buildRpnModel(anchorStride: Int, anchorsPerLocation: Int, depth: Int): Module[Float] = {
    val featureMap = Input()
    var shared = SpatialConvolution(256, 512, 3, 3,
      anchorStride, anchorStride, -1, -1)
      .setName("rpn_conv_shared").inputs(featureMap)
    shared = ReLU(true).inputs(shared)
    // Anchor Score. [batch, height, width, anchors per location * 2].
    var x = SpatialConvolution(512, 2 * anchorsPerLocation, 1, 1).setName("rpn_class_raw")
      .inputs(shared)
    x = Transpose(Array((2, 3), (3, 4))).inputs(x)
    // Reshape to [batch, anchors, 2]
    val rpnClassLogits = InferReshape(Array(0, -1, 2)).inputs(x)

    // Softmax on last dimension of BG/FG.
    val rpnProbs = TimeDistributed(SoftMax()).setName("rpn_class_xxx").inputs(rpnClassLogits)
    // Bounding box refinement. [batch, H, W, anchors per location, depth]
    // where depth is [x, y, log(w), log(h)]
    x = SpatialConvolution(512, anchorsPerLocation * 4, 1, 1).setName("rpn_bbox_pred")
      .inputs(shared)
    x = Transpose(Array((2, 3), (3, 4))).inputs(x)
    // Reshape to [batch, anchors, 4]
    val rpnBbox = InferReshape(Array(0, -1, 4)).inputs(x)

    Graph(input = featureMap, output = Array(rpnClassLogits, rpnProbs, rpnBbox))
  }

  def apply(NUM_CLASSES: Int = 81): Module[Float] = {
    val data = Input()
    val imInfo = Input()
    val resnetBranches = resnetGraph(data, "resnet101", stage5 = true)
    val c1 = resnetBranches(0)
    val c2 = resnetBranches(1)
    val c3 = resnetBranches(2)
    val c4 = resnetBranches(3)
    val c5 = resnetBranches(4)

    val p5_add = SpatialConvolution(2048, 256, 1, 1).setName("fpn_c5p5").inputs(c5)
    val fpn_p5upsampled = UpSampling2D(Array(2, 2)).setName("fpn_p5upsampled").inputs(p5_add)
    val fpn_c4p4 = SpatialConvolution(1024, 256, 1, 1).setName("fpn_c4p4").inputs(c4)
    val p4_add = CAddTable().inputs(fpn_p5upsampled, fpn_c4p4)

    val fpn_p4upsampled = UpSampling2D(Array(2, 2)).setName("fpn_p4upsampled").inputs(p4_add)
    val fpn_c3p3 = SpatialConvolution(512, 256, 1, 1).setName("fpn_c3p3").inputs(c3)
    val p3_add = CAddTable().inputs(fpn_p4upsampled, fpn_c3p3)

    val fpn_p3upsampled = UpSampling2D(Array(2, 2)).setName("fpn_p3upsampled").inputs(p3_add)
    val fpn_c2p2 = SpatialConvolution(256, 256, 1, 1).setName("fpn_c2p2").inputs(c2)
    val p2_add = CAddTable().inputs(fpn_p3upsampled, fpn_c2p2)

    // Attach 3x3 conv to all P layers to get the final feature maps.
    val p2 = SpatialConvolution(256, 256, 3, 3, padW = -1, padH = -1)
      .setName("fpn_p2").inputs(p2_add)
    val p3 = SpatialConvolution(256, 256, 3, 3, padW = -1, padH = -1)
      .setName("fpn_p3").inputs(p3_add)
    val p4 = SpatialConvolution(256, 256, 3, 3, padW = -1, padH = -1)
      .setName("fpn_p4").inputs(p4_add)
    val p5 = SpatialConvolution(256, 256, 3, 3, padW = -1, padH = -1)
      .setName("fpn_p5").inputs(p5_add)
    // P6 is used for the 5th anchor scale in RPN. Generated by
    // subsampling from P5 with stride of 2.
    val p6 = SpatialMaxPooling(1, 1, 2, 2).setName("fpn_p6").inputs(p5)
    // Note that P6 is used in RPN, but not in the classifier heads.
    val rpn_feature_maps = Array(p2, p3, p4, p5, p6)
    val mrcnn_feature_maps = Array(p2, p3, p4, p5)

    // RPN Model
    val rpn = buildRpnModel(RPN_ANCHOR_STRIDE, RPN_ANCHOR_RATIOS.length, 256)
    val mapTable = MapTable(rpn).inputs(rpn_feature_maps)

    // Concatenate layer outputs
    // Convert from list of lists of level outputs to list of lists
    // of outputs across levels.
    // e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
    val rpn_class_logits = JoinTable(2, 3).setName("rpn_class_logits")
      .inputs(select(rpn_feature_maps.length, 1, mapTable))
    val rpn_class = JoinTable(2, 3).setName("rpn_class")
      .inputs(select(rpn_feature_maps.length, 2, mapTable))
    val rpn_bbox = JoinTable(2, 3).setName("rpn_bbox")
      .inputs(select(rpn_feature_maps.length, 3, mapTable))

    val rpn_rois = ProposalMaskRcnn(6000, 1000)
      .setName("ROI")
      .inputs(rpn_class, rpn_bbox)

    val (mrcnn_class_logits, mrcnn_class, mrcnn_bbox) =
      fpnClassifierGraph(rpn_rois, mrcnn_feature_maps, IMAGE_SHAPE, POOL_SIZE, NUM_CLASSES)

    val detections = DetectionOutputMRcnn().inputs(rpn_rois, mrcnn_class, mrcnn_bbox, imInfo)
    val detection_boxes = MulConstant(1 / 1024f).inputs(detections)
    val mrcnn_mask = buildFpnMaskGraph(detection_boxes, mrcnn_feature_maps,
      IMAGE_SHAPE,
      MASK_POOL_SIZE,
      NUM_CLASSES)

    Graph(Array(data, imInfo), Array(detections, mrcnn_class, mrcnn_bbox, mrcnn_mask,
      rpn_rois, rpn_class, rpn_bbox)).setName("mask_rcnn")
  }

  def select(total: Int, dim: Int, input: ModuleNode[Float]): Array[ModuleNode[Float]] = {
    require(dim >= 1 && dim <= 3)
    (1 to total).map(i => {
      val level = SelectTable(i).inputs(input)
      SelectTable(dim).inputs(level)
    }).toArray
  }

  /**
   * Builds the computation graph of the feature pyramid network classifier
   * and regressor heads.
   *
   * @param rois [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized coordinates.
   * @param featureMaps List of feature maps from diffent layers of the pyramid,
   * [P2, P3, P4, P5]. Each has a different resolution.
   * @param imageShape : [height, width, depth]
   * @param poolSize : The width of the square feature map generated from ROI Pooling.
   * @param numClasses : number of classes, which determines the depth of the results
   * @return
   * logits: [N, NUM_CLASSES] classifier logits (before softmax)
   * probs: [N, NUM_CLASSES] classifier probabilities
   * bbox_deltas: [N, (dy, dx, log(dh), log(dw))] Deltas to apply to proposal boxes
   */
  def fpnClassifierGraph(rois: ModuleNode[Float], featureMaps: Array[ModuleNode[Float]],
    imageShape: Array[Int], poolSize: Int, numClasses: Int)
  : (ModuleNode[Float], ModuleNode[Float], ModuleNode[Float]) = {
    // ROI Pooling
    // Shape: [batch, num_boxes, pool_height, pool_width, channels]
    var x = PyramidROIAlign(poolSize, poolSize,
      imgH = imageShape(0), imgW = imageShape(1), imgC = imageShape(2))
      .inputs(Array(rois) ++ featureMaps)
    // Two 1024 FC layers (implemented with Conv2D for consistency)
    x = SpatialConvolution(256, 1024, poolSize, poolSize).setName("mrcnn_class_conv1").inputs(x)
    x = SpatialBatchNormalization(1024, eps = 0.001).setName("mrcnn_class_bn1").inputs(x)
    x = ReLU(true).inputs(x)
    x = SpatialConvolution(1024, 1024, 1, 1).setName("mrcnn_class_conv2").inputs(x)
    x = SpatialBatchNormalization(1024, eps = 0.001).setName("mrcnn_class_bn2").inputs(x)
    x = ReLU(true).inputs(x)
    val shared = Squeeze(Array(4, 3), batchMode = false).setName("pool_squeeze").inputs(x)
    // Classifier head
    val mrcnn_class_logits = Linear(1024, numClasses).setName("mrcnn_class_logits")
      .inputs(shared)
    val mrcnn_probs = SoftMax().setName("mrcnn_class").inputs(mrcnn_class_logits)


    // BBox head
    // [batch, boxes, num_classes * (dy, dx, log(dh), log(dw))]
    x = Linear(1024, numClasses * 4).setName("mrcnn_bbox_fc").inputs(shared)
    // Reshape to [batch, boxes, num_classes, (dy, dx, log(dh), log(dw))]
    val mrcnn_bbox = Reshape(Array(1, numClasses, 4)).setName("mrcnn_bbox").inputs(x)
    (mrcnn_class_logits, mrcnn_probs, mrcnn_bbox)
  }

  /**
   *
   * """Builds the computation graph of the mask head of Feature Pyramid Network.
   * *
   * rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
   * coordinates.
   * feature_maps: List of feature maps from diffent layers of the pyramid,
   * [P2, P3, P4, P5]. Each has a different resolution.
   * image_shape: [height, width, depth]
   * pool_size: The width of the square feature map generated from ROI Pooling.
   * num_classes: number of classes, which determines the depth of the results
   * *
   * Returns: Masks [batch, roi_count, height, width, num_classes]
   * """
   *
   * @param rois
   * @param featureMaps
   * @param imageShape
   * @param poolSize
   * @param num_classes
   */
  def buildFpnMaskGraph(rois: ModuleNode[Float], featureMaps: Array[ModuleNode[Float]],
    imageShape: Array[Int], poolSize: Int, num_classes: Int): ModuleNode[Float] = {

    // ROI Pooling
    // Shape: [batch, boxes, pool_height, pool_width, channels]
    var x = PyramidROIAlign(poolSize, poolSize,
      imgH = imageShape(0), imgW = imageShape(1), imgC = imageShape(2)).setName("roi_align_mask")
      .inputs(Array(rois) ++ featureMaps)

    // Conv layers
    x = SpatialConvolution(256, 256, 3, 3, padH = -1, padW = -1)
      .setName("mrcnn_mask_conv1").inputs(x)
    x = SpatialBatchNormalization(256, eps = 0.001).setName("mrcnn_mask_bn1").inputs(x)
    x = ReLU(true).inputs(x)

    x = SpatialConvolution(256, 256, 3, 3, padH = -1, padW = -1)
      .setName("mrcnn_mask_conv2").inputs(x)
    x = SpatialBatchNormalization(256, eps = 0.001).setName("mrcnn_mask_bn2").inputs(x)
    x = ReLU(true).inputs(x)


    x = SpatialConvolution(256, 256, 3, 3, padH = -1, padW = -1)
      .setName("mrcnn_mask_conv3").inputs(x)
    x = SpatialBatchNormalization(256, eps = 0.001).setName("mrcnn_mask_bn3").inputs(x)
    x = ReLU(true).inputs(x)

    x = SpatialConvolution(256, 256, 3, 3, padH = -1, padW = -1)
      .setName("mrcnn_mask_conv4").inputs(x)
    x = SpatialBatchNormalization(256, eps = 0.001).setName("mrcnn_mask_bn4").inputs(x)
    x = ReLU(true).inputs(x)

    x = SpatialFullConvolution(256, 256, 2, 2, 2, 2).setName("mrcnn_mask_deconv").inputs(x)
    x = ReLU(true).inputs(x)
    x = SpatialConvolution(256, num_classes, 1, 1, 1, 1).setName("mrcnn_mask").inputs(x)
    x = Sigmoid().inputs(x)
    x = Unsqueeze(1).inputs(x)
    x
  }

  def visualize(mat: OpenCVMat, rois: Tensor[Float],
    labelMap: Map[Int, String],
    thresh: Double): OpenCVMat = {
    (1 to rois.size(1)).foreach(i => {
      val score = rois.valueAt(i, 2)
      if (score > thresh) {
        val className = labelMap(rois.valueAt(i, 1).toInt)
        val bbox = BoundingBox(rois.valueAt(i, 3), rois.valueAt(i, 4),
          rois.valueAt(i, 5), rois.valueAt(i, 6))
        mat.drawBoundingBox(bbox, s"$className $score")
      }
    })
    mat
  }
}








