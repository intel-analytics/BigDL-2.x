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

package com.intel.analytics.bigdl.models.maskrcnn

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.segmentation.RLEMasks
import com.intel.analytics.bigdl.nn.Nms
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.RoiImageInfo
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import com.intel.analytics.bigdl.utils.{RandomGenerator, T, Table}
import org.scalatest.{FlatSpec, Matchers}

class MaskRCNNSpec extends FlatSpec with Matchers {
  "build maskrcnn" should "be ok" in {
    RandomGenerator.RNG.setSeed(100)
    val resNetOutChannels = 32
    val backboneOutChannels = 32
    val mask = new MaskRCNN(resNetOutChannels, backboneOutChannels)
    mask.evaluate()
    val input = Tensor[Float](1, 3, 224, 256).rand()
    val output = mask.forward(T(input, Tensor[Float](T(T(224f, 256f, 224f, 256f)))))
  }

  "build maskrcnn with batch size > 1" should "be ok" in {
    RandomGenerator.RNG.setSeed(1)
    val resNetOutChannels = 32
    val backboneOutChannels = 32
    val mask = new MaskRCNN(resNetOutChannels, backboneOutChannels)
    mask.evaluate()
    val maskBatch = mask.asInstanceOf[Module[Float]].cloneModule()
    maskBatch.evaluate()
    val mask3 = mask.asInstanceOf[Module[Float]].cloneModule()
    mask3.evaluate()

    val input1 = Tensor[Float](1, 3, 224, 256).rand()
    val input2 = Tensor[Float](1, 3, 224, 256).rand()

    val input = Tensor[Float](2, 3, 224, 256)
    input.narrow(1, 1, 1).copy(input1)
    input.narrow(1, 2, 1).copy(input2)

    val output1 = mask.forward(T(input1,
      Tensor[Float](T(T(224f, 256f, 224f, 256f))))).toTable[Table](1)
    val output2 = mask3.forward(T(input2,
      Tensor[Float](T(T(224f, 256f, 224f, 256f))))).toTable[Table](1)
    val output = maskBatch.forward(T(input,
      Tensor[Float](T(T(224f, 256f, 224f, 256f), T(224f, 256f, 224f, 256f))))).toTable
    val first = output[Table](1)
    val second = output[Table](2)

    first.get[Tensor[Float]](RoiImageInfo.BBOXES) should be(
      output1.get[Tensor[Float]](RoiImageInfo.BBOXES))
    first.get[Tensor[Float]](RoiImageInfo.CLASSES) should be(
      output1.get[Tensor[Float]](RoiImageInfo.CLASSES))
    first.get[Tensor[Float]](RoiImageInfo.SCORES) should be(
      output1.get[Tensor[Float]](RoiImageInfo.SCORES))

    second.get[Tensor[Float]](RoiImageInfo.BBOXES) should be(
      output2.get[Tensor[Float]](RoiImageInfo.BBOXES))
    second.get[Tensor[Float]](RoiImageInfo.CLASSES) should be(
      output2.get[Tensor[Float]](RoiImageInfo.CLASSES))
    second.get[Tensor[Float]](RoiImageInfo.SCORES) should be(
      output2.get[Tensor[Float]](RoiImageInfo.SCORES))

    // for masks
    val firstMasks = first.get[Array[RLEMasks]](RoiImageInfo.MASKS).get
    val expectedMasks = output1.get[Array[RLEMasks]](RoiImageInfo.MASKS).get
    for (i <- 0 to firstMasks.length - 1) {
      firstMasks(i).counts should be(expectedMasks(i).counts)
    }

    val secondMasks = second.get[Array[RLEMasks]](RoiImageInfo.MASKS).get
    val expectedMasks2 = output2.get[Array[RLEMasks]](RoiImageInfo.MASKS).get

    for (i <- 0 to secondMasks.length - 1) {
      secondMasks(i).counts should be(expectedMasks2(i).counts)
    }
  }

  "NMS" should "be ok" in {
    val boxes = Tensor[Float](T(
      T(18.0357, 0.0000, 41.2893, 37.1173),
      T(30.0285, 6.2588, 53.1850, 39.0000),
      T(26.0422, 0.0000, 49.1954, 39.0000),
      T( 5.9485, 14.0573, 29.1708, 39.0000),
      T(42.0456, 0.0000, 57.0000, 37.1553),
      T(21.9588, 14.0357, 45.1161, 39.0000),
      T( 6.0533, 0.0000, 29.4083, 39.0000),
      T( 2.0541, 2.3791, 25.4243, 39.0000),
      T(14.0495, 2.3053, 37.3108, 39.0000),
      T(46.0309, 6.4025, 57.0000, 39.0000),
      T(22.0302, 2.4089, 45.1933, 39.0000),
      T(13.9671, 14.0175, 37.1495, 39.0000),
      T(10.0404, 0.0000, 33.3284, 33.2829),
      T(34.0374, 0.0000, 57.0000, 36.9072),
      T(38.0379, 6.2769, 57.0000, 39.0000),
      T(41.9751, 14.0583, 57.0000, 39.0000),
      T( 0.0000, 0.0000, 13.2693, 33.3124),
      T(38.0422, 0.0000, 57.0000, 28.9761),
      T( 0.0000, 14.0690, 17.1186, 39.0000),
      T( 0.0000, 6.0356, 13.2223, 39.0000),
      T( 0.0000, 0.0000, 17.3122, 39.0000),
      T(22.0270, 0.0000, 45.1928, 25.2032),
      T(46.0094, 0.0000, 57.0000, 33.0826),
      T( 0.0000, 0.0000, 33.7101, 13.0355),
      T( 2.0302, 0.0000, 25.4260, 25.4481),
      T(42.0226, 0.0000, 57.0000, 25.1449),
      T(30.0364, 0.0000, 53.0853, 25.0766),
      T(14.0171, 0.0000, 37.2881, 25.2999),
      T(34.0521, 0.0000, 57.0000, 12.9051),
      T( 0.0000, 3.8999, 57.0000, 39.0000),
      T( 2.0133, 0.0000, 49.6427, 12.9898),
      T(28.0456, 0.0000, 57.0000, 39.0000),
      T( 0.0000, 11.8925, 47.3868, 39.0000),
      T( 8.0708, 11.9606, 57.0000, 39.0000),
      T( 0.0000, 0.0000, 27.2810, 39.0000),
      T( 0.0000, 0.0000, 47.4577, 35.2592),
      T( 0.0000, 0.0000, 57.0000, 39.0000),
      T( 0.0000, 0.0000, 57.0000, 39.0000),
      T(21.9457, 0.0000, 57.0000, 12.8811),
      T( 0.0000, 0.0000, 57.0000, 39.0000),
      T( 0.0000, 0.0000, 57.0000, 27.0690),
      T(13.8674, 22.0563, 44.9398, 39.0000),
      T(33.8700, 25.9730, 57.0000, 39.0000),
      T( 0.0000, 22.0516, 20.9330, 39.0000),
      T(41.9213, 21.9873, 57.0000, 39.0000),
      T(17.8165, 0.0000, 57.0000, 16.8779),
      T( 1.7646, 18.1004, 32.9480, 39.0000),
      T(11.8512, 0.0000, 57.0000, 35.4317),
      T(29.8503, 22.0435, 57.0000, 39.0000),
      T( 9.7594, 18.0566, 40.9166, 39.0000),
      T(33.7746, 1.9632, 57.0000, 24.9071),
      T( 0.0000, 14.0776, 24.9558, 39.0000),
      T(21.7241, 18.0735, 52.8998, 39.0000),
      T( 0.0000, 0.0000, 29.2906, 29.5339),
      T(41.8249, 0.0000, 57.0000, 17.0812),
      T( 0.0000, 0.0000, 17.3257, 17.4717),
      T( 0.0000, 0.0000, 17.1572, 25.5946),
      T( 0.0000, 0.0000, 45.4454, 17.0065),
      T( 0.0000, 2.0042, 21.2122, 33.4895),
      T(37.8946, 18.1178, 57.0000, 39.0000),
      T( 0.0000, 5.9850, 25.1862, 29.1060),
      T( 1.7353, 6.0499, 33.1671, 37.4231),
      T(21.6518, 26.0054, 57.0000, 39.0000),
      T( 5.7049, 0.0000, 37.2819, 29.4436),
      T(29.7011, 14.0272, 57.0000, 39.0000),
      T(17.7255, 0.0000, 49.0772, 29.2946),
      T(29.6133, 9.9153, 57.0000, 32.7949),
      T( 0.0000, 26.0193, 32.8463, 39.0000),
      T(17.6348, 10.0788, 48.9423, 39.0000),
      T(21.6906, 2.1241, 52.9483, 33.3707),
      T( 5.6194, 0.0000, 53.3307, 21.0163),
      T(13.8104, 0.0000, 45.2210, 17.3200),
      T(13.5956, 9.9687, 57.0000, 32.8566),
      T( 5.7003, 10.0389, 37.0897, 39.0000),
      T(13.7149, 2.0202, 45.0843, 33.2768),
      T( 9.7322, 5.9888, 41.1038, 37.3045),
      T( 5.5910, 26.0368, 52.8697, 39.0000),
      T(29.7840, 0.0000, 57.0000, 17.1027),
      T( 5.7736, 0.0000, 37.3917, 17.4214),
      T( 0.0000, 13.9622, 36.9701, 36.8555),
      T( 0.0000, 9.9967, 45.0663, 32.9533),
      T( 0.0000, 0.0000, 33.2938, 21.2008),
      T( 0.0000, 0.0000, 25.3888, 17.4817),
      T(21.7062, 0.0000, 53.0319, 21.2508),
      T( 9.6736, 0.0000, 41.2481, 21.3898),
      T( 0.0000, 1.9933, 37.2186, 25.1230),
      T( 5.5202, 5.9523, 53.1432, 28.9392),
      T(25.5138, 5.9795, 57.0000, 28.8653),
      T( 0.0000, 10.0011, 28.9181, 33.0324),
      T( 5.5488, 14.0092, 52.8771, 36.8956),
      T( 9.5096, 1.9473, 57.0000, 24.9822),
      T(17.5084, 13.9728, 57.0000, 36.8385),
      T( 0.0000, 22.0156, 40.7790, 39.0000),
      T(17.5165, 22.0209, 57.0000, 39.0000),
      T( 9.5040, 17.9792, 56.7784, 39.0000),
      T( 0.0000, 5.9792, 41.1165, 29.0066)))

    val scores = Tensor[Float](
      T(0.1117, 0.8158, 0.2626, 0.4839, 0.6765, 0.7539, 0.2627, 0.0428, 0.2080,
      0.1180, 0.1217, 0.7356, 0.7118, 0.7876, 0.4183, 0.9014, 0.9969, 0.7565,
      0.2239, 0.3023, 0.1784, 0.8238, 0.5557, 0.9770, 0.4440, 0.9478, 0.7445,
      0.4892, 0.2426, 0.7003, 0.5277, 0.2472, 0.7909, 0.4235, 0.0169, 0.2209,
      0.9535, 0.7064, 0.1629, 0.8902, 0.5163, 0.0359, 0.6476, 0.3430, 0.3182,
      0.5261, 0.0447, 0.5123, 0.9051, 0.5989, 0.4450, 0.7278, 0.4563, 0.3389,
      0.6211, 0.5530, 0.6896, 0.3687, 0.9053, 0.8356, 0.3039, 0.6726, 0.5740,
      0.9233, 0.9178, 0.7590, 0.7775, 0.6179, 0.3379, 0.2170, 0.9454, 0.7116,
      0.1157, 0.6574, 0.3451, 0.0453, 0.9798, 0.5548, 0.6868, 0.4920, 0.0748,
      0.9605, 0.3271, 0.0103, 0.9516, 0.2855, 0.2324, 0.9141, 0.7668, 0.1659,
      0.4393, 0.2243, 0.8935, 0.0497, 0.1780, 0.3011))

    val thresh = 0.5f
    val inds = new Array[Int](scores.nElement())
    val nms = new Nms
    val keepN = nms.nms(scores, boxes, thresh, inds)

    val expectedOutput = Array[Float](2.0f, 5.0f, 8.0f, 9.0f, 16.0f,
      21.0f, 23.0f, 24.0f, 25.0f, 36.0f, 42.0f, 43.0f, 49.0f, 55.0f,
      64.0f, 76.0f, 77.0f, 84.0f, 87.0f, 88.0f)

    for (i <- 0 to keepN - 1) {
      require(expectedOutput.contains(inds(i) - 1), s"${i} ${inds(i)}")
    }
  }

  "NMS test with " should "be ok" in {
    val bbox = Tensor[Float](T(
      T(897.1850, 313.4036, 932.1763, 374.2394),
      T(455.7833, 333.2753, 500.8198, 415.9607),
      T(359.4648, 344.7227, 419.7825, 415.5826),
      T(896.3477, 313.1893, 932.4266, 373.6151),
      T(453.1522, 334.3315, 501.1705, 421.0176),
      T(897.9015, 313.4834, 931.5834, 372.5941),
      T(896.4880, 313.9242, 931.5134, 375.4740),
      T(868.6584, 330.6160, 911.9927, 384.6833),
      T(942.7654, 292.9069, 999.7523, 358.1204),
      T(928.7173, 290.2841, 1019.7722, 345.7566),
      T(993.7571, 297.5816, 1018.6810, 345.1978),
      T(888.2090, 314.3195, 929.4616, 381.5802),
      T(889.7837, 313.1184, 928.6500, 372.4649),
      T(980.4796, 253.2323, 992.5759, 278.4875),
      T(868.4745, 334.2823, 909.0101, 385.3784),
      T(895.6448, 313.4649, 931.4463, 372.2554),
      T(913.7177, 341.4454, 1000.9200, 385.1247),
      T(984.8840, 252.4099, 994.6163, 279.6371),
      T(940.8889, 296.0774, 997.9278, 359.2623),
      T(894.1754, 314.3835, 931.2900, 378.0296),
      T(932.7524, 291.2802, 997.4014, 358.1486),
      T(946.0168, 294.9995, 988.2959, 353.5110),
      T(974.5388, 254.3933, 988.2365, 277.1239),
      T(925.1069, 338.8027, 997.5690, 384.6586),
      T(995.1877, 297.3512, 1019.7518, 343.0744),
      T(985.8417, 252.9171, 995.6226, 281.1850),
      T(975.4414, 254.2575, 987.0934, 275.7443),
      T(896.0717, 313.9886, 931.0652, 377.5052),
      T(867.7359, 337.7741, 907.4892, 386.2584),
      T(896.1373, 313.2400, 931.5672, 374.0488),
      T(938.9003, 295.1502, 997.1716, 361.1665),
      T(982.5619, 252.5668, 994.3489, 276.8817),
      T(896.9540, 314.2319, 932.6215, 375.3799),
      T(933.0647, 292.4633, 1006.3927, 353.1749),
      T(987.3625, 252.8081, 996.3571, 280.9350),
      T(455.1857, 334.0815, 502.3899, 415.9973),
      T(974.3162, 254.5754, 989.2289, 277.5487),
      T(873.7986, 333.4843, 909.5336, 384.7832),
      T(994.9200, 297.4040, 1019.5212, 344.3844),
      T(977.9858, 253.4250, 990.1346, 275.3441),
      T(897.6171, 313.9396, 930.9672, 363.0933),
      T(972.5175, 302.8481, 1003.7957, 352.4216),
      T(952.8575, 314.9540, 996.2098, 360.9853),
      T(897.7755, 312.8476, 932.1740, 375.4126),
      T(935.5133, 308.0755, 999.0497, 368.1589),
      T(896.4603, 314.3011, 933.0221, 374.9068),
      T(946.8304, 296.2131, 984.3329, 345.4001),
      T(974.9713, 254.1632, 985.7785, 273.4439),
      T(921.4911, 284.3484, 988.0775, 343.6074),
      T(453.4486, 334.5969, 501.3417, 416.7791),
      T(879.2617, 324.7776, 913.9814, 380.1909),
      T(896.1531, 315.2972, 929.5491, 377.5840),
      T(976.2934, 254.3697, 992.0207, 281.3517),
      T(359.7283, 345.4523, 422.3232, 416.0405),
      T(987.8149, 253.1223, 996.3849, 282.5437),
      T(977.9693, 308.9249, 1004.4815, 351.7527),
      T(934.2255, 295.4119, 997.7376, 360.3417),
      T(983.8524, 252.6548, 995.5056, 282.6053),
      T(992.8503, 299.1681, 1019.4303, 346.0417),
      T(926.0668, 288.1923, 1005.9279, 360.2895),
      T(921.8798, 283.3901, 958.5684, 335.0197),
      T(892.4288, 316.7297, 924.1523, 377.9731),
      T(865.8591, 336.5531, 912.1065, 386.3830),
      T(898.3209, 313.6562, 933.8464, 375.7697),
      T(949.4941, 295.6840, 981.8075, 335.6603),
      T(944.0931, 295.7336, 1000.0449, 358.7041),
      T(893.5613, 314.3767, 929.3250, 361.8053),
      T(897.1752, 314.2693, 930.7831, 370.4089),
      T(986.4834, 252.4445, 996.8439, 281.0986),
      T(966.5795, 303.4845, 999.5891, 344.3765),
      T(359.3189, 344.2225, 422.1163, 416.8134),
      T(980.1839, 252.5571, 993.9897, 272.0330),
      T(985.8586, 252.1824, 997.0685, 282.9771),
      T(950.0735, 304.3362, 997.5048, 360.0750),
      T(949.5023, 298.7940, 999.4456, 359.3679),
      T(984.8854, 251.3433, 995.3776, 277.1697),
      T(878.3315, 323.2667, 920.9296, 384.7577),
      T(866.8826, 337.7082, 907.4388, 386.9977),
      T(930.4151, 286.1067, 996.6746, 346.9260),
      T(449.7876, 335.0375, 501.5532, 424.3545),
      T(970.6074, 296.2614, 1015.4398, 349.3472),
      T(936.3362, 299.4994, 999.0040, 360.4084),
      T(887.7698, 316.0671, 921.5828, 375.5379),
      T(866.1887, 327.5018, 913.1649, 385.5780),
      T(451.1341, 334.8470, 501.5725, 420.2753),
      T(966.8165, 299.6295, 1001.6013, 350.6080),
      T(929.5203, 292.7051, 961.2527, 342.3767),
      T(985.9116, 252.4055, 995.6019, 278.5398),
      T(928.5327, 317.6143, 998.8132, 375.1267),
      T(924.2203, 286.4263, 964.5058, 340.0036),
      T(993.9672, 298.4504, 1019.6760, 344.0639),
      T(993.6530, 298.9571, 1018.6897, 344.1803),
      T(357.6289, 347.2077, 427.7265, 416.1507),
      T(975.6861, 309.5988, 1001.9472, 345.2947),
      T(1052.2827, 306.5223, 1063.3223, 337.4540),
      T(893.9320, 313.9812, 931.5121, 378.7151),
      T(950.3990, 295.4264, 1002.1595, 355.8525),
      T(927.2559, 289.3035, 998.7040, 362.0621),
      T(973.4485, 307.2058, 998.9310, 339.6187),
      T(865.3060, 335.3534, 912.1841, 386.1739),
      T(872.6038, 336.4193, 910.7700, 385.0566),
      T(871.1727, 318.4342, 923.3215, 383.5176),
      T(923.0536, 282.2944, 972.0072, 336.2596),
      T(985.0390, 308.6352, 1010.7421, 350.6218),
      T(444.0601, 336.1603, 500.7296, 423.0992),
      T(869.2928, 332.5573, 910.4033, 384.0341),
      T(986.0456, 251.7105, 996.0561, 275.1382),
      T(945.4684, 298.7881, 994.5966, 358.8904),
      T(883.4898, 331.6453, 910.9511, 379.8238),
      T(940.1200, 293.8811, 1005.2361, 354.3233),
      T(954.6428, 296.6301, 979.5766, 326.1757),
      T(964.2259, 293.8177, 1016.9499, 345.5342),
      T(949.8438, 294.3049, 992.1930, 348.0120),
      T(994.1414, 297.6946, 1019.7372, 344.3959),
      T(944.2752, 296.6947, 983.6104, 344.6640),
      T(922.5219, 285.9380, 957.7973, 338.0446),
      T(354.8602, 349.5603, 429.7922, 416.0119),
      T(359.9245, 345.8270, 424.7833, 416.7082),
      T(896.8448, 313.5126, 930.7410, 371.4969),
      T(899.2472, 311.8966, 931.7090, 367.2727),
      T(916.7671, 314.6682, 996.5461, 384.1572),
      T(897.3294, 313.7223, 930.9153, 366.3692),
      T(955.1675, 296.3772, 995.6541, 341.4865),
      T(988.6592, 254.4973, 997.6077, 283.3870),
      T(958.2998, 318.5701, 996.8839, 360.5596),
      T(878.6404, 312.4912, 939.2751, 382.1180),
      T(942.3732, 299.2073, 996.5104, 347.8272),
      T(945.8544, 305.3195, 998.3005, 360.8294),
      T(867.2707, 336.2115, 910.3326, 386.3813),
      T(989.5474, 255.2382, 999.1593, 282.3305),
      T(948.8654, 297.4831, 1000.3220, 358.7383),
      T(959.0654, 297.3557, 997.5435, 337.0765),
      T(986.8717, 297.8730, 1021.5273, 346.7177),
      T(923.0396, 284.0523, 967.0013, 338.6024),
      T(920.8279, 283.3512, 966.9508, 337.6205),
      T(975.0892, 253.6959, 987.7636, 270.7398),
      T(983.1747, 252.4163, 993.9336, 280.0854),
      T(897.1261, 312.8062, 931.5692, 365.5327),
      T(925.8576, 282.2936, 989.2410, 340.9687),
      T(457.6447, 333.8348, 502.1255, 419.0621),
      T(929.3680, 317.9347, 1000.9109, 378.6516),
      T(931.9888, 292.2040, 1014.3851, 351.3676),
      T(939.6970, 325.0891, 1002.1588, 377.7377),
      T(937.0275, 294.9764, 1000.0521, 359.9520),
      T(361.4387, 344.3737, 418.4546, 417.0056),
      T(935.3657, 295.6170, 1001.8279, 357.1074),
      T(447.8221, 333.7355, 500.2914, 423.8980),
      T(358.5627, 348.3210, 426.6114, 416.0293),
      T(942.4774, 294.9196, 996.1514, 360.9478),
      T(355.6061, 347.0658, 423.3835, 415.2331),
      T(897.5903, 313.1249, 932.2655, 373.6089),
      T(357.3052, 345.8806, 428.0344, 418.0151),
      T(360.9688, 345.8139, 423.1559, 413.8298),
      T(358.0542, 344.5368, 422.1435, 415.4480),
      T(986.8827, 296.2814, 1030.8202, 344.9389),
      T(869.0630, 334.7263, 913.7510, 386.4895),
      T(449.1287, 333.6480, 505.2426, 424.2687),
      T(921.8153, 329.4345, 992.2134, 385.6496),
      T(359.5635, 344.9244, 423.2573, 415.3024),
      T(878.1603, 312.8528, 928.2896, 383.3929),
      T(872.1131, 324.2969, 917.3246, 384.7457),
      T(897.4950, 318.9093, 940.0261, 381.3441),
      T(448.2094, 334.0672, 501.8153, 423.0515),
      T(929.2242, 293.8395, 1000.8837, 352.6609),
      T(451.7765, 334.3492, 501.5195, 418.6037),
      T(934.4990, 289.2999, 1014.6516, 348.2116),
      T(889.9292, 312.7710, 935.9241, 376.5245),
      T(357.8701, 345.1031, 418.5174, 415.8235),
      T(454.7349, 333.8158, 500.2321, 414.6725),
      T(926.7469, 295.1546, 1001.5960, 361.4129),
      T(947.5048, 293.6343, 999.5144, 359.6602),
      T(357.0127, 346.7641, 437.0735, 436.5526),
      T(359.8571, 344.0298, 424.8551, 413.9603),
      T(888.2206, 312.1265, 946.9496, 365.7358),
      T(361.8871, 346.2571, 425.2443, 415.1584),
      T(931.9264, 344.0161, 1001.4952, 384.5714),
      T(935.9602, 307.9165, 1000.4966, 363.4359),
      T(449.1622, 335.5356, 501.0027, 425.3539),
      T(939.4246, 289.7769, 998.1415, 365.2235),
      T(937.6185, 298.4802, 1001.5556, 360.3358),
      T(913.2161, 300.2504, 997.0823, 371.8651),
      T(925.2327, 286.6145, 998.6547, 360.5739),
      T(452.0296, 333.0158, 502.7156, 423.5693),
      T(956.8554, 294.1949, 1004.6817, 360.6414),
      T(990.3675, 296.7340, 1020.4952, 347.9465),
      T(436.7827, 333.2799, 499.7540, 428.1917),
      T(354.7817, 344.2999, 422.8938, 429.8361),
      T(445.9945, 332.3218, 504.5183, 419.6527),
      T(356.9930, 345.0077, 422.6898, 416.8002),
      T(359.9024, 347.1724, 447.2239, 438.6215),
      T(930.1599, 288.2958, 1007.5668, 367.4672),
      T(890.3512, 307.0296, 986.4042, 383.4467)))

    val scores = Tensor[Float](
      T(0.8895, 0.9511, 0.9799, 0.9506, 0.9808, 0.9182, 0.8226, 0.2990, 0.8350,
        0.3171, 0.8467, 0.6840, 0.2517, 0.2627, 0.3000, 0.8631, 0.0790, 0.5911,
        0.7802, 0.8842, 0.5869, 0.6082, 0.4752, 0.0886, 0.6948, 0.6305, 0.4881,
        0.7345, 0.1136, 0.9514, 0.6845, 0.1704, 0.8708, 0.5591, 0.6080, 0.9622,
        0.4447, 0.3963, 0.5799, 0.0939, 0.5659, 0.1663, 0.4193, 0.7579, 0.3835,
        0.9022, 0.4478, 0.4581, 0.2037, 0.8378, 0.2552, 0.3402, 0.0867, 0.9663,
        0.3352, 0.1342, 0.6891, 0.2075, 0.4518, 0.3642, 0.0553, 0.2398, 0.1638,
        0.4666, 0.4430, 0.7205, 0.0781, 0.9210, 0.4735, 0.0672, 0.9619, 0.0522,
        0.3523, 0.6908, 0.6146, 0.2338, 0.2402, 0.1276, 0.3867, 0.7665, 0.2867,
        0.6170, 0.3110, 0.5327, 0.9125, 0.1714, 0.0521, 0.5585, 0.1243, 0.0681,
        0.6715, 0.5854, 0.3556, 0.0916, 0.0519, 0.7547, 0.5319, 0.4566, 0.0615,
        0.2157, 0.1761, 0.5554, 0.0843, 0.0555, 0.5980, 0.4277, 0.1303, 0.8261,
        0.2421, 0.7401, 0.1352, 0.1726, 0.4677, 0.6657, 0.4990, 0.1112, 0.1743,
        0.9252, 0.8494, 0.4821, 0.3603, 0.7493, 0.3581, 0.0843, 0.1877, 0.0510,
        0.6207, 0.4427, 0.1903, 0.0574, 0.7567, 0.1311, 0.3934, 0.1065, 0.0734,
        0.1276, 0.3197, 0.7413, 0.0748, 0.8815, 0.1857, 0.1483, 0.0995, 0.7282,
        0.9192, 0.6015, 0.6803, 0.0685, 0.7498, 0.2033, 0.8497, 0.6608, 0.9190,
        0.8556, 0.1348, 0.1649, 0.4675, 0.0945, 0.9043, 0.0679, 0.3472, 0.0681,
        0.5856, 0.5952, 0.7874, 0.3340, 0.3464, 0.9608, 0.9078, 0.1791, 0.8079,
        0.0590, 0.1971, 0.0504, 0.8636, 0.0506, 0.2310, 0.5520, 0.5228, 0.2222,
        0.2537, 0.3059, 0.6870, 0.2897, 0.4688, 0.1099, 0.0970, 0.1799, 0.8663,
        0.0548, 0.0747, 0.1079))

    val index = Array[Int](
      1, 2, 6, 9, 11, 12, 14, 18, 20, 21, 22, 23, 24, 26,
      31, 33, 34, 38, 39, 40, 41, 45, 46, 51, 55, 57, 60, 61,
      63, 66, 70, 75, 76, 79, 82, 83, 84, 85, 87, 91, 94, 99,
      102, 105, 107, 110, 113, 121, 130, 134, 136, 140, 145, 146, 159, 161,
      165, 166, 169, 174, 175, 177, 178, 181, 190, 196, 197, 198, 207, 211,
      214, 215, 219, 222, 223, 224, 225, 228, 230, 244, 257, 260, 262, 266,
      270, 273, 277, 282, 284, 285, 293, 295, 297, 302, 308, 315, 316, 329,
      330, 333, 334, 337, 341, 342, 343, 346, 351, 360, 361, 362, 372, 377,
      383, 395, 401, 403, 405, 407, 408, 415, 417, 418, 422, 429, 437, 439,
      445, 449, 455, 457, 459, 476, 482, 485, 490, 492, 495, 498, 506, 512,
      531, 536, 538, 552, 554, 559, 561, 563, 564, 567, 568, 578, 579, 592,
      595, 598, 604, 608, 623, 631, 636, 637, 639, 646, 652, 659, 667, 680,
      696, 698, 713, 714, 739, 744, 755, 760, 773, 776, 780, 786, 804, 824,
      842, 850, 862, 878, 908, 923, 954, 957, 958, 995)

    val expectedOut = Array[Float](2, 4, 8, 10, 13, 25, 26, 29, 42,
      48, 55, 64, 77, 80, 94, 98, 101, 108, 115, 120, 129, 131, 175)

    val nmsTool: Nms = new Nms
    val out = nmsTool.nms(scores, bbox, 0.5f, index, orderWithBBox = true)

    out should be(expectedOut.length)
    for (i <- 0 to (out - 1)) {
      index(i) should be(expectedOut(i) + 1)
    }
  }
}

class MaskRCNNSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val resNetOutChannels = 32
    val backboneOutChannels = 32
    val mask = new MaskRCNN(resNetOutChannels, backboneOutChannels).setName("MaskRCNN")
    mask.evaluate()
    val input = T(Tensor[Float](1, 3, 224, 256).rand(),
      Tensor[Float](T(T(224f, 256f, 224f, 256f))))

    runSerializationTest(mask, input)
  }
}
