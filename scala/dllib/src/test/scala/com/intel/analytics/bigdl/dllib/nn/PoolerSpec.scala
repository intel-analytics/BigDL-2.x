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

package com.intel.analytics.bigdl.nn

import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.util.Random

class PoolerSpec extends FlatSpec with Matchers {
  "updateOutput Float type" should "work properly" in {
    val feature0 = Array(
      0.883362829685211182, 0.017709493637084961, 0.740627527236938477,
      0.975574254989624023, 0.904063880443572998, 0.293959677219390869,
      0.301572918891906738, 0.235482156276702881)

    val feature1 = Array(
      0.873747766017913818, 0.145658850669860840, 0.256294071674346924,
      0.280913352966308594, 0.062630355358123779, 0.272662281990051270,
      0.524160504341125488, 0.110454082489013672, 0.619955241680145264,
      0.568557560443878174, 0.214293479919433594, 0.648296296596527100,
      0.165463507175445557, 0.419352889060974121, 0.852317929267883301,
      0.628634154796600342, 0.678495228290557861, 0.896998584270477295,
      0.890723347663879395, 0.488525688648223877, 0.384370744228363037,
      0.571207761764526367, 0.788873314857482910, 0.954643964767456055,
      0.969983577728271484, 0.203537940979003906, 0.782353222370147705,
      0.848326086997985840, 0.304318606853485107, 0.800064325332641602,
      0.424848318099975586, 0.603751122951507568)

    val feature2 = Array(
      0.023863613605499268, 0.100520193576812744, 0.579659581184387207,
      0.491799056529998779, 0.695049762725830078, 0.174113810062408447,
      0.514802277088165283, 0.645381748676300049, 0.610754907131195068,
      0.642783403396606445, 0.261436760425567627, 0.865309834480285645,
      0.779586195945739746, 0.805720150470733643, 0.039021611213684082,
      0.052066206932067871, 0.859684348106384277, 0.286012887954711914,
      0.183007895946502686, 0.657920598983764648, 0.486495614051818848,
      0.339991390705108643, 0.349600136280059814, 0.292829811573028564,
      0.874850273132324219, 0.923728287220001221, 0.853209257125854492,
      0.078126728534698486, 0.975298523902893066, 0.889039456844329834,
      0.757552802562713623, 0.009770631790161133, 0.639949500560760498,
      0.384162366390228271, 0.993775784969329834, 0.225636243820190430,
      0.152042329311370850, 0.518522977828979492, 0.346138358116149902,
      0.560805261135101318, 0.197446644306182861, 0.270632088184356689,
      0.537619173526763916, 0.282237291336059570, 0.418838739395141602,
      0.348786175251007080, 0.827486872673034668, 0.671141088008880615,
      0.734223365783691406, 0.461709976196289062, 0.463822364807128906,
      0.256826639175415039, 0.187998294830322266, 0.387186825275421143,
      0.027970135211944580, 0.336534321308135986, 0.078408479690551758,
      0.748133420944213867, 0.996697187423706055, 0.590924799442291260,
      0.363863050937652588, 0.244512259960174561, 0.605456709861755371,
      0.989919960498809814, 0.998104333877563477, 0.318823933601379395,
      0.293298780918121338, 0.240437865257263184, 0.269145488739013672,
      0.321916043758392334, 0.241542100906372070, 0.097301602363586426,
      0.139740049839019775, 0.727295756340026855, 0.735020518302917480,
      0.977046966552734375, 0.562069535255432129, 0.962157845497131348,
      0.896494269371032715, 0.919544279575347900, 0.769982337951660156,
      0.902598083019256592, 0.699079096317291260, 0.970299720764160156,
      0.877977848052978516, 0.445257008075714111, 0.903108179569244385,
      0.029258608818054199, 0.953712522983551025, 0.740538537502288818,
      0.229142010211944580, 0.324616789817810059, 0.546005189418792725,
      0.471910834312438965, 0.479964077472686768, 0.404208302497863770,
      0.816056787967681885, 0.116290867328643799, 0.845461726188659668,
      0.313867926597595215, 0.281320571899414062, 0.693770170211791992,
      0.623112499713897705, 0.370123684406280518, 0.595665276050567627,
      0.433298051357269287, 0.971214890480041504, 0.087709188461303711,
      0.069373369216918945, 0.274347186088562012, 0.470574259757995605,
      0.883642554283142090, 0.518250524997711182, 0.118440926074981689,
      0.606658637523651123, 0.529120385646820068, 0.991135418415069580,
      0.020969033241271973, 0.601271688938140869, 0.031737148761749268,
      0.699844896793365479, 0.006896257400512695, 0.478346049785614014,
      0.267558634281158447, 0.762180626392364502, 0.907826840877532959,
      0.316000878810882568, 0.405982732772827148)

    val features = new Table()
    features.insert(Tensor(Storage(feature0.map(x => x.toFloat))).resize(1, 2, 2, 2))
    features.insert(Tensor(Storage(feature1.map(x => x.toFloat))).resize(1, 2, 4, 4))
    features.insert(Tensor(Storage(feature2.map(x => x.toFloat))).resize(1, 2, 8, 8))
    val rois = Tensor[Float](
      T(T(0, 0, 3, 3),
        T(2, 2, 50, 50),
        T(50, 50, 500, 500))).resize(3, 4)
    val input = T(features, rois)

    val pooler = Pooler[Float](resolution = 2, scales = Array(1.0f, 0.5f, 0.25f), samplingRatio = 2)
    val res = pooler.forward(input)
    val expectedRes = Array(
      0.710301160812377930, 0.338120758533477783,
      0.451076686382293701, 0.243893563747406006,
      0.327536046504974365, 0.126878187060356140,
      0.128067761659622192, 0.058870539069175720,
      0.157158538699150085, 0.000000000000000000,
      0.000000000000000000, 0.000000000000000000,
      0.150937780737876892, 0.000000000000000000,
      0.000000000000000000, 0.000000000000000000,
      0.000000000000000000, 0.000000000000000000,
      0.000000000000000000, 0.000000000000000000,
      0.000000000000000000, 0.000000000000000000,
      0.000000000000000000, 0.000000000000000000)

    for (i <- expectedRes.indices) {
      assert(Math.abs(res.storage().array()(i) - expectedRes(i)) < 1e-6)
    }
  }

  "updateOutput Double type" should "work properly" in {
    val feature0 = Array(
      0.883362829685211182, 0.017709493637084961, 0.740627527236938477,
      0.975574254989624023, 0.904063880443572998, 0.293959677219390869,
      0.301572918891906738, 0.235482156276702881)

    val feature1 = Array(
      0.873747766017913818, 0.145658850669860840, 0.256294071674346924,
      0.280913352966308594, 0.062630355358123779, 0.272662281990051270,
      0.524160504341125488, 0.110454082489013672, 0.619955241680145264,
      0.568557560443878174, 0.214293479919433594, 0.648296296596527100,
      0.165463507175445557, 0.419352889060974121, 0.852317929267883301,
      0.628634154796600342, 0.678495228290557861, 0.896998584270477295,
      0.890723347663879395, 0.488525688648223877, 0.384370744228363037,
      0.571207761764526367, 0.788873314857482910, 0.954643964767456055,
      0.969983577728271484, 0.203537940979003906, 0.782353222370147705,
      0.848326086997985840, 0.304318606853485107, 0.800064325332641602,
      0.424848318099975586, 0.603751122951507568)

    val feature2 = Array(
      0.023863613605499268, 0.100520193576812744, 0.579659581184387207,
      0.491799056529998779, 0.695049762725830078, 0.174113810062408447,
      0.514802277088165283, 0.645381748676300049, 0.610754907131195068,
      0.642783403396606445, 0.261436760425567627, 0.865309834480285645,
      0.779586195945739746, 0.805720150470733643, 0.039021611213684082,
      0.052066206932067871, 0.859684348106384277, 0.286012887954711914,
      0.183007895946502686, 0.657920598983764648, 0.486495614051818848,
      0.339991390705108643, 0.349600136280059814, 0.292829811573028564,
      0.874850273132324219, 0.923728287220001221, 0.853209257125854492,
      0.078126728534698486, 0.975298523902893066, 0.889039456844329834,
      0.757552802562713623, 0.009770631790161133, 0.639949500560760498,
      0.384162366390228271, 0.993775784969329834, 0.225636243820190430,
      0.152042329311370850, 0.518522977828979492, 0.346138358116149902,
      0.560805261135101318, 0.197446644306182861, 0.270632088184356689,
      0.537619173526763916, 0.282237291336059570, 0.418838739395141602,
      0.348786175251007080, 0.827486872673034668, 0.671141088008880615,
      0.734223365783691406, 0.461709976196289062, 0.463822364807128906,
      0.256826639175415039, 0.187998294830322266, 0.387186825275421143,
      0.027970135211944580, 0.336534321308135986, 0.078408479690551758,
      0.748133420944213867, 0.996697187423706055, 0.590924799442291260,
      0.363863050937652588, 0.244512259960174561, 0.605456709861755371,
      0.989919960498809814, 0.998104333877563477, 0.318823933601379395,
      0.293298780918121338, 0.240437865257263184, 0.269145488739013672,
      0.321916043758392334, 0.241542100906372070, 0.097301602363586426,
      0.139740049839019775, 0.727295756340026855, 0.735020518302917480,
      0.977046966552734375, 0.562069535255432129, 0.962157845497131348,
      0.896494269371032715, 0.919544279575347900, 0.769982337951660156,
      0.902598083019256592, 0.699079096317291260, 0.970299720764160156,
      0.877977848052978516, 0.445257008075714111, 0.903108179569244385,
      0.029258608818054199, 0.953712522983551025, 0.740538537502288818,
      0.229142010211944580, 0.324616789817810059, 0.546005189418792725,
      0.471910834312438965, 0.479964077472686768, 0.404208302497863770,
      0.816056787967681885, 0.116290867328643799, 0.845461726188659668,
      0.313867926597595215, 0.281320571899414062, 0.693770170211791992,
      0.623112499713897705, 0.370123684406280518, 0.595665276050567627,
      0.433298051357269287, 0.971214890480041504, 0.087709188461303711,
      0.069373369216918945, 0.274347186088562012, 0.470574259757995605,
      0.883642554283142090, 0.518250524997711182, 0.118440926074981689,
      0.606658637523651123, 0.529120385646820068, 0.991135418415069580,
      0.020969033241271973, 0.601271688938140869, 0.031737148761749268,
      0.699844896793365479, 0.006896257400512695, 0.478346049785614014,
      0.267558634281158447, 0.762180626392364502, 0.907826840877532959,
      0.316000878810882568, 0.405982732772827148)

    val features = new Table()
    features.insert(Tensor(Storage(feature0.map(x => x))).resize(1, 2, 2, 2))
    features.insert(Tensor(Storage(feature1.map(x => x))).resize(1, 2, 4, 4))
    features.insert(Tensor(Storage(feature2.map(x => x))).resize(1, 2, 8, 8))
    val rois = Tensor[Double](
      T(T(0, 0, 3, 3),
        T(2, 2, 50, 50),
        T(50, 50, 500, 500))).resize(3, 4)
    val input = T(features, rois)

    val pooler = Pooler[Double](resolution = 2, scales = Array(1.0f, 0.5f, 0.25f),
      samplingRatio = 2)
    val res = pooler.forward(input)
    val expectedRes = Array(
      0.710301160812377930, 0.338120758533477783,
      0.451076686382293701, 0.243893563747406006,
      0.327536046504974365, 0.126878187060356140,
      0.128067761659622192, 0.058870539069175720,
      0.157158538699150085, 0.000000000000000000,
      0.000000000000000000, 0.000000000000000000,
      0.150937780737876892, 0.000000000000000000,
      0.000000000000000000, 0.000000000000000000,
      0.000000000000000000, 0.000000000000000000,
      0.000000000000000000, 0.000000000000000000,
      0.000000000000000000, 0.000000000000000000,
      0.000000000000000000, 0.000000000000000000)

    for (i <- expectedRes.indices) {
      assert(Math.abs(res.storage().array()(i) - expectedRes(i)) < 1e-6)
    }
  }
}

class PoolerSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val input = T()
    val feature0 = Tensor[Float](1, 1, 2, 2).apply1(_ => Random.nextFloat())
    val feature1 = Tensor[Float](1, 1, 4, 4).apply1(_ => Random.nextFloat())
    val feature2 = Tensor[Float](1, 1, 8, 8).apply1(_ => Random.nextFloat())
    val features = T(feature0, feature1, feature2)
    val rois = Tensor[Float](1, 4).apply1(_ => Random.nextFloat())
    input(1.0f) = features
    input(2.0f) = rois
    val pooler = new Pooler[Float](resolution = 2, scales = Array(1.0f, 0.5f, 0.25f),
      samplingRatio = 2).setName("pooler")
    runSerializationTest(pooler, input)
  }
}
