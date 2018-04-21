package com.intel.analytics.zoo.models.recommendation

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

class NCFSpec extends FlatSpec with Matchers with BeforeAndAfter {

  "NeuralCF without MF forward and backward" should "work properly" in {
    val userCount = 10
    val itemCount = 10
    val model = NeuralCF[Float](userCount, itemCount, 5, 5, 5, Array(10, 5), false)
    val ran = new Random(42L)
    val data: Seq[Tensor[Float]] = (1 to 50).map(i => {
      val uid = Math.abs(ran.nextInt(userCount - 1)).toFloat + 1
      val iid = Math.abs(ran.nextInt(userCount - 1)).toFloat + 1
      val feature: Tensor[Float] = Tensor(T(T(uid, iid)))
      val label = Math.abs(ran.nextInt(4)).toFloat + 1
      feature
    })
    data.map { input =>
      val output = model.forward(input)
      val gradInput = model.backward(input, output)
    }
  }

  "NeuralCF with MF forward and backward" should "work properly" in {
    val userCount = 10
    val itemCount = 10
    val model = NeuralCF[Float](userCount, itemCount, 5, 5, 5, Array(10, 5), true, 3)
    val ran = new Random(42L)
    val data: Seq[Tensor[Float]] = (1 to 50).map(i => {
      val uid = Math.abs(ran.nextInt(userCount - 1)).toFloat + 1
      val iid = Math.abs(ran.nextInt(userCount - 1)).toFloat + 1
      val feature: Tensor[Float] = Tensor(T(T(uid, iid)))
      val label = Math.abs(ran.nextInt(4)).toFloat + 1
      feature
    })
    data.map { input =>
      val output = model.forward(input)
      val gradInput = model.backward(input, output)
    }
  }

}
