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

package com.intel.analytics.zoo.pipeline.inference

import java.util

import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class FloatInferenceModelSpec extends FlatSpec with Matchers with BeforeAndAfter
  with InferenceSupportive {

  val inputTensor1 = Tensor[Float](3, 5, 5).rand()
  val inputTensor2 = Tensor[Float](3, 5, 5).rand()
  val inputTensor3 = Tensor[Float](3, 5, 5).rand()
  val inputJTensor1 = transferTensorToJTensor(inputTensor1)
  val inputJTensor2 = transferTensorToJTensor(inputTensor2)
  val inputJTensor3 = transferTensorToJTensor(inputTensor3)
  val inputTensorsArray = Array(inputJTensor1, inputJTensor2, inputJTensor3)

  val inputTensorList1 = util.Arrays.asList(inputJTensor1)
  val inputTensorList2 = util.Arrays.asList(inputJTensor2)
  val inputTensorList3 = util.Arrays.asList(inputJTensor3)
  val inputTensorList = util.Arrays.asList(inputTensorList1, inputTensorList2, inputTensorList3)

  val inputTableList1 = util.Arrays.asList(inputJTensor1, inputJTensor2, inputJTensor3)
  val inputTableList2 = util.Arrays.asList(inputJTensor3, inputJTensor2, inputJTensor1)
  val inputTableList = util.Arrays.asList(inputTableList1, inputTableList2)

  "transferTensorsToTensorOfBatch" should "work" in {
    val tensorOfBatch = transferTensorsToTensorOfBatch(inputTensorsArray)
    val data1 = inputTensorsArray.map(_.getData).flatten
    val data2 = tensorOfBatch.storage().array()
    data1 should be (data2)
    tensorOfBatch.size() should be (Array(3, 3, 5, 5))
  }

  "transferListOfActivityToActivityOfBatch" should "work" in {
    val tensorOfBatch = transferListOfActivityToActivityOfBatch(inputTensorList,
      inputTensorList.size()).toTensor[Float]
    val data1 = inputTensorsArray.map(_.getData).flatten
    val data2 = tensorOfBatch.storage().array()
    data1 should be (data2)
    tensorOfBatch.size() should be (Array(3, 3, 5, 5))

    val tableOfBatch = transferListOfActivityToActivityOfBatch(inputTableList,
      inputTableList.size()).toTable
    val tensors = tableOfBatch.toSeq[Tensor[Float]]
    tensors.length should be (3)
    val data3 = Array(inputJTensor1, inputJTensor3).map(_.getData).flatten
    val data4 = tensors(0).storage().array()
    data3 should be (data4)
    val data5 = Array(inputJTensor2, inputJTensor2).map(_.getData).flatten
    val data6 = tensors(1).storage().array()
    data5 should be (data6)
    val data7 = Array(inputJTensor3, inputJTensor1).map(_.getData).flatten
    val data8 = tensors(2).storage().array()
    data7 should be (data8)
  }

  "transferBatchTensorToJListOfJListOfJTensor" should "work" in {
    val tensorOfBatch = transferTensorsToTensorOfBatch(inputTensorsArray)
    val listofListOfJTensor: util.List[util.List[JTensor]] =
      transferBatchTensorToJListOfJListOfJTensor(tensorOfBatch, 3)
    listofListOfJTensor.size should be (3)
    listofListOfJTensor.get(0).get(0).getData should be (inputJTensor1.getData)
    listofListOfJTensor.get(1).get(0).getData should be (inputJTensor2.getData)
    listofListOfJTensor.get(2).get(0).getData should be (inputJTensor3.getData)
    listofListOfJTensor.get(0).get(0).getShape should be (inputJTensor1.getShape)
    listofListOfJTensor.get(1).get(0).getShape should be (inputJTensor2.getShape)
    listofListOfJTensor.get(2).get(0).getShape should be (inputJTensor3.getShape)
  }

  "transferBatchTableToJListOfJListOfJTensor" should "work" in {
    val tableOfBatch = transferListOfActivityToActivityOfBatch(inputTableList,
      inputTableList.size()).toTable
    val listofListOfJTensor: util.List[util.List[JTensor]] =
      transferBatchTableToJListOfJListOfJTensor(tableOfBatch, 2)
    listofListOfJTensor.size should be (2)
    listofListOfJTensor.get(0).get(0).getData should be (inputJTensor1.getData)
    listofListOfJTensor.get(0).get(1).getData should be (inputJTensor2.getData)
    listofListOfJTensor.get(0).get(2).getData should be (inputJTensor3.getData)
    listofListOfJTensor.get(1).get(0).getData should be (inputJTensor3.getData)
    listofListOfJTensor.get(1).get(1).getData should be (inputJTensor2.getData)
    listofListOfJTensor.get(1).get(2).getData should be (inputJTensor1.getData)
    listofListOfJTensor.get(0).get(0).getShape should be (inputJTensor1.getShape)
    listofListOfJTensor.get(0).get(1).getShape should be (inputJTensor2.getShape)
    listofListOfJTensor.get(0).get(2).getShape should be (inputJTensor3.getShape)
    listofListOfJTensor.get(1).get(0).getShape should be (inputJTensor3.getShape)
    listofListOfJTensor.get(1).get(1).getShape should be (inputJTensor2.getShape)
    listofListOfJTensor.get(1).get(2).getShape should be (inputJTensor1.getShape)
  }
}
