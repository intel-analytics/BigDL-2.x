package com.intel.analytics.zoo.ppml.vfl

import com.intel.analytics.bigdl.nn.{CAddTable, Sequential}
import com.intel.analytics.bigdl.optim.{OptimMethod, ValidationMethod, ValidationResult}

import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.{Criterion, Module}
import com.intel.analytics.zoo.ppml.DLAggregator
import org.apache.log4j.Logger
import com.intel.analytics.zoo.ppml.common.Aggregator._
import com.intel.analytics.zoo.ppml.common.FLPhase._

class VflNNAggregator(classifier: Module[Float],
                      optimMethod: OptimMethod[Float],
                      criterion: Criterion[Float],
                      validationMethods: Array[ValidationMethod[Float]]) extends DLAggregator{
  module = Sequential[Float]().add(CAddTable[Float]())
  if (classifier != null) {
    module.add(classifier)
  }

  def setClientNum(clientNum: Int): this.type = {
    this.clientNum = clientNum
    this
  }


  override def aggregate(): Unit = {
    val inputTable = getInputTableFromStorage(TRAIN)

    val output = module.forward(inputTable)
    val loss = criterion.forward(output, target)
    val gradOutput = criterion.backward(output, target)
    val gradInput = module.backward(inputTable, gradOutput)

    // TODO: Multi gradinput
    postProcess(TRAIN, gradInput, T(loss))

  }
  // TODO: add evaluate and predict


}

object VflAggregator {
  val logger = Logger.getLogger(this.getClass)

  def apply(clientNum: Int,
            classifier: Module[Float],
            optimMethod: OptimMethod[Float],
            criterion: Criterion[Float]): VflNNAggregator = {
    new VflNNAggregator(classifier, optimMethod, criterion, null).setClientNum(clientNum)
  }

  def apply(clientNum: Int,
            classifier: Module[Float],
            optimMethod: OptimMethod[Float],
            criterion: Criterion[Float],
            validationMethods: Array[ValidationMethod[Float]]): VflNNAggregator = {
    new VflNNAggregator(classifier, optimMethod, criterion, validationMethods).setClientNum(clientNum)
  }
}