package com.intel.analytics.zoo.ppml.vfl

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.{CAddTable, Concat, JoinTable, Sequential}
import com.intel.analytics.bigdl.optim.{OptimMethod, ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.{Criterion, Module}
import com.intel.analytics.zoo.ppml.DLAggregator
import com.intel.analytics.zoo.ppml.Util._
import com.intel.analytics.zoo.ppml.generated.FLProto._
import org.apache.log4j.Logger
import com.intel.analytics.zoo.ppml.Aggregator._

import collection.JavaConverters._

class VflNNAggregator(classifier: Module[Float],
                    optimMethod: OptimMethod[Float],
                    criterion: Criterion[Float],
                    validationMethods: Array[ValidationMethod[Float]]) extends DLAggregator{
  import VflAggregator.logger
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



  override def aggEvaluate(agg: Boolean): Unit = {
    val evaluatedResults: scala.collection.mutable.Map[String, ValidationResult] =
      scala.collection.mutable.Map()
    val inputTable = getInputTableFromStorage(EVAL)

      //      val name = outputs.keys.seq.head
      val classifierOutput = module.forward(inputTable)
      validationMethods.foreach(validation => {
        val vName = validation.toString()
        evaluatedResults(vName) = if (evaluatedResults.contains(vName)) {
          evaluatedResults(vName) + validation(classifierOutput, target)
        } else {
          validation(classifierOutput, target)
        }
      })

      if (agg) {
        val previousVersion = evalStorage.version
        val evalVersion = previousVersion + 1
        val metaData = TableMetaData.newBuilder()
          .setName("evaluateResult")
          .setVersion(evalVersion)
          .build()
        val aggResult = Table.newBuilder()
          .setMetaData(metaData)
        evaluatedResults.foreach{er =>
          val erTensor = Tensor[Float](T(er._2.result()._1, er._2.result()._2.toFloat))
          aggResult.putTable(er._1, toFloatTensor(erTensor))
        }
        logger.info(s"$evalVersion run evaluation successfully: result is ${evaluatedResults.mkString(" ")}")
        evaluatedResults.clear()
        evalStorage.save(evalVersion, aggResult.build())
      }
    } else {
      throw new RuntimeException("unimplemented multiOutputs")
    }
  }

  override def aggPredict(): Unit = {
    throw new IllegalArgumentException("Unimplemented method.")
  }

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