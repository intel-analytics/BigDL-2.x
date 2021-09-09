package com.intel.analytics.zoo.ppml

import com.intel.analytics.bigdl.nn.Sequential
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.zoo.ppml.Util.toFloatTensor
import com.intel.analytics.zoo.ppml.generated.FLProto.TableMetaData
import com.intel.analytics.zoo.ppml.Aggregator._
import com.intel.analytics.zoo.ppml.generated.FLProto
import org.apache.log4j.Logger

import collection.JavaConverters._


trait DLAggregator extends Aggregator {
  val logger = Logger.getLogger(this.getClass)
  var module: Sequential[Float] = null
  var target: Tensor[Float] = null
  def getInputTableFromStorage(storageType: Int): Table = {
    val storage = getStorage(storageType)
    val aggData = storage.localData.asScala.mapValues(_.getTableMap).values
      .flatMap(_.asScala).groupBy(_._1)
      .map{data =>
        (data._1, data._2.map {v =>
          val data = v._2.getTensorList.asScala.toArray.map(_.toFloat)
          val shape = v._2.getShapeList.asScala.toArray.map(_.toInt)
          Tensor[Float](data, shape)
        })
      }
    target = Tensor[Float]()
    if (aggData.contains("target")) {
      val t = aggData("target").head
      target.resizeAs(t).copy(t)
    }
    // TODO: multiple input
    val outputs = aggData.filter(_._1 != "target")
    require(outputs.size == 1)

    T.seq(outputs.values.head.toSeq)
  }
  def postProcess(aggType: Int, grad: Activity = null, loss: Activity = null): Unit = {
    def updateStorage(storage: Storage, table: FLProto.Table): Unit = {
      storage.localData.clear()
      storage.serverData = table
      storage.version += 1
      logger.info(s"${trainStorage.version} run aggregate successfully: loss is ${loss}")
    }
    val storage = aggType
    val gradProto = if (grad != null) {
      toFloatTensor(grad.toTable.apply[Tensor[Float]](1))
    } else null
    val lossProto = if (loss != null) {
      toFloatTensor(Tensor[Float](loss.toTable))
    } else null
    val metaBuilder = TableMetaData.newBuilder()
    var aggregatedTable: FLProto.Table = null
    if (aggType == TRAIN) {
      val meta = metaBuilder.setName("gradInput").setVersion(trainStorage.version).build()
      aggregatedTable = FLProto.Table.newBuilder()
        .setMetaData(meta)
        .putTable("gradInput", gradProto)
        .putTable("loss", lossProto)
        .build()
    } else if (aggType == EVAL) {
      val meta = metaBuilder.setName("evaluateResult").setVersion(evalStorage.version).build()
      aggregatedTable = FLProto.Table.newBuilder()
        .setMetaData(meta)
        .build()
    } else if (aggType == PREDICT) {
      val meta = metaBuilder.setName("predictResult").setVersion(predictStorage.version).build()
    }
    updateStorage(aggTypeMap.get(aggType), aggregatedTable);
  }
}
