package com.intel.analytics.zoo.orca.python

import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}
import java.util.{List => JList}

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.common.PythonZoo

import scala.reflect.ClassTag

object PythonOrca {

  def ofFloat(): PythonOrca[Float] = new PythonOrca[Float]()

  def ofDouble(): PythonOrca[Double] = new PythonOrca[Double]()
}

class PythonOrca[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T] {
  def inferenceModelDistriPredict(model: InferenceModel, sc: JavaSparkContext,
                                  inputs: JavaRDD[JList[com.intel.analytics.bigdl.python.api
                                  .JTensor]],
                                  inputIsTable: Boolean): JavaRDD[JList[Object]] = {
    val broadcastModel = sc.broadcast(model)
    inputs.rdd.mapPartitions(partition => {
      val localModel = broadcastModel.value
      partition.map(inputs => {
        val inputActivity = jTensorsToActivity(inputs, inputIsTable)
        val outputActivity = localModel.doPredict(inputActivity)
        activityToList(outputActivity)
      })
    })
  }
}
