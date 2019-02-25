package com.intel.analytics.zoo.common

import java.io.File

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.optim.LocalPredictor
import com.intel.analytics.zoo.models.common.ZooModel
import org.apache.spark.ml.bundle.SparkBundleContext
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import ml.combust.bundle.BundleFile
import ml.combust.bundle.dsl.Bundle
import ml.combust.mleap.spark.SparkSupport._
import ml.combust.mleap.runtime.MleapSupport._
import ml.combust.mleap.runtime.frame.{DefaultLeapFrame, Transformer}
import resource.managed

trait MleapFrame2Sample {

  def toSample(frame: DefaultLeapFrame):Array[Sample[Float]]

}

trait MleapSavePredict {

  def saveModels[T](df: DataFrame, mlibPipeline: Pipeline, model: Module[T], savePath: String) = {

    val mleapModelPath = s"jar:file:$savePath/mleapmodel/spark-pipeline.zip"

    new File(mleapModelPath).delete()

    val bigDLModelPath = s"$savePath/bigdlModel/model"

    val pipeline: PipelineModel = mlibPipeline.fit(df)
    // then serialize pipeline
    val sbc = SparkBundleContext().withDataset(pipeline.transform(df))
    for (bf <- managed(BundleFile(mleapModelPath))) {
      pipeline.writeBundle.save(bf)(sbc).get
    }

    model.saveModule(bigDLModelPath, null, true)

    println("save model finished:" + mleapModelPath +":" + bigDLModelPath)

  }

  def loadNpredict[T](frame: DefaultLeapFrame, savePath: String, featureToSample: MleapFrame2Sample) = {


    val mleapModelPath = s"jar:file:$savePath/mleapmodel/spark-pipeline.zip"
    val bigDLModelPath = s"$savePath/bigdlModel/model"

    val bfiles = managed(BundleFile(mleapModelPath))

    val bundle: Bundle[Transformer] = (for (bundleFile <- bfiles) yield {
      bundleFile.loadMleapBundle().get
    }).opt.get

    val mleapPipeline = bundle.root
    val predictFrame = mleapPipeline.transform(frame).get

    val featureSample = featureToSample.toSample(predictFrame)

    val bigdlModel = LocalPredictor(Module.loadModule[Float](bigDLModelPath))

    bigdlModel.predict(featureSample)

  }
}
