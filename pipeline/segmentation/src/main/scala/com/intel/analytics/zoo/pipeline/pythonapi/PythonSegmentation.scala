package com.intel.analytics.zoo.pipeline.pythonapi

import java.util.{Map => JMap}

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.python.api.PythonBigDL
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.{BytesToMat, FeatureTransformer, MatToFloats}
import com.intel.analytics.zoo.pipeline.utils._
import org.apache.log4j.Logger

import scala.collection.JavaConverters._
import scala.reflect.ClassTag

object PythonSegmentation {

  def ofFloat(): PythonSegmentation[Float] = new PythonSegmentation[Float]()

  def ofDouble(): PythonSegmentation[Double] = new PythonSegmentation[Double]()

  val logger = Logger.getLogger(getClass)
}

class PythonSegmentation[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDL[T] {
  def createUnmodeDetection(): UnmodeDetection = {
    UnmodeDetection()
  }

  def createImageMeta(classNum: Int): ImageMeta = {
    ImageMeta(classNum)
  }

  def createVisualizer(labelMap: JMap[Int, String], thresh: Float = 0.3f,
    encoding: String): FeatureTransformer = {
    Visualizer(labelMap.asScala.toMap, thresh, encoding, Visualizer.visualized) ->
      BytesToMat(Visualizer.visualized) -> MatToFloats(shareBuffer = false)
  }

  def readCocoLabelMap(): JMap[Int, String] = {
    LabelReader.readCocoLabelMap().asJava
  }

  def shareMemory(model: Module[T]): Unit = {
    ModuleUtil.shareMemory(model.asInstanceOf[Module[Float]])
  }
}
