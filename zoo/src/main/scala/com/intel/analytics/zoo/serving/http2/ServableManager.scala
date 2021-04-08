package com.intel.analytics.zoo.serving.http2

import scala.collection.mutable


class ServableManager {
  private var modelVersionMap = new mutable.HashMap[String, mutable.HashMap[String, Servable]]

  def load(dir : String) = {
    val instancesYaml = scala.io.Source.fromFile(dir).mkString
    val modelInfoList = YamlUtil.fromYaml(classOf[Models], instancesYaml).models
    for (modelInfo <- modelInfoList){

    }
  }

  def retriveModels(modelName : String): List[Servable] = {
    List()
  }

  def retriveModel(modelName : String, modelVersion : String): Servable = {
    null
  }
}

abstract class Servable (modelMetaData: ModelMetaData){
  protected def predict(inputData : List [String]) : List[String]
}

abstract class ModelMetaData(modelName: String, modelVersion: String){

}

class InfluenceModelServable (modelMetaData: ModelMetaData) extends Servable (modelMetaData){
  protected def predict(inputData : List [String]) : List[String] = {
    null
  }
}

class InfluenceModelMetaData(modelName: String, modelVersion: String, modelPath: String,
                             weightPath: String, modelType: String) extends ModelMetaData(modelName, modelVersion)
