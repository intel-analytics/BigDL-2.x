package com.intel.analytics.zoo.serving.http

import com.fasterxml.jackson.core.JsonParser
import com.fasterxml.jackson.databind.module.SimpleModule
import com.fasterxml.jackson.databind.node.{ArrayNode, IntNode, ObjectNode, TextNode}
import com.fasterxml.jackson.databind.{DeserializationContext, JsonDeserializer, JsonNode, ObjectMapper}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

class ServingFrontendSerializer extends JsonDeserializer[Activity]{
  var intBuffer: ArrayBuffer[Int] = null
  var floatBuffer: ArrayBuffer[Float] = null
  var stringBuffer: ArrayBuffer[String] = null
  var shapeBuffer: ArrayBuffer[Int] = null
  var valueCount: Int = 0
  var shapeMask: Map[Int, Boolean] = null
  override def deserialize(p: JsonParser, ctxt: DeserializationContext): Activity = {
    val oc = p.getCodec
    val node = oc.readTree[JsonNode](p)
    val inputsIt = node.get("instances").get(0).elements()
    val tensorBuffer = new ArrayBuffer[Tensor[Float]]()
    while (inputsIt.hasNext) {
      initBuffer()
      parse(inputsIt.next(), 0)
      if (shapeBuffer.isEmpty) shapeBuffer.append(1)
      if (!floatBuffer.isEmpty) {
        tensorBuffer.append(Tensor[Float](floatBuffer.toArray, shapeBuffer.toArray))
      } else {
        // add string, string tensor, sparse tensor in the future
        throw new Error("???")
      }

    }
    T.array(tensorBuffer.toArray)
  }
  def parse(node: JsonNode, currentShapeDim: Int): Unit = {
    if (node.isInstanceOf[ArrayNode]) {

      val iter = node.elements()
      if (shapeMask.get(currentShapeDim) == None) {
        shapeBuffer.append(node.size())
        shapeMask += (currentShapeDim -> true)
      }
      while (iter.hasNext) {
        parse(iter.next(), currentShapeDim + 1)
      }
    } else if (node.isInstanceOf[TextNode]) {
      stringBuffer.append(node.asText())
    } else if (node.isInstanceOf[ObjectNode]) {
      // currently used for SparseTensor only maybe
    } else {
      // v1: int, float, double would all parse to float
      floatBuffer.append(node.asDouble().toFloat)
    }

  }
  def initBuffer(): Unit = {
    floatBuffer = new ArrayBuffer[Float]()
    shapeBuffer = new ArrayBuffer[Int]()
    stringBuffer = new ArrayBuffer[String]()
    shapeMask = Map[Int, Boolean]()
  }

}
object ServingFrontendSerializer {
  def deserialize(str: String): Activity = {
    val mapper = new ObjectMapper()
    val module = new SimpleModule()
    module.addDeserializer(classOf[Activity], new ServingFrontendSerializer())
    mapper.registerModule(module)
    mapper.readValue(str, classOf[Activity])
  }
}
