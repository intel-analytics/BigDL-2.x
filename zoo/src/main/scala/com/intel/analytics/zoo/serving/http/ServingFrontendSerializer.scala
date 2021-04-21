package com.intel.analytics.zoo.serving.http

import com.fasterxml.jackson.core.JsonParser
import com.fasterxml.jackson.databind.{DeserializationContext, JsonDeserializer, JsonNode}

import scala.collection.mutable

class ServingFrontendSerializer extends JsonDeserializer[Instances]{
  override def deserialize(p: JsonParser, ctxt: DeserializationContext): Instances = {
    val a = mutable.LinkedHashMap[String, Any]()
    var l: List[mutable.LinkedHashMap[String, Any]] = null
    val oc = p.getCodec
    val node = oc.readTree[JsonNode](p)
    node
    new Instances(l)
  }

}
