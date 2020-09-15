package com.intel.analytics.zoo.serving.arrow

import java.io.ByteArrayInputStream
import java.util.Base64

import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.{Float4Vector, IntVector, VectorSchemaRoot}
import org.apache.arrow.vector.holders.NullableIntHolder
import org.apache.arrow.vector.ipc.ArrowStreamReader

class ArrowDeserializer {
  def getFromSchemaRoot(vectorSchemaRoot: VectorSchemaRoot): (Array[Float], Array[Int]) = {
    val dataVector = vectorSchemaRoot.getVector("data").asInstanceOf[Float4Vector]
    val shapeVector = vectorSchemaRoot.getVector("shape").asInstanceOf[IntVector]
    val dataArray = new Array[Float](dataVector.getValueCount)
    (0 until dataArray.size).foreach(i => dataArray(i) = dataVector.get(i))
    var shapeArray = Array[Int]()
    val nullableHolder = new NullableIntHolder()
    (0 until dataArray.size).foreach(i => {
      shapeVector.get(i, nullableHolder)
      if (nullableHolder.isSet == 1) {
        shapeArray = shapeArray :+ nullableHolder.value
      }
    })
    (dataArray, shapeArray)
  }
  def create(b64string: String): Array[(Array[Float], Array[Int])] = {
    var result = Array[(Array[Float], Array[Int])]()
    val readAllocator = new RootAllocator(Int.MaxValue)
    val byteArr = Base64.getDecoder.decode(b64string)
    val reader = new ArrowStreamReader(new ByteArrayInputStream(byteArr), readAllocator)
    val schema = reader.getVectorSchemaRoot.getSchema
    while (reader.loadNextBatch()) {
      val vectorSchemaRoot = reader.getVectorSchemaRoot
      result = result :+ getFromSchemaRoot(vectorSchemaRoot)
    }
    result
  }
}
object ArrowDeserializer {
  def apply(b64string: String): Array[(Array[Float], Array[Int])] = {
    val deserializer = new ArrowDeserializer()
    deserializer.create(b64string)
  }
}
