package com.intel.analytics.zoo.serving.arrow
import java.io.{ByteArrayInputStream, ByteArrayOutputStream, ObjectOutputStream}
import java.util.Base64
import java.nio.charset.StandardCharsets
import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.ipc.{ArrowStreamReader, ArrowStreamWriter}
import org.scalatest.{FlatSpec, Matchers}
import org.apache.arrow.vector.{BitVector, FieldVector, VarCharVector, VectorSchemaRoot}
import java.nio.charset.StandardCharsets

import scala.collection.JavaConverters._
import org.apache.arrow.vector.types.pojo.Field

class ArrowSerializerSpec extends FlatSpec with Matchers {
  "Arrow Serialization" should "work" in {
    val data = Array(1.2f,2,4,5,6)
    val shape = Array(1,2,3,4,5)
    val ser = new ArrowSerializer(data, shape)
    val byteArr = ser.create()

    val allocator = new RootAllocator(Int.MaxValue)
    val reader = new ArrowStreamReader(new ByteArrayInputStream(byteArr), allocator)
    val schema = reader.getVectorSchemaRoot.getSchema
    while (reader.loadNextBatch()) {
      val schemaRoot = reader.getVectorSchemaRoot
      schemaRoot
    }
    require(schema.getFields.size() == 2, "schame number wrong.")
  }
  "Arrow example" should "work" in {
    val allocator = new RootAllocator(Int.MaxValue)
    val bitVector = new BitVector("boolean", allocator)
    val varCharVector = new VarCharVector("varchar", allocator)
    for (i <- 0 until 10) {
      bitVector.setSafe(i, if (i % 2 == 0) 0
      else 1)
      varCharVector.setSafe(i, ("test" + i).getBytes(StandardCharsets.UTF_8))
    }
    bitVector.setValueCount(10)
    varCharVector.setValueCount(10)

    val fields = List(bitVector.getField, varCharVector.getField)
    val vectors = List[FieldVector](bitVector, varCharVector)
    val root = new VectorSchemaRoot(fields.asJava, vectors.asJava)
    val out = new ByteArrayOutputStream()
    val writer = new ArrowStreamWriter(root, null, out)
    writer.start()
    // write the first batch
    writer.writeBatch()
    val readAllocator = new RootAllocator(Int.MaxValue)
    val reader = new ArrowStreamReader(new ByteArrayInputStream(out.toByteArray), readAllocator)
    val schema = reader.getVectorSchemaRoot.getSchema
    while (reader.loadNextBatch()) {
      val schemaRoot = reader.getVectorSchemaRoot
      schemaRoot.getFieldVectors.asScala.foreach(v => {
        require(v.getValueCount == 10, "vector size wrong")
      })
    }
    require(schema.getFields.size() == 2, "schema number wrong.")
  }

}