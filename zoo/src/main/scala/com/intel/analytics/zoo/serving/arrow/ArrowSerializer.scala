package com.intel.analytics.zoo.serving.arrow

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}
import java.util

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.serving.utils.Conventions
import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.complex.{BaseRepeatedValueVector, ListVector}
import org.apache.arrow.vector.ipc.{ArrowStreamReader, ArrowStreamWriter}
import org.apache.arrow.vector.types.pojo.{ArrowType, Field, FieldType, Schema}
import org.apache.arrow.vector.{FieldVector, Float4Vector, IntVector, VectorSchemaRoot}

import scala.collection.JavaConverters._

class ArrowSerializer(data: Array[Float], shape: Array[Int]) {
  val allocator = new RootAllocator(Int.MaxValue)
//  def getFields(): List[Field] = {
//    List(dataVector.getField, shapeVector.getField)
//  }
//  def getVectors(): List[FieldVector] = {
//    List(dataVector, shapeVector)
//  }
  def copyDataToVector(vector: Float4Vector): Unit = {
    vector.allocateNew(data.size)
    (0 until data.size).foreach(i => vector.set(i, data(i)))
    vector.setValueCount(data.size)
  }
  def copyShapeToVector(vector: IntVector): Unit = {
    vector.allocateNew(shape.size)
    (0 until shape.size).foreach(i => vector.set(i, shape(i)))
    vector.setValueCount(shape.size)
  }

  /**
   * Create a vector of data and shape of type ListVector
   * @return
   */
  def createVector(): VectorSchemaRoot = {
    val vectorSchemaRoot = VectorSchemaRoot.create(
      ArrowSerializer.getSchema, new RootAllocator(Integer.MAX_VALUE))
    // copy data into data ListVector
    val dataList = vectorSchemaRoot.getVector("data").asInstanceOf[ListVector]
    val dataVector = dataList.getDataVector.asInstanceOf[Float4Vector]
    dataList.startNewValue(0)
    copyDataToVector(dataVector)
    dataList.endValue(0, data.size)
    dataList.setValueCount(1)
    // copy shape into shape ListVector
    val shapeList = vectorSchemaRoot.getVector("shape").asInstanceOf[ListVector]
    val shapeVector = shapeList.getDataVector.asInstanceOf[IntVector]
    shapeList.startNewValue(0)
    copyShapeToVector(shapeVector)
    shapeList.endValue(0, shape.size)
    shapeList.setValueCount(1)

    vectorSchemaRoot
  }
  def createRawVector(): VectorSchemaRoot = {
    val allocator = new RootAllocator(Integer.MAX_VALUE)
    val dataVector = new Float4Vector("data", allocator)
    val shapeVector = new IntVector("shape", allocator)
    copyDataToVector(dataVector)
    copyShapeToVector(shapeVector)
    val fields = List(dataVector.getField, shapeVector.getField)
    val vectors = List[FieldVector](dataVector, shapeVector)
    new VectorSchemaRoot(fields.asJava, vectors.asJava)
  }
  def create(): Array[Byte] = {
    val vectorSchemaRoot = createRawVector()
    val out = new ByteArrayOutputStream()
    val writer = new ArrowStreamWriter(vectorSchemaRoot, null, out)
    writer.start()
    writer.writeBatch()
    writer.end()
//    val byteArray = out.toByteArray
//    writer.close()
//    out.close()
    out.toByteArray
  }
}
object ArrowSerializer {
  def getSchema: Schema = {
    val dataField = new Field("data",
      FieldType.nullable(new ArrowType.List()), List(
        new Field("", FieldType.nullable(Conventions.ARROW_FLOAT), null)).asJava)
    val shapeField = new Field("shape",
      FieldType.nullable(new ArrowType.List()), List(
        new Field("", FieldType.nullable(Conventions.ARROW_INT), null)).asJava)
    new Schema(List(dataField, shapeField).asJava, null)
  }
  def apply(tensor: Tensor[Float]): Unit = {

  }
}
