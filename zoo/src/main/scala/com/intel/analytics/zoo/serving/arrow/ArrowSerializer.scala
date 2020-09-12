package com.intel.analytics.zoo.serving.arrow

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}
import java.util

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.serving.utils.{Conventions, TensorUtils}
import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.complex.{BaseRepeatedValueVector, ListVector}
import org.apache.arrow.vector.ipc.{ArrowStreamReader, ArrowStreamWriter}
import org.apache.arrow.vector.types.pojo.{ArrowType, Field, FieldType, Schema}
import org.apache.arrow.vector.{FieldVector, Float4Vector, IntVector, VectorSchemaRoot}

import scala.collection.JavaConverters._

class ArrowSerializer(data: Array[Float], shape: Array[Int]) {


  def copyDataToVector(vector: Float4Vector): Unit = {
    vector.allocateNew(data.size)
    (0 until data.size).foreach(i => vector.set(i, data(i)))
    vector.setValueCount(data.size)
  }
  def copyShapeToVector(vector: IntVector): Unit = {
    vector.allocateNew(shape.size)
    (0 until shape.size).foreach(i => vector.set(i, shape(i)))
    vector.setValueCount(data.size)
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
    dataList.setValueCount(5)
    // copy shape into shape ListVector
    val shapeList = vectorSchemaRoot.getVector("shape").asInstanceOf[ListVector]
    val shapeVector = shapeList.getDataVector.asInstanceOf[IntVector]
    shapeList.startNewValue(0)
    copyShapeToVector(shapeVector)
    shapeList.endValue(0, shape.size)
    shapeList.setValueCount(1)

    vectorSchemaRoot
  }
  def copyToSchemaRoot(vectorSchemaRoot: VectorSchemaRoot): Unit = {
    vectorSchemaRoot.setRowCount(data.size)
    val dataVector = vectorSchemaRoot.getVector("data").asInstanceOf[Float4Vector]
    val shapeVector = vectorSchemaRoot.getVector("shape").asInstanceOf[IntVector]
    copyDataToVector(dataVector)
    copyShapeToVector(shapeVector)
  }


}
object ArrowSerializer {
  def getSchema: Schema = {
    val dataField = new Field("data", FieldType.nullable(Conventions.ARROW_FLOAT), null)
    val shapeField = new Field("shape", FieldType.nullable(Conventions.ARROW_INT), null)
    new Schema(List(dataField, shapeField).asJava, null)
  }
  def writeTensor(tensor: Tensor[Float],
                  vectorSchemaRoot: VectorSchemaRoot,
                  writer: ArrowStreamWriter): Unit = {
    val shape = tensor.size()
    val totalSize = TensorUtils.getTotalSize(tensor)
    val data = tensor.resize(totalSize).toArray()
    val serializer = new ArrowSerializer(data, shape)
    serializer.copyToSchemaRoot(vectorSchemaRoot)
    writer.writeBatch()
  }
  def apply(t: Activity, idx: Int): Array[Byte] = {
    val allocator = new RootAllocator(Int.MaxValue)
    val out = new ByteArrayOutputStream()
    val vectorSchemaRoot = VectorSchemaRoot.create(getSchema, allocator)
    val writer = new ArrowStreamWriter(vectorSchemaRoot, null, out)
    writer.start()
    if (t.isTable) {
      t.toTable.keySet.foreach(key => {
        val tensor = t.toTable(key).asInstanceOf[Tensor[Float]].select(1, idx)
        writeTensor(tensor, vectorSchemaRoot, writer)
      })

    } else if (t.isTensor) {
      val tensor = t.toTensor[Float].select(1, idx)
      writeTensor(tensor, vectorSchemaRoot, writer)
    } else {
      throw new Error("Your input for Post-processing is invalid, " +
        "neither Table nor Tensor, please check.")
    }
    vectorSchemaRoot.close()
    writer.end()
    writer.close()
    out.flush()
    out.close()
    out.toByteArray
  }
}
