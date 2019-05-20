/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.mkl._
import com.intel.analytics.bigdl.tensor.DnnStorage

sealed trait MemoryData extends Serializable {
  def shape: Array[Int]
  def layout: Int
  def dataType: Int
  def setShape(shape: Array[Int]): Unit
  def setLayout(layout: Int): Unit
  def setDataType(dataType: Int): Unit

  private var _mask: Int = -1
  private var _scales: Array[Float] = Array.emptyFloatArray

  def mask: Int = _mask
  def setMask(s: Int): Unit = _mask = s
  def scales: Array[Float] = _scales
  def setScales(f: Array[Float]): Unit = _scales = f

  def isLayoutFixed(): Boolean = {
    layout != Memory.Format.format_undef && layout != Memory.Format.any
  }

  def cloneFormat(): MemoryData

  private val UNDEFINED: Long = -1
  private val ERROR: Long = 0

  @transient private var primitive: Long = UNDEFINED
  @transient private var primitiveDesc: Long = UNDEFINED
  @transient private var description: Long = UNDEFINED

  def getMemoryDescription(): Long = {
    if (description == UNDEFINED || description == ERROR) {
      description = MklDnn.MemoryDescInit(shape.length, shape, dataType, layout)
    }
    description
  }

  def getPrimitiveDescription(runtime: MklDnnRuntime): Long = {
    require(runtime != null, s"Have you initialized the MklDnnRuntime?")
    if (primitiveDesc == UNDEFINED || primitiveDesc == ERROR) {
      primitiveDesc =
        MklDnn.MemoryPrimitiveDescCreate(getMemoryDescription(), runtime.engine)
    }
    primitiveDesc
  }

  def getPrimitive(runtime: MklDnnRuntime): Long = {
    require(runtime != null, s"Have you initialized the MklDnnRuntime?")
    if (primitive == UNDEFINED || primitive == ERROR) {
      primitive =
        MklDnn.PrimitiveCreate0(getPrimitiveDescription(runtime))
    }
    primitive
  }

  def setPrimitiveDescription(desc: Long): Unit = {
    primitiveDesc = desc
  }

  def setMemoryDescription(desc: Long): Unit = {
    description = desc
  }

  def getRealSize: Long = {
    require(primitiveDesc != UNDEFINED && primitiveDesc != ERROR)
    MklDnn.PrimitiveDescGetSize(primitiveDesc) / getDataTypeBytes
  }

  def getPaddingShape: Array[Int] = {
    require(description != UNDEFINED && description != ERROR)
    Memory.GetPaddingShape(description)
  }

  private def getDataTypeBytes: Int = {
    dataType match {
      case DataType.F32 => DnnStorage.FLOAT_BYTES
      case DataType.S32 => DnnStorage.INT_BYTES
      case DataType.S8 => DnnStorage.INT8_BYTES
      case DataType.U8 => DnnStorage.INT8_BYTES
      case _ => throw new UnsupportedOperationException(s"unsupported data type")
    }
  }
}

case class HeapData(private var _shape: Array[Int], private var _layout: Int,
  private var _dataType: Int = DataType.F32) extends MemoryData {

  override def dataType: Int = _dataType

  override def setDataType(dataType: Int): Unit = _dataType = dataType

  override def setShape(shape: Array[Int]): Unit = _shape = shape.clone()

  override def setLayout(layout: Int): Unit = _layout = layout

  override def shape: Array[Int] = _shape.clone()

  override def layout: Int = _layout

  override def hashCode(): Int = {
    val seed = 37
    var hash = 1
    hash = hash * seed + this.layout
    var d = 0
    while (d < this.shape.length) {
      hash = hash * seed + this.shape(d)
      d += 1
    }

    hash = hash * seed + this.dataType

    hash
  }

  override def equals(obj: Any): Boolean = {
    if (obj == null) {
      return false
    }
    if (!obj.isInstanceOf[HeapData]) {
      return false
    }
    val other = obj.asInstanceOf[HeapData]
    if (this.eq(other)) {
      return true
    }
    if (this.layout != other.layout) {
      return false
    }
    if (this.shape == null && other.shape == null) {
      return true
    }
    if (this.shape != null && other.shape != null) {
      if (this.shape.length != other.shape.length) return false
      var i = 0
      while(i < this.shape.length) {
        if (this.shape(i) != other.shape(i)) return false
        i += 1
      }
      return true
    } else {
      return false
    }
  }

  override def toString: String = {
    s"HeapData([${shape.mkString("x")}], ${layout})"
  }

  override def cloneFormat(): MemoryData = new HeapData(_shape, _layout, _dataType)

  def toNative(): NativeData = {
    NativeData(shape, layout)
  }
}

case class NativeData(private var _shape: Array[Int], private var _layout: Int,
  private var _dataType: Int = DataType.F32) extends MemoryData {

  override def shape: Array[Int] = _shape.clone()

  override def layout: Int = _layout

  override def setShape(shape: Array[Int]): Unit = _shape = shape.clone()

  override def setLayout(layout: Int): Unit = _layout = layout

  override def hashCode(): Int = {
    val seed = 41
    var hash = 1
    hash = hash * seed + this.layout
    var d = 0
    while (d < this.shape.length) {
      hash = hash * seed + this.shape(d)
      d += 1
    }

    hash = hash * seed + this.dataType

    hash
  }

  override def equals(obj: Any): Boolean = {
    if (obj == null) {
      return false
    }
    if (!obj.isInstanceOf[NativeData]) {
      return false
    }
    val other = obj.asInstanceOf[NativeData]
    if (this.eq(other)) {
      return true
    }
    if (this.layout != other.layout) {
      return false
    }
    if (this.shape == null && other.shape == null) {
      return true
    }
    if (this.shape != null && other.shape != null) {
      if (this.shape.length != other.shape.length) return false
      var i = 0
      while(i < this.shape.length) {
        if (this.shape(i) != other.shape(i)) return false
        i += 1
      }
      return true
    } else {
      return false
    }
  }

  override def toString: String = {
    s"NativeData([${shape.mkString("x")}], ${layout}, ${dataType}, ${mask}, ${scales})"
  }

  override def cloneFormat(): MemoryData = new NativeData(_shape, _layout, _dataType)

  override def dataType: Int = _dataType

  override def setDataType(dataType: Int): Unit = _dataType = dataType
}

private[mkldnn] object MemoryData {
  def noUndef(formats: Array[MemoryData]): Boolean = {
    if (formats == null || formats.length == 0) return true
    formats.foreach(f => if (f.layout == Memory.Format.format_undef) return false)
    return true
  }

  def isSizeCompatible(actual: MemoryData, expect: MemoryData): Boolean = {
    if (expect == null) return true
    if (actual == null) return false
    if (actual.shape.length != expect.shape.length) return false
    actual.shape.zip(expect.shape).foreach {case (a, e) => if (a != e) return false}
    return true
  }

  def primitiveOutput(pd: Long): NativeData = {
    val outputPD = MklDnn.PrimitiveDescQueryPd(pd, Query.DstPd, 0)
    val memoryDesc = MklDnn.PrimitiveDescQueryMemory(outputPD)
    val shape = Memory.GetShape(memoryDesc)
    val paddingShape = Memory.GetPaddingShape(memoryDesc)
    val layout = Memory.GetLayout(memoryDesc)
    val dataType = Memory.GetDataType(memoryDesc)
    val size = MklDnn.PrimitiveDescGetSize(outputPD)

    val memory = NativeData(shape, layout, dataType)
    memory.setMemoryDescription(memoryDesc)
    memory.setPrimitiveDescription(outputPD)
    memory
  }

  def operationWant(primDesc: Long, queryType: Int): NativeData = {
    val memoryPrimDesc = MklDnn.PrimitiveDescQueryPd(primDesc, queryType, 0)
    val memoryDesc = MklDnn.PrimitiveDescQueryMemory(memoryPrimDesc)
    val shape = Memory.GetShape(memoryDesc)
    val paddingShape = Memory.GetPaddingShape(memoryDesc)
    val layout = Memory.GetLayout(memoryDesc)
    val dataType = Memory.GetDataType(memoryDesc)

    val memory = NativeData(shape, layout, dataType)
    memory.setMemoryDescription(memoryDesc)
    memory.setPrimitiveDescription(memoryPrimDesc)
    memory
  }
}
