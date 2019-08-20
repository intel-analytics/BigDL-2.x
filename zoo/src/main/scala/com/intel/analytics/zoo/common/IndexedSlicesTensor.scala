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

package com.intel.analytics.bigdl.tensor

import breeze.linalg.{DenseMatrix, DenseVector}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import org.apache.spark.mllib.linalg.{Matrix, Vector}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
  * Tensor's sparse representation.
  *
  * To describe an IndexedSlicesTensor, we need indices, values, and shape:
  * Indices means non-zero elements' indices; values means the values of the non-zero elements;
  * Shape means the dense shape of this IndexedSlicesTensor.
  *
  * For example, an 2D 3x4 DenseTensor:
  *  1, 0, 0, 4
  *  0, 2, 0, 0
  *  0, 0, 3, 0
  *
  *  it's sparse representation should be
  *  indices(0) = Array(0, 0, 1, 2)
  *  indices(1) = Array(0, 3, 1, 2)
  *  values     = Array(1, 4, 2, 3)
  *  shape      = Array(3, 4)
  *
  * @param _indices non-zero elements' indices, should be zero-based and ascending.
  * @param _values values of the non-zero elements
  * @param _storageOffset storageOffset, both _values and _indices's storage offset.
  * @param _nElement number of non-zero elements
  * @param _shape dense shape
  * @param _indicesOffset indices' offset, Default is zeros, will vary in narrowed/selected tensor.
  *                       The true indices should be (_indices - _indicesOffset).
  * @param nDimension dimensions.
  * @tparam T should be Double or Float
  */
// indices is zero based.
class IndexedSlicesTensor[@specialized(Float, Double) T: ClassTag](
   var _indices : Array[Int],
   var _values : Array[Array[T]],
   var _shape : Array[Int],
   var nDimension: Int)
 (implicit ev: TensorNumeric[T]) extends Tensor[T] {

  override def dim(): Int = _shape.length

  nDimension = _shape.length

  override def setValue(d1: Int, value: T): IndexedSlicesTensor.this.type = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
    this
  }

  override def setValue(d1: Int, d2: Int, value: T): IndexedSlicesTensor.this.type = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
    this
  }

  override def setValue(d1: Int, d2: Int, d3: Int, value: T): IndexedSlicesTensor.this.type = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
    this
  }

  override def setValue(d1: Int, d2: Int, d3: Int, d4: Int, value: T): IndexedSlicesTensor.this.type = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
    this
  }

  override def setValue(
                         d1: Int, d2: Int,
                         d3: Int, d4: Int, d5: Int, value: T): IndexedSlicesTensor.this.type = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
    this
  }

  override def unfold(dim: Int, size: Int, step: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
    this
  }

  override def nElement(): Int = _values.map(_.length).reduceLeft(_ + _)

  override def size(): Array[Int] = {
    _shape.slice(0, this.nDimension)
  }

  override def size(dim: Int): Int = {
    _shape(dim - 1)
  }

  override def stride(): Array[Int] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def stride(dim: Int): Int = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def fill(v: T): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def zero(): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def randn(): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def randn(mean: Double, stdv: Double): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def rand(): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def rand(lowerBound: Double, upperBound: Double): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def bernoulli(p: Double): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def transpose(dim1: Int, dim2: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"SparseTennewIndicesOffsetsor: Unimplemented method")
  }

  override def t(): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def apply(index: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def apply(indexes: Array[Int]): T = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def valueAt(d1: Int): T = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def valueAt(d1: Int, d2: Int): T = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def valueAt(d1: Int, d2: Int, d3: Int): T = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def valueAt(d1: Int, d2: Int, d3: Int, d4: Int): T = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def valueAt(d1: Int, d2: Int, d3: Int, d4: Int, d5: Int): T = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def apply(t: Table): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def update(index: Int, value: T): Unit = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def update(index: Int, src: Tensor[T]): Unit = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def update(indexes: Array[Int], value: T): Unit = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def update(t: Table, value: T): Unit = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def update(t: Table, src: Tensor[T]): Unit = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def update(filter: (T) => Boolean, value: T): Unit = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def isContiguous(): Boolean = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def contiguous(): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def isSameSizeAs(other: Tensor[_]): Boolean = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def resizeAs(src: Tensor[_]): Tensor[T] = {
    require(src.isInstanceOf[IndexedSlicesTensor[T]],
      "Only support resize as IndexedSlicesTensor")
    val x = src.asInstanceOf[IndexedSlicesTensor[T]]
    val values = x._values.map(z => new Array[T](z.length))
    IndexedSlicesTensor(new Array[Int](x._indices.length),
      values, x._shape.clone())
  }

  override def resize(sizes: Array[Int], strides: Array[Int]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def resize(size1: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def resize(size1: Int, size2: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def resize(size1: Int, size2: Int, size3: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def resize(size1: Int, size2: Int, size3: Int, size4: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def resize(size1: Int, size2: Int, size3: Int, size4: Int, size5: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def select(dim: Int, index: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def storage(): Storage[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def storageOffset(): Int = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def set(other: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def set(
                    storage: Storage[T], storageOffset: Int,
                    sizes: Array[Int], strides: Array[Int]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def set(): Tensor[T] = {
    this._indices = Array()
    if (this._values != null) {
      this._values = this._values.map(x => Array[T]())
    }
    this._shape = Array()
    this
  }

  override def narrow(dim: Int, index: Int, size: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def numNonZeroByRow(): Array[Int] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def copy(other: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def apply1(func: (T) => T): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def map(other: Tensor[T], func: (T, T) => T): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def squeeze(): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def squeeze(dim: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def squeezeNewTensor(): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def view(sizes: Array[Int]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def repeatTensor(sizes: Array[Int]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def expandAs(template: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def expand(sizes: Array[Int]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def split(size: Int, dim: Int): Array[Tensor[T]] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def split(dim: Int): Array[Tensor[T]] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def toBreezeVector(): DenseVector[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def toMLlibVector(): Vector = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def toBreezeMatrix(): DenseMatrix[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def toMLlibMatrix(): Matrix = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def getType(): TensorDataType = {
    ev.getType()
  }

  override def diff(other: Tensor[T], count: Int, reverse: Boolean): Boolean = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def addSingletonDimension(t: Tensor[T], dim: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def reshape(sizes: Array[Int]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def save(path: String, overWrite: Boolean): IndexedSlicesTensor.this.type = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def getTensorNumeric(): TensorNumeric[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def resize(size: Array[Int], nElement: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }


  // scalastyle:off methodName
  override def +(s: T): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def +(t: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def -(s: T): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def -(t: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def unary_-(): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def /(s: T): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def /(t: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def *(s: T): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def *(t: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }
  // scalastyle:on methodName

  override def sum(): T = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def sum(dim: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def sum(x: Tensor[T], dim: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def mean(): T = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def mean(dim: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def max(): T = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def max(dim: Int): (Tensor[T], Tensor[T]) = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def max(values: Tensor[T], indices: Tensor[T], dim: Int): (Tensor[T], Tensor[T]) = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def min(): T = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def min(dim: Int): (Tensor[T], Tensor[T]) = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def min(values: Tensor[T], indices: Tensor[T], dim: Int): (Tensor[T], Tensor[T]) = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def scatter(dim: Int, index: Tensor[T], src: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def gather(dim: Int, index: Tensor[T], src: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def conv2(kernel: Tensor[T], vf: Char): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def xcorr2(kernel: Tensor[T], vf: Char): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def sqrt(): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def abs(): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def add(value: T, y: Tensor[T]): Tensor[T] = {
    require(y.isInstanceOf[IndexedSlicesTensor[T]], "only support add IndexedSlicesTensor")
    add(y.clone().mul(value))
    this
  }

  override def add(y: Tensor[T]): Tensor[T] = {
    require(y.isInstanceOf[IndexedSlicesTensor[T]], "only support add IndexedSlicesTensor")

    val iy = y.asInstanceOf[IndexedSlicesTensor[T]]

    val yIndices = iy._indices
    val yValues = iy._values

    if (isEmpty) {
      _indices = yIndices.clone()
      _values = yValues.map(z => z.clone())
      _shape = iy._shape.clone()
    } else {
      require(iy._shape.deep == this._shape.deep,
        "only support add IndexedSlicesTensor with same shape")

      val indices = new ArrayBuffer[Int]()
      val values = new ArrayBuffer[Array[T]]()

      var i = 0
      var j = 0
      while (i < _indices.length && j < yIndices.length) {
        if (_indices(i) < yIndices(j)) {
          indices.append(_indices(i))
          values.append(_values(i))
          i += 1
        } else if (_indices(i) > yIndices(j)) {
          indices.append(yIndices(j))
          values.append(yValues(j))
          j += 1
        } else {
          indices.append(yIndices(j))
          values.append(yValues(j).zip(_values(i)).map { case (x, y) => ev.plus(x, y) })
          i += 1
          j += 1
        }
      }

      if (i < _indices.length) {
        indices ++= _indices.slice(i, _indices.length)
        values ++= _values.slice(i, _values.length)
      } else if (j < yIndices.length) {
        indices ++= yIndices.slice(j, yIndices.length)
        values ++= yValues.slice(j, yValues.length)
      }
      this._indices = indices.toArray
      this._values = values.toArray
    }
    this
  }

  override def add(x: Tensor[T], value: T, y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def add(value: T): Tensor[T] = {
    require(!isEmpty, "Doesn't support add a constant to an empty IndexedSlicesTensor")
    _values.foreach(_.foreach(ev.plus(_, value)))
    this
  }

  override def add(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def dot(y: Tensor[T]): T = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def cmax(value: T): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def dist(y: Tensor[T], norm: Int): T = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def addcmul(value: T, tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def addcmul(tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def addcdiv(value: T, tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def sub(value: T, y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  // Puts the result of x - value * y in current tensor
  override def sub(x: Tensor[T], value: T, y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def sub(y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def sub(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def sub(value: T): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def cmul(y: Tensor[T]): Tensor[T] = {
    cmul(this, y)
  }

  override def cmul(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    require(x.isInstanceOf[IndexedSlicesTensor[_]],
      "Only support IndexedSlicesTensor in this operation")
    require(y.isInstanceOf[IndexedSlicesTensor[_]],
      "Only support IndexedSlicesTensor in this operation")

    val ix = x.asInstanceOf[IndexedSlicesTensor[T]]
    val iy = y.asInstanceOf[IndexedSlicesTensor[T]]
    require(iy._shape.deep == ix._shape.deep,
      "only support add IndexedSlicesTensor with same shape")

    val yIndices = iy._indices
    val yValues = iy._values

    val indices = new ArrayBuffer[Int]()
    val values = new ArrayBuffer[Array[T]]()

    var i = 0
    var j = 0
    while (i < _indices.length && j < yIndices.length) {
      if (_indices(i) < yIndices(j)) {
        i += 1
      } else if (_indices(i) < yIndices(j)) {
        j += 1
      } else {
        indices.append(yIndices(j))
        values.append(yValues(j).zip(_values(i)).map { case (x, y) => ev.times(x, y) })
        i += 1
        j += 1
      }
    }
    _indices = indices.toArray
    _values = values.toArray
    _shape = ix._shape
    this
  }

  override def cdiv(y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def div(y: Tensor[T]): Tensor[T] = {
    require(y.isInstanceOf[IndexedSlicesTensor[_]],
      "Only support IndexedSlicesTensor in this operation")

    val iy = y.asInstanceOf[IndexedSlicesTensor[T]]
    require(iy._shape.deep == this._shape.deep,
      "only support add IndexedSlicesTensor with same shape")
    require(iy._indices.deep == this._indices.deep,
      "only support add IndexedSlicesTensor with same shape")

    var i = 0
    while (i < _values.length) {
      _values(i) = _values(i).zip(iy._values(i)).map { case (x, y) => ev.divide(x, y)}
      i += 1
    }
    this
  }

  override def cdiv(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def mul(value: T): Tensor[T] = {
    if (!isEmpty) {
      var i = 0
      var j = 0
      while (i < _values.length) {
        while (j < _values(i).length) {
          _values(i)(j) = ev.times(_values(i)(j), value)
          j += 1
        }
        i += 1
      }
    }
    this
  }

  override def div(value: T): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def mul(x: Tensor[T], value: T): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def addmm(v1: T, M: Tensor[T], v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def addmm(M: Tensor[T], mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def addmm(mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def addmm(v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def addmm(v1: T, v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def mm(mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def addr(t1: Tensor[T], t2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def addr(v1: T, t1: Tensor[T], t2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def addr(v1: T, t1: Tensor[T], v2: T, t2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def addr(v1: T, t1: Tensor[T], v2: T, t2: Tensor[T], t3: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def uniform(args: T*): T = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def addmv(
                      beta: T, vec1: Tensor[T], alpha: T,
                      mat: Tensor[T], vec2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def addmv(beta: T, alpha: T, mat: Tensor[T], vec2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def addmv(alpha: T, mat: Tensor[T], vec2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def mv(mat: Tensor[T], vec2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def baddbmm(
                        beta: T, M: Tensor[T],
                        alpha: T, batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def baddbmm(beta: T, alpha: T, batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def baddbmm(alpha: T, batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def bmm(batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def pow(y: Tensor[T], n: T): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def pow(n: T): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def square(): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def topk(
                     k: Int, dim: Int, increase: Boolean, result: Tensor[T],
                     indices: Tensor[T], sortedResult: Boolean = true): (Tensor[T], Tensor[T]) = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def log(y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def exp(y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def sqrt(y: Tensor[T]): Tensor[T] = {
    require(y.isInstanceOf[IndexedSlicesTensor[T]], "Only support IndexedSlicesTensor")
    val iy = y.asInstanceOf[IndexedSlicesTensor[T]]
    _indices = iy._indices.clone()
    _values = iy._values.map(_.map(ev.sqrt(_)))
    _shape = iy._shape.clone()
    this
  }

  override def log1p(y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }
  override def log(): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }
  override def exp(): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def log1p(): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def abs(x: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def norm(y: Tensor[T], value: Int, dim: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def gt(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def lt(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def le(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def eq(x: Tensor[T], y: T): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def maskedFill(mask: Tensor[T], e: T): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def maskedCopy(mask: Tensor[T], y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def maskedSelect(mask: Tensor[T], y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def norm(value: Int): T = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def sign(): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def ge(x: Tensor[T], value: Double): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def indexAdd(dim: Int, index: Tensor[T], y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def index(dim: Int, index: Tensor[T], y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def cmax(y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def cmax(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def cmin(y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def cmin(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def range(xmin: Double, xmax: Double, step: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def toTensor[D](implicit ev: TensorNumeric[D]): Tensor[D] = {
    if (ev.getType() == ev.getType()) {
      this.asInstanceOf[Tensor[D]]
    } else {
      throw new IllegalArgumentException(s"The type ${ev.getType().getClass}" +
        s" in toTensor[${ev.getType().getClass}] is not same" +
        s"as the numeric type ${ev.getType().getClass} of the " +
        "corresponding module, please keep them same.")
    }
  }

  override def equals(obj: Any): Boolean = {
    if (obj == null) {
      return false
    }
    if (!obj.isInstanceOf[IndexedSlicesTensor[T]]) {
      return false
    }
    val other = obj.asInstanceOf[IndexedSlicesTensor[T]]
    if (this.eq(other)) {
      return true
    }
    if (this.nDimension != other.nDimension) {
      return false
    }
    var d = 1
    while (d <= this.nDimension) {
      if (this.size(d) != other.size(d)) {
        return false
      }
      d += 1
    }

    _values.deep == other._values.deep &&
      _indices.deep == other._indices.deep &&
      this._shape.deep == other._shape.deep
  }

  override def hashCode(): Int = {
    val state = Seq(_indices, _values, _shape, nDimension)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }

  override def isEmpty: Boolean = {
    _indices.length == 0
  }

  override def isScalar: Boolean = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def value(): T = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def setValue(value: T): IndexedSlicesTensor.this.type = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def applyFun[A: ClassTag](
                                      t: Tensor[A],
                                      func: (A) => T): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def zipWith[A: ClassTag, B: ClassTag](
                                                  t1: Tensor[A],
                                                  t2: Tensor[B],
                                                  func: (A, B) => T): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def prod(): T = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def prod(x: Tensor[T], dim: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def tanh(): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def tanh(y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def forceFill(v: Any): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def emptyInstance(): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def cast[@specialized(Long, Int, Short, Double, Float) D: ClassTag]
  (castTensor: Tensor[D])
  (implicit ev1: TensorNumeric[D]): Tensor[D] = {
    castTensor.getType() match {
      case FloatType =>
        castTensor.applyFun[T](this.asInstanceOf[IndexedSlicesTensor[T]],
          x => ev.toType[Float](x).asInstanceOf[D])
      case DoubleType =>
        castTensor.applyFun[T](this.asInstanceOf[IndexedSlicesTensor[T]],
          x => ev.toType[Double](x).asInstanceOf[D])
      case LongType =>
        castTensor.applyFun[T](this.asInstanceOf[IndexedSlicesTensor[T]],
          x => ev.toType[Long](x).asInstanceOf[D])
      case IntType =>
        castTensor.applyFun[T](this.asInstanceOf[IndexedSlicesTensor[T]],
          x => ev.toType[Int](x).asInstanceOf[D])
      case ShortType =>
        castTensor.applyFun[T](this.asInstanceOf[IndexedSlicesTensor[T]],
          x => ev.toType[Short](x).asInstanceOf[D])
      case _ =>
        throw new RuntimeException("Unspported type")
    }
    castTensor
  }

  override def getTensorType: TensorType = SparseType

  override def floor(y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def floor(): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def ceil(): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def inv(): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def negative(x: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def reduce(dim: Int, result: Tensor[T], reducer: (T, T) => T): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def toArray(): Array[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def erf(): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def erfc(): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def logGamma(): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def digamma(): Tensor[T] = {
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")
  }

  override def clamp(minValue: Double, maxValue: Double): Tensor[T] =
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")

  override def sumSquare(): T =
    throw new UnsupportedOperationException(s"IndexedSlicesTensor: Unimplemented method")

  override private[bigdl] def toQuantizedTensor: QuantizedTensor[T] =
    throw new IllegalArgumentException("IndexedSlicesTensor cannot be cast to QuantizedTensor")

  override def clone(): Tensor[T] = {
    if (!isEmpty) {
      val values = this._values.map(z => z.clone())
      IndexedSlicesTensor(this._indices.clone(),
        values, this._shape.clone())
    } else IndexedSlicesTensor()
  }
}

object IndexedSlicesTensor {
  def apply[T: ClassTag](
    indices : Array[Int],
    values : Array[Array[T]],
    shape : Array[Int])(
    implicit ev: TensorNumeric[T]): IndexedSlicesTensor[T] = {
    new IndexedSlicesTensor(indices, values, shape, shape.length)
  }

  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): IndexedSlicesTensor[T] = {
    new IndexedSlicesTensor[T](Array[Int](), Array[Array[T]](), Array[Int](), 1)
  }
}
