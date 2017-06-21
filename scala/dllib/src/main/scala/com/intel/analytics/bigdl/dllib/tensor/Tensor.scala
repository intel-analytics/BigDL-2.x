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

import java.io.Serializable

import breeze.linalg.{DenseMatrix => BrzDenseMatrix, DenseVector => BrzDenseVector}
import com.intel.analytics.bigdl.mkl.MKL
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{File, Table}
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector, Matrix, Vector}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * It is the class for handling numeric data.
 *
 * @tparam T should be Double or Float
 */
trait Tensor[T] extends Serializable with TensorMath[T] with Activity {
  /**
   * Dimension number of the tensor. For empty tensor, its dimension number is 0
   *
   * @return dimension number
   */
  def nDimension(): Int

  /**
   * A shortcut of nDimension()
   *
   * @see nDimension()
   */
  def dim(): Int

  /**
   * Size of tensor. Return an array of which each value represent the size on the
   * dimension(i + 1), i is the index of the corresponding value
   * It will generate a new array each time you invoke the method
   *
   * @return size array
   */
  def size(): Array[Int]

  /**
   * size of the tensor on the given dimension
   *
   * @param dim dimension, count from 1
   * @return size
   */
  def size(dim: Int): Int

  /**
   * Jumps between element on the each dimension in the storage.
   * It will generate a new array each time you invoke the method
   *
   * @return strides array
   */
  def stride(): Array[Int]

  /**
   * Jumps between element on the given dimension in the storage.
   *
   * @param dim dimension, count from 1
   * @return jump
   */
  def stride(dim: Int): Int

  /**
   * Fill with a given value. It will change the value of the current tensor and return itself
   *
   * @param v value to fill the tensor
   * @return current tensor
   */
  def fill(v: T): Tensor[T]

  /**
   * Fill with zero. It will change the value of the current tensor and return itself
   *
   * @return current tensor
   */
  def zero(): Tensor[T]

  /**
   * Fill with random value(normal gaussian distribution).
   * It will change the value of the current tensor and return itself
   *
   * @return current tensor
   */
  def randn(): Tensor[T]

  /**
   * Fill with random value(normal gaussian distribution with the specified mean
   * and stdv).
   * It will change the value of the current tensor and return itself
   *
   * @return current tensor
   */
  def randn(mean: Double, stdv: Double): Tensor[T]

  /**
   * Fill with random value(uniform distribution).
   * It will change the value of the current tensor and return itself
   *
   * @return current tensor
   */
  def rand(): Tensor[T]

  /**
   * Fill with random value(uniform distribution between [lowerBound, upperBound])
   * It will change the value of the current tensor and return itself
   *
   * @return current tensor
   */
  def rand(lowerBound: Double, upperBound: Double): Tensor[T]

  /**
   * Fill with random value(bernoulli distribution).
   * It will change the value of the current tensor and return itself
   *
   * @return current tensor
   */
  def bernoulli(p: Double): Tensor[T]

  /** *
   * Create a new tensor which exchanges the given dimensions of the current tensor
   *
   * @param dim1 dimension to be exchanged, count from one
   * @param dim2 dimension to be exchanged, count from one
   * @return new tensor
   */
  def transpose(dim1: Int, dim2: Int): Tensor[T]

  /**
   * Shortcut of transpose(1, 2) for 2D tensor
   *
   * @see transpose()
   */
  def t(): Tensor[T]

  /**
   * Query tensor on a given index. Tensor should not be empty
   *
   * @param index count from 1
   * @return
   */
  def apply(index: Int): Tensor[T]

  /**
   * Query the value on a given index. Tensor should not be empty
   *
   * @param indexes the indexes length should be same as the tensor dimension length and each
   *                value count from 1
   * @return the value on the given index
   */
  def apply(indexes: Array[Int]): T

  /**
   * Query the value on a given position. The number of parameters
   * should be equal to the dimension number of the tensor.
   * Tensor should not be empty.
   *
   * @param d1,( d2, d3, d4, d5) the given position
   * @return the value on a given position
   */

  def valueAt(d1: Int): T

  def valueAt(d1: Int, d2: Int): T

  def valueAt(d1: Int, d2: Int, d3: Int): T

  def valueAt(d1: Int, d2: Int, d3: Int, d4: Int): T

  def valueAt(d1: Int, d2: Int, d3: Int, d4: Int, d5: Int): T

  /**
   * Subset the tensor by apply the element of the given table to corresponding dimension of the
   * tensor. The element of the given table can be an Int or another Table.
   * An Int means select on current dimension; A table means narrow on current dimension,
   * the table should has two elements, of which the first is start index and
   * the second is the end index. An empty table is equals to Table(1, size_of_current_dimension)
   * If the table length is less than the tensor dimension, the missing dimension is applied by
   * an empty table
   *
   * @see select
   * @see narrow
   * @param t The table length should be less than or equal to the tensor dimensions
   * @return
   */
  def apply(t: Table): Tensor[T]

  /**
   * For tensor(i) = value. If tensor(i) is another tensor, it will fill the selected subset by
   * the given value
   *
   * @param index index
   * @param value value to write
   */
  def update(index: Int, value: T): Unit

  /**
   * Copy the give tensor value to the select subset of the current tensor by the given index.
   * The subset should
   * has the same size of the given tensor
   *
   * @param index index
   * @param src   tensor to write
   */
  def update(index: Int, src: Tensor[T]): Unit

  /**
   * Write the value to the value indexed by the given index array
   *
   * @param indexes index array. It should has same length with the tensor dimension
   * @param value   value to write
   */
  def update(indexes: Array[Int], value: T): Unit

  /**
   * Write the value on a given position. The number of parameters
   * should be equal to the dimension number of the tensor.
   *
   * @param d1,( d2, d3, d4, d5) the given position
   * @param value the written value
   * @return
   */
  def setValue(d1: Int, value: T): this.type

  def setValue(d1: Int, d2: Int, value: T): this.type

  def setValue(d1: Int, d2: Int, d3: Int, value: T): this.type

  def setValue(d1: Int, d2: Int, d3: Int, d4: Int, value: T): this.type

  def setValue(d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, value: T): this.type

  /**
   * Fill the select subset of the current tensor with the given value.
   * The element of the given table can be an Int or another Table. An Int means select on current
   * dimension; A tablemeans narrow on current dimension, the table should has two elements,
   * of which the first is start index and the second is the end index. An empty table is equals
   * to Table(1, size_of_current_dimension) If the table length is less than the tensor dimension,
   * the missing dimension is applied by an empty table
   *
   * @param t     subset table
   * @param value value to write
   */
  def update(t: Table, value: T): Unit

  /**
   * Copy the given tensor value to the select subset of the current tensor
   * The element of the given table can be an Int or another Table. An Int means select on current
   * dimension; A table means narrow on current dimension, the table should has two elements,
   * of which the first is start index and the second is the end index. An empty table is equals
   * to Table(1, size_of_current_dimension) If the table length is less than the tensor dimension,
   * the missing dimension is applied by an empty table
   *
   * @param t   subset table
   * @param src tensor to copy
   */
  def update(t: Table, src: Tensor[T]): Unit

  /**
   * Update the value meeting the filter criteria with the give value
   *
   * @param filter filter
   * @param value  value to update
   */
  def update(filter: T => Boolean, value: T): Unit

  /**
   * Check if the tensor is contiguous on the storage
   *
   * @return true if it's contiguous
   */
  def isContiguous(): Boolean

  /**
   * Get a contiguous tensor from current tensor
   *
   * @return the current tensor if it's contiguous; or a new contiguous tensor with separated
   *         storage
   */
  def contiguous(): Tensor[T]

  /**
   * Check if the size is same with the give tensor
   *
   * @param other tensor to be compared
   * @return true if they have same size
   */
  def isSameSizeAs(other: Tensor[_]): Boolean

  /**
   * Get a new tensor with same value and different storage
   *
   * @return new tensor
   */
  override def clone(): Tensor[T] = {
    this
  }

  /**
   * Resize the current tensor to the same size of the given tensor. It will still use the same
   * storage if the storage
   * is sufficient for the new size
   *
   * @param src target tensor
   * @return current tensor
   */
  def resizeAs(src: Tensor[_]): Tensor[T]

  /**
   * Resize the current tensor to the give shape
   *
   * @param sizes   Array describe the size
   * @param strides Array describe the jumps
   * @return
   */
  def resize(sizes: Array[Int], strides: Array[Int] = null): Tensor[T]

  def resize(size1: Int): Tensor[T]

  def resize(size1: Int, size2: Int): Tensor[T]

  def resize(size1: Int, size2: Int, size3: Int): Tensor[T]

  def resize(size1: Int, size2: Int, size3: Int, size4: Int): Tensor[T]

  def resize(size1: Int, size2: Int, size3: Int, size4: Int, size5: Int): Tensor[T]

  //  def repeatTensor(result: Tensor, tensor: Tensor, size: Int*)

  /**
   * Element number
   *
   * @return element number
   */
  def nElement(): Int

  /**
   * Remove the dim-th dimension and return the subset part. For instance
   * tensor =
   * 1 2 3
   * 4 5 6
   * tensor.select(1, 1) is [1 2 3]
   * tensor.select(1, 2) is [4 5 6]
   * tensor.select(2, 3) is [3 6]
   *
   * @param dim
   * @param index
   * @return
   */
  def select(dim: Int, index: Int): Tensor[T]

  /**
   * Get the storage
   *
   * @return storage
   */
  def storage(): Storage[T]

  /**
   * tensor offset on the storage
   *
   * @return storage offset, count from 1
   */
  def storageOffset(): Int

  /**
   * The Tensor is now going to "view" the same storage as the given tensor. As the result,
   * any modification in the elements of the Tensor will have an impact on the elements of the
   * given tensor, and vice-versa. This is an efficient method, as there is no memory copy!
   *
   * @param other the given tensor
   * @return current tensor
   */
  def set(other: Tensor[T]): Tensor[T]

  /**
   * The Tensor is now going to "view" the given storage, starting at position storageOffset (>=1)
   * with the given dimension sizes and the optional given strides. As the result, any
   * modification in the elements of the Storage will have an impact on the elements of the Tensor,
   * and vice-versa. This is an efficient method, as there is no memory copy!
   *
   * If only storage is provided, the whole storage will be viewed as a 1D Tensor.
   *
   * @param storage
   * @param storageOffset
   * @param sizes
   * @param strides
   * @return current tensor
   */
  def set(storage: Storage[T], storageOffset: Int = 1, sizes: Array[Int] = null,
          strides: Array[Int] = null): Tensor[T]

  /**
   * Shrunk the size of the storage to 0, and also the tensor size
   *
   * @return
   */
  def set(): Tensor[T]

  /**
   * Get a subset of the tensor on dim-th dimension. The offset is given by index, and length is
   * give by size. The important difference with select is that it will not reduce the dimension
   * number. For Instance
   * tensor =
   * 1 2 3
   * 4 5 6
   * tensor.narrow(1, 1, 1) is [1 2 3]
   * tensor.narrow(2, 2, 2) is
   * 2 3
   * 5 6
   *
   * @param dim
   * @param index
   * @param size
   * @return
   */
  def narrow(dim: Int, index: Int, size: Int): Tensor[T]

  /**
   * Copy the value of the given tensor to the current. They should have same size. It will use
   * the old storage
   *
   * @param other source tensor
   * @return current tensor
   */
  def copy(other: Tensor[T]): Tensor[T]

  /**
   * Apply a function to each element of the tensor and modified it value if it return a double
   *
   * @param func applied function
   * @return current tensor
   */
  def apply1(func: T => T): Tensor[T]

  /**
   * Map value of another tensor to corresponding value of current tensor and apply function on
   * the two value and change the value of the current tensor
   * The another tensor should has the same size of the current tensor
   *
   * @param other another tensor
   * @param func  applied function
   * @return current tensor
   */
  def map(other: Tensor[T], func: (T, T) => T): Tensor[T]

  /**
   * Removes all singleton dimensions of the tensor
   *
   * @return current tensor
   */
  def squeeze(): Tensor[T]

  /**
   * Removes given dimensions of the tensor if it's singleton
   *
   * @return current tensor
   */
  def squeeze(dim: Int): Tensor[T]

  /**
   * Create a new tensor that removes all singleton dimensions of the tensor
   *
   * @return create a new tensor
   */
  def squeezeNewTensor(): Tensor[T]

  /**
   * Return a new tensor with specified sizes. The input tensor must be contiguous, and the
   * elements number in the given sizes must be equal to the current tensor
   *
   * @param sizes
   * @return new tensor
   */
  def view(sizes: Int*): Tensor[T] = {
    view(sizes.toArray)
  }

  def view(sizes: Array[Int]): Tensor[T]

  /**
 *
 * Returns a tensor which contains all slices of size @param size
   * in the dimension @param dim. Step between two slices is given by @param step.
   *
   * @param dim
   * @param size
   * @param step Step between two slices
   * @return new tensor
   */
  def unfold(dim: Int, size: Int, step: Int): Tensor[T]

  /**
   * Repeating a tensor allocates new memory, unless result is provided, in which case its memory
   * is resized. sizes specify the number of times the tensor is repeated in each dimension.
   *
   * @param sizes
   * @return
   */
  def repeatTensor(sizes: Array[Int]): Tensor[T]

  /**
   * This is equivalent to this.expand(template.size())
   *
   * @param template the given tensor
   * @return
   */
  def expandAs(template: Tensor[T]): Tensor[T]

  /**
   * Expanding a tensor allocates new memory, tensor where singleton dimensions can be expanded
   * to multiple ones by setting the stride to 0. Any dimension that has size 1 can be expanded
   * to arbitrary value with new memory allocation. Attempting to expand along a dimension that
   * does not have size 1 will result in an error.
   *
   * @param sizes the size that tensor will expend to
   * @return
   */
  def expand(sizes: Array[Int]): Tensor[T]

  /**
   * Splits current tensor along dimension dim into a result table of Tensors of size size
   * (a number) or less (in the case of the last Tensor). The sizes of the non-dim dimensions
   * remain unchanged. Internally, a series of narrows are performed along dimensions dim.
   * Argument dim defaults to 1.
   *
   * @param size
   * @param dim
   * @return
   */
  def split(size: Int, dim: Int = 1): Array[Tensor[T]]

  /**
   * spilt one tensor into multi tensor along the `dim` dimension
   * @param dim the specific dimension
   * @return
   */
  def split(dim: Int) : Array[Tensor[T]]

  /**
   * convert the tensor to BreezeVector, the dimension of the tensor need to be 1.
   *
   * @return BrzDenseVector
   */
  def toBreezeVector(): BrzDenseVector[T]

  /**
   * convert the tensor to MLlibVector, the dimension of the
   * tensor need to be 1, and tensor need to be continuous.
   *
   * @return Vector
   */
  def toMLlibVector(): Vector

  /**
   * convert the tensor to BreezeMatrix, the dimension of the tensor need to be 2.
   *
   * @return BrzDenseMatrix
   */
  def toBreezeMatrix(): BrzDenseMatrix[T]

  /**
   * convert the tensor to MLlibMatrix, the dimension of the
   * tensor need to be 2, and tensor need to be continuous.
   *
   * @return Matrix
   */
  def toMLlibMatrix(): Matrix

  /**
   * return the tensor datatype( DoubleType or FloatType)
   *
   * @return
   */
  def getType(): TensorDataType

  /**
   * Compare and print differences between two tensors
   *
   * @param other
   * @param count
   * @return true if there's difference, vice versa
   */
  def diff(other: Tensor[T], count: Int = 1, reverse: Boolean = false): Boolean

  /**
   * view this.tensor and add a Singleton Dimension to `dim` dimension
   *
   * @param t source tensor
   * @param dim the specific dimension, default is 1
   * @return this
   */
  def addSingletonDimension(t: Tensor[T] = this, dim: Int = 1): Tensor[T]

  /**
   * create a new tensor without any change of the tensor
   *
   * @param sizes the size of the new Tensor
   * @return
   */
  def reshape(sizes: Array[Int]): Tensor[T]

  /**
   * Save the tensor to given path
   *
   * @param path
   * @param overWrite
   * @return
   */
  def save(path : String, overWrite : Boolean = false) : this.type

  override def toTable: Table =
    throw new IllegalArgumentException("Tensor cannot be cast to Table")

  /**
   * Return true because it's a Tensor implemented from [[Activity]]
   *
   * @return true
   */
  override def isTensor: Boolean = true


  /**
   * Return false because it's not a Table
   *
   * @return false
   */
  override def isTable: Boolean = false

  /**
   * Return tensor numeric
   * @return
   */
  def getTensorNumeric(): TensorNumeric[T]

  /**
   * Compare with other tensor. The shape of the other tensor must be same with this tensor.
   * If element wise difference is less than delta, return true.
   * @param other
   * @param delta
   * @return
   */
  def almostEqual(other: Tensor[T], delta : Double): Boolean = {
    var result = true
    this.map(other, (a, b) => {
      val tn = getTensorNumeric()
      if (tn.isGreater(tn.abs(tn.minus(a, b)), tn.fromType(delta))) {
        result = false
      }
      a
    })
    return result
  }
}

/**
 * Numeric type of tensor.
 */
sealed trait TensorDataType

object DoubleType extends TensorDataType

object FloatType extends TensorDataType

object Tensor {

  // pre-load MKL library. If we do not do it here,
  // libjmkl.so will be loaded when one of the methods of in MKL is called.
  MKL.isMKLLoaded

  /**
   * Returns an empty tensor.
   *
   * @param ev
   * @tparam T
   * @return
   */
  def apply[@specialized(Float, Double) T: ClassTag]()(
    implicit ev: TensorNumeric[T]): Tensor[T] = new DenseTensor[T]()

  /**
   * Create a tensor up to 5 dimensions. The tensor size will be `d1 x d2 x d3 x d4 x d5`.
   *
   * @param d1,(d2, d3, d4, d5)
   * @param ev
   * @tparam T
   * @return
   */
  def apply[@specialized(Float, Double) T: ClassTag](d1: Int)(
    implicit ev: TensorNumeric[T]): Tensor[T] = new DenseTensor[T](d1)

  def apply[@specialized(Float, Double) T: ClassTag](d1: Int, d2: Int)(
    implicit ev: TensorNumeric[T]): Tensor[T] = new DenseTensor[T](d1, d2)

  def apply[@specialized(Float, Double) T: ClassTag](d1: Int, d2: Int, d3: Int)(
    implicit ev: TensorNumeric[T]): Tensor[T] = new DenseTensor[T](d1, d2, d3)

  def apply[@specialized(Float, Double) T: ClassTag](d1: Int, d2: Int, d3: Int, d4: Int)(
    implicit ev: TensorNumeric[T]): Tensor[T] = new DenseTensor[T](d1, d2, d3, d4)

  def apply[@specialized(Float, Double) T: ClassTag](d1: Int, d2: Int, d3: Int, d4: Int, d5: Int)(
    implicit ev: TensorNumeric[T]): Tensor[T] = new DenseTensor[T](d1, d2, d3, d4, d5)

  /**
   * Create a tensor with a table
   * @param xs the table contains a multi-dimensional numbers
   * @return a new Tensor
   */
  def apply[@specialized(Float, Double) T: ClassTag](xs : Table)(
    implicit ev: TensorNumeric[T]): Tensor[T] = {
    val flatTable = xs.flatten()
    val content = new Array[T](flatTable.length())
    for (i <- 1 to content.length) {
      content(i - 1) = flatTable(i)
    }

    val dims = new ArrayBuffer[Int]()

    def getDims(xs: Table): ArrayBuffer[Int] = xs match {
      case _ if xs.length() != 0 =>
        dims.append(xs.length())
        if (xs(1).isInstanceOf[Table]) {
          getDims(xs(1))
        }
        dims
      case otherwise => dims
    }

    getDims(xs)

    new DenseTensor[T](
      new ArrayStorage[T](content), 0, dims.toArray,
      DenseTensor.size2Stride(dims.toArray), dims.length)
  }

  /**
   * Create a tensor on given dimensions. The tensor size will be the product of dims
   *
   * @param dims
   * @param ev
   * @tparam T
   * @return
   */
  def apply[@specialized(Float, Double) T: ClassTag](dims: Int*)(
    implicit ev: TensorNumeric[T]): Tensor[T] =
    new DenseTensor[T](new ArrayStorage[T](new Array[T](dims.product)), 0, dims.toArray,
      DenseTensor.size2Stride(dims.toArray), dims.length)

  /**
   * Create a tensor on given sizes. The tensor size will be the product of sizes
   *
   * @param sizes
   * @param ev
   * @tparam T
   * @return
   */
  def apply[@specialized(Float, Double) T: ClassTag](sizes: Array[Int])(
    implicit ev: TensorNumeric[T]): Tensor[T] =
    new DenseTensor(new ArrayStorage[T](new Array[T](sizes.product)), 0, sizes.clone(),
      DenseTensor.size2Stride(sizes.clone()), sizes.length)

  /**
   * Returns a tensor which uses the existing Storage storage.
   *
   * @param storage the given storage
   * @param ev
   * @tparam T
   * @return
   */
  def apply[@specialized(Float, Double) T: ClassTag](storage: Storage[T])(
    implicit ev: TensorNumeric[T]): Tensor[T] = {
    new DenseTensor(storage.asInstanceOf[Storage[T]])
  }

  /**
   * Returns a tensor with the given array and shape
   *
   * @param data the given storage
   * @param shape the given shape
   * @param ev
   * @tparam T
   * @return
   */
  def apply[@specialized(Float, Double) T: ClassTag](data: Array[T],
    shape: Array[Int])(implicit ev: TensorNumeric[T]): Tensor[T] = {
    new DenseTensor[T]().set(Storage[T](data), storageOffset = 1, sizes = shape)
  }

  /**
   * Returns a tensor which uses the existing Storage storage, starting at
   * position storageOffset (>=1). The size of each dimension of the tensor
   * is given by the optional Array size. If not given, the size will be computed
   * as the length of storage. The jump necessary to go from one element to the
   * next one in each dimension is given by the optional Array stride. If not
   * given, the stride() will be computed such that the tensor is as contiguous
   * as possible in memory.
   *
   * @param storage
   * @param storageOffset
   * @param size
   * @param stride
   * @param ev
   * @tparam T
   * @return
   */
  def apply[@specialized(Float, Double) T: ClassTag](storage: Storage[T],
                                                     storageOffset: Int,
                                                     size: Array[Int] = null,
                                                     stride: Array[Int] = null)
                                                    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    new DenseTensor(storage.asInstanceOf[Storage[T]], storageOffset, size, stride)
  }

  /**
   * create a tensor with a given tensor. The tensor will have same size
   * with the given tensor.
   *
   * @param other the given tensor
   * @param ev
   * @tparam T
   * @return
   */
  def apply[@specialized(Float, Double) T: ClassTag](other: Tensor[T])(
    implicit ev: TensorNumeric[T]): Tensor[T] = new DenseTensor(other)

  /**
   * create a tensor with a given breeze vector. The tensor will have the same size
   * with the given breeze vector.
   *
   * @param vector the given breeze vector
   * @param ev
   * @tparam T
   * @return
   */
  def apply[@specialized(Float, Double) T: ClassTag](vector: BrzDenseVector[T])(
    implicit ev: TensorNumeric[T]): Tensor[T] = apply(Storage(vector.data),
    vector.offset + 1, Array(vector.length), Array(vector.stride))

  /**
   * create a tensor with a given spark Densevector. The tensor will have the same size
   * with the given spark Densevector.
   *
   * @param vector the given spark Densevector
   * @return
   */
  def apply(vector: DenseVector): Tensor[Double] =
    apply[Double](Storage(vector.toArray))

  /**
   * create a tensor with a given breeze matrix. The tensor will have the same size with
   * the given breeze matrix.
   *
   * @param matrix the given breeze matrix
   * @param ev
   * @tparam T
   * @return
   */
  def apply[@specialized(Float, Double) T: ClassTag](matrix: BrzDenseMatrix[T])(
    implicit ev: TensorNumeric[T]): Tensor[T] = apply(Storage(matrix.data),
    matrix.offset + 1, Array(matrix.rows, matrix.cols),
    if (matrix.isTranspose) Array(matrix.majorStride, 1) else Array(1, matrix.majorStride))

  /**
   * create a tensor with a given spark Densematrix. The tensor will have the same size with
   * the given spark Densematrix.
   *
   * @param matrix
   * @return
   */
  def apply(matrix: DenseMatrix): Tensor[Double] = {
    val strides = if (matrix.isTransposed) {
      Array(matrix.numCols, 1)
    } else {
      Array(1, matrix.numRows) // column major
    }
    apply(Storage(matrix.toArray), 1, Array(matrix.numRows, matrix.numCols), strides)
  }

  /**
   * This is equivalent to DenseTensor.randperm[T](size)
   *
   * @param size
   * @param ev
   * @tparam T
   * @return
   */
  def randperm[@specialized(Float, Double) T: ClassTag](size: Int)(
    implicit ev: TensorNumeric[T]): Tensor[T] = DenseTensor.randperm[T](size)

  /**
   * This is equivalent to tensor.expand(sizes.toArray)
   *
   * @param tensor
   * @param sizes
   * @tparam T
   * @return
   */
  def expand[T](tensor: Tensor[T], sizes: Int*): Tensor[T] = tensor.expand(sizes.toArray)


  /**
   * This is equivalent to tensor.expandAs(template)
   *
   * @param tensor
   * @param template
   * @tparam T
   * @return
   */
  def expandAs[T](tensor: Tensor[T], template: Tensor[T]): Tensor[T] = tensor.expandAs(template)

  /**
   * This is equivalent to tensor.repeatTensor(sizes.toArray)
   *
   * @param tensor
   * @param sizes
   * @tparam T
   * @return
   */
  def repeatTensor[T](tensor: Tensor[T], sizes: Int*): Tensor[T] =
    tensor.repeatTensor(sizes.toArray)

  def load[T](path : String) : Tensor[T] = {
    File.load[Tensor[T]](path)
  }

  /**
   * This is equivalent to DenseTensor.range(xmin, xmax, step)
 *
   * @param xmin
   * @param xmax
   * @param step
   * @return
   */
  def range[@specialized(Float, Double) T: ClassTag](xmin: Double, xmax: Double, step: Int = 1)(
    implicit ev: TensorNumeric[T]): Tensor[T] = DenseTensor.range[T](xmin, xmax, step)

  /**
   * return a tensor of sizes filled with 1.
   * @param sizes
   * @return a tensor
   */
  def ones[@specialized(Float, Double) T: ClassTag](sizes: Int*)(
    implicit ev: TensorNumeric[T]): Tensor[T] = {
    DenseTensor.ones[T](sizes.toArray)
  }

  /**
   * Returns a 1D Gaussian kernel of size size, mean mean and standard deviation sigma.
   * @param size
   * @param sigma
   * @param amplitude
   * @param normalize
   * @param mean
   * @param tensor If tensor is set, will discard size, and write result to tensor.
   * @return
   */
  def gaussian1D[@specialized(Float, Double) T: ClassTag](
      size: Int = 3,
      sigma: Double = 0.25,
      amplitude: Int = 1,
      normalize: Boolean = false,
      mean: Double = 0.5,
      tensor: Tensor[T] = null)(implicit ev: TensorNumeric[T]): Tensor[T] = {
    DenseTensor.gaussian1D[T](size, sigma, amplitude, normalize, mean, tensor)
  }
}
