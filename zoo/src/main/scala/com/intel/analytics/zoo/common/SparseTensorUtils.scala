package com.intel.analytics.bigdl.tensor

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object SparseTensorUtils {
  def addSparseTensor[T: ClassTag](tensor: Tensor[T], tensor2: Tensor[T])
                                  (implicit ev: TensorNumeric[T]): SparseTensor[T] = {
    val sparseT = tensor.asInstanceOf[SparseTensor[T]]
    val sparseT2 = tensor2.asInstanceOf[SparseTensor[T]]
    require(sparseT._indices.length == 2 && sparseT2._indices.length == 2,
      "Only support 2D sparse tensor")
    require(sparseT._shape.deep  == sparseT2._shape.deep,
      "added sparse tensors should have the same shape")
    val tIndice1 = sparseT._indices.head.array()
    val tIndice2 = sparseT._indices.last.array()
    val tValue = sparseT.storage().array()

    val t2Indice1 = sparseT2._indices.head.array()
    val t2Indice2 = sparseT2._indices.last.array()
    val t2Value = sparseT2.storage().array()

    val indice = ArrayBuffer[Int]()
    val indice2 = ArrayBuffer[Int]()
    val value = ArrayBuffer[T]()

    var i = 0
    var j = 0

    while (i < tIndice1.length && j < t2Indice1.length) {
      if (tIndice1(i) == t2Indice1(j)) {
        indice += tIndice1(i)
        if (tIndice2(i) == t2Indice2(j)) {
          indice2 += tIndice2(i)
          value += (ev.plus(tValue(i), t2Value(j)))
          i += 1
          j += 1
        } else if (tIndice2(i) < t2Indice2(j)) {
          indice2 += tIndice2(i)
          value += tValue(i)
          i += 1
        } else {
          indice2 += t2Indice2(j)
          value += t2Value(j)
          j += 1
        }
      } else if (tIndice1(i) > t2Indice1(j)) {
        indice += t2Indice1(j)
        indice2 += t2Indice2(j)
        value += t2Value(j)
        j += 1
      } else {
        indice += tIndice1(i)
        indice2 += tIndice2(i)
        value += tValue(i)
        i += 1
      }
    }
    if (i < tIndice1.length) {
      indice ++= tIndice1.slice(i, tIndice1.length)
      indice2 ++= tIndice2.slice(i, tIndice1.length)
      value ++= tValue.slice(i, tIndice1.length)
    } else if (j < t2Indice1.length) {
      indice ++= t2Indice1.slice(j, t2Indice1.length)
      indice2 ++= t2Indice2.slice(j, t2Indice1.length)
      value ++= t2Value.slice(j, t2Indice1.length)
    }
    SparseTensor(Array(indice.toArray, indice2.toArray), Storage[T](value.toArray),
      sparseT._shape, sparseT._shape.length)
  }

  def addConstant[T: ClassTag](tensor: Tensor[T], constant: T)
                              (implicit ev: TensorNumeric[T]): SparseTensor[T] = {
    val sparseT = tensor.asInstanceOf[SparseTensor[T]]
    val values = sparseT._values.array()

    SparseTensor(sparseT._indices.map(_.toArray.clone()),
      Storage[T](values.map(ev.plus(_, constant))),
      sparseT._shape, sparseT._shape.length)
  }

  def copySparseTensor[T: ClassTag](src: SparseTensor[T], dst: SparseTensor[T])
    (implicit ev: TensorNumeric[T]): Unit = {
    require(src._indices.head.size == dst._indices.head.size &&
      src._indices.last.size == dst._indices.last.size &&
      src._values.size == dst._values.size, "dst must has the same size with src")

    dst._indices.head.copy(src._indices.head)
    dst._indices.last.copy(src._indices.last)
    dst._values.copy(src._values)
  }
}
