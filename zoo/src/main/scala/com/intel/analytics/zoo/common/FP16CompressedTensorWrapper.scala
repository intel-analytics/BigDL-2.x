package com.intel.analytics.bigdl.parameters

import java.nio.ByteBuffer

import com.intel.analytics.bigdl.tensor.Tensor

import scala.reflect.ClassTag

class FP16CompressedTensorWrapper[T: ClassTag]() {

  private var fp16CompressedTensor: FP16CompressedTensor[T] = null

  def apply(length: Int) = {
    fp16CompressedTensor = new FP16CompressedTensor[T](length)
    this
  }

  def apply(tensor: Tensor[T]) = {
    fp16CompressedTensor = new FP16CompressedTensor[T](tensor)
    this
  }

  def apply(bytes: ByteBuffer) = {
    fp16CompressedTensor = new FP16CompressedTensor[T](bytes)
    this
  }

  def compress(offset: Int, src: Tensor[T], srcOffset: Int, length: Int): Unit = {
    fp16CompressedTensor.compress(offset, src, srcOffset, length)
  }

  def compress(src: Tensor[T]): Unit = compress(0, src, 0, src.nElement())

  def deCompress(srcOffset: Int, tensor: Tensor[T],
                 tgtOffset: Int, length: Int): Unit = {
    fp16CompressedTensor.deCompress(srcOffset, tensor, tgtOffset, length)
  }

  def deCompress(tensor: Tensor[T]): Unit = fp16CompressedTensor.deCompress(tensor)

  def bytes(): ByteBuffer = fp16CompressedTensor.bytes()
}
