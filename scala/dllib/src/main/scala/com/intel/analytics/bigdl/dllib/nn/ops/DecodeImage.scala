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
package com.intel.analytics.bigdl.nn.ops

import java.awt.image.{BufferedImage, DataBufferByte}
import java.io.ByteArrayInputStream
import java.nio.{ByteBuffer, ByteOrder}
import javax.imageio.ImageIO

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.tensorflow.framework.DataType

import scala.reflect.ClassTag

class DecodeImage[T: ClassTag](val channels: Int)(implicit ev: TensorNumeric[T])
  extends Operation[Tensor[ByteString], Tensor[Int], T] {

  output = Tensor[Int]()
  override def updateOutput(input: Tensor[ByteString]): Tensor[Int] = {
    require(input.isScalar, "only support ByteString scalar")
    val image = ImageIO.read(new ByteArrayInputStream(input.value().toByteArray))
    require(image != null, "Can't decode image")
    val imageWidth = image.getWidth
    val imageHeight = image.getHeight

    val expectedChannels = if (channels == 0) {
      image.getColorModel.getNumComponents
    } else {
      require(channels == image.getColorModel.getNumComponents,
        "Only support inputs channels equal to desired channels")
      channels
    }

    output.resize(imageHeight, imageWidth, expectedChannels)

    val outputData = output.storage().array()
    val offset = output.storageOffset() - 1
    val length = imageHeight * imageWidth * expectedChannels

    copyImageData(image, outputData, offset, length)
    output
  }

  protected def copyImageData(image: BufferedImage,
                              outputData: Array[Int],
                              offset: Int,
                              length: Int): Unit = {
    val data = image.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData

    bytesToInts(data, outputData, offset, length)
  }

  private def bytesToInts(bytes: Array[Byte], ints: Array[Int], start: Int, length: Int): Unit = {
    if (bytes.length == length) {
      var i = 0
      while (i < length) {
        ints(i + start) = bytes(i) & 0xff
        i += 1
      }
    } else if (bytes.length * 3 == length) {
      var i = 0
      while (i < length) {
        val index = i / 3
        ints(i + start) = bytes(index) & 0xff
        i += 1
      }
    } else {
      throw new IllegalArgumentException("image data is not equal to output buffer")
    }
  }

  override def updateGradInput(input: Tensor[ByteString],
                               gradOutput: Tensor[Int]): Tensor[ByteString] = {
    throw new UnsupportedOperationException("no backward on ParseExample")
  }
}

class DecodeJpeg[T: ClassTag](channels: Int, ratio: Int = 1)(implicit ev: TensorNumeric[T])
  extends DecodeImage[T](channels) {
  require(ratio == 1, "currently not supported sub-sampling")
}

class DecodePng[T: ClassTag](channels: Int)(implicit ev: TensorNumeric[T])
  extends DecodeImage[T](channels)

class DecodeGif[T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends DecodeImage[T](3) {

  override def updateOutput(input: Tensor[ByteString]): Tensor[Int] = {
    require(input.isScalar, "only support ByteString scalar")

    val reader = ImageIO.getImageReadersByFormatName("gif").next()

    val is = ImageIO.createImageInputStream(new ByteArrayInputStream(input.value().toByteArray))

    // val reader = ImageIO.getImageReaders(is).next()

    reader.setInput(is)

    val numOfFrames = reader.getNumImages(true)
    val imageHeight = reader.getHeight(0)
    val imageWidth = reader.getWidth(0)

    output.resize(numOfFrames, imageHeight, imageWidth, channels)
    val outputData = output.storage().array()
    val offset = output.storageOffset() - 1
    val imageSize = imageHeight * imageWidth * channels

    var i = 0
    while (i < numOfFrames) {
      val image = reader.read(i)
      require(image != null, s"Can't decode ${i}th frame")
      require(imageHeight == image.getHeight,
        s"Different frame should have the same height," +
          s"first image height: $imageHeight, ${i}th image height: ${image.getHeight}")
      require(imageWidth == image.getWidth,
        s"Different frame should have the same width," +
          s"first image width: $imageWidth, ${i}th image width: ${image.getWidth}")

      val currentOffset = offset + i * imageSize

      copyImageData(image, outputData, currentOffset, imageSize)

      i = i + 1
    }
    output
  }

}

class DecodeRaw[T: ClassTag](val outType: DataType,
                             val littleEndian: Boolean)(implicit ev: TensorNumeric[T])
  extends Operation[Tensor[ByteString], Activity, T] {
  output = {
    outType match {
      case DataType.DT_UINT8 => Tensor[Int]()
      case DataType.DT_INT16 => Tensor[Int]()
      case DataType.DT_INT32 => Tensor[Int]()
      case DataType.DT_INT8 => Tensor[Int]()
      case DataType.DT_INT64 => Tensor[Long]()
      case DataType.DT_FLOAT => Tensor[Float]()
      case DataType.DT_DOUBLE => Tensor[Double]()
      case _ => throw new IllegalArgumentException(s"$outType are not supported")
    }
  }

  @transient private val byteOrder =
    if (littleEndian) ByteOrder.LITTLE_ENDIAN else ByteOrder.BIG_ENDIAN

  override def updateOutput(input: Tensor[ByteString]): Activity = {
    require(input.isContiguous(), "only support contiguous input")
    val offset = input.storageOffset() - 1
    val data = input.storage().array()
    val firstElem = data(offset)

    val buffer = ByteBuffer.wrap(firstElem.toByteArray)
    buffer.order(byteOrder)
    outType match {
      case DataType.DT_UINT8 => decodeUint8(input, buffer.array().length)
      case DataType.DT_INT8 => decodeInt8(input, buffer.array().length)
      case DataType.DT_INT16 => decodeInt16(input, buffer.asShortBuffer().array().length)
      case DataType.DT_INT32 => decodeInt32(input, buffer.asIntBuffer().array().length)
      case DataType.DT_INT64 => decodeInt64(input, buffer.asLongBuffer().array().length)
      case DataType.DT_FLOAT => decodeFloat(input, buffer.asFloatBuffer().array().length)
      case DataType.DT_DOUBLE => decodeDouble(input, buffer.asDoubleBuffer().array().length)
    }
    output
  }

  private def decodeDouble(input: Tensor[ByteString], featureSize: Int): Unit = {
    val typedOutput = output.asInstanceOf[Tensor[Double]]
    val size = input.size().toSeq :+ featureSize
    typedOutput.resize(size.toArray)

    val outputData = typedOutput.storage().array()
    val outputOffset = typedOutput.storageOffset() - 1
    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1

    val dataSize = input.nElement()
    var i = 0
    while (i < dataSize) {
      val bytes = inputData(inputOffset + i).toByteArray
      val buffer = ByteBuffer.wrap(bytes)
      buffer.order(byteOrder)
      val typedInputData = buffer.asDoubleBuffer().array()
      require(typedInputData.length == featureSize,
        s"each element should have the same size, first elem size: $featureSize, " +
          s"${i}th elem size: ${typedInputData.length}")
      System.arraycopy(typedInputData, 0, outputData, outputOffset + i * featureSize, featureSize)
      i = i + 1
    }
  }

  private def decodeFloat(input: Tensor[ByteString], featureSize: Int): Unit = {
    val typedOutput = output.asInstanceOf[Tensor[Float]]
    val size = input.size().toSeq :+ featureSize
    typedOutput.resize(size.toArray)

    val outputData = typedOutput.storage().array()
    val outputOffset = typedOutput.storageOffset() - 1
    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1

    val dataSize = input.nElement()
    var i = 0
    while (i < dataSize) {
      val bytes = inputData(inputOffset + i).toByteArray
      val buffer = ByteBuffer.wrap(bytes)
      buffer.order(byteOrder)
      val typedInputData = buffer.asFloatBuffer().array()
      require(typedInputData.length == featureSize,
        s"each element should have the same size, first elem size: $featureSize, " +
          s"${i}th elem size: ${typedInputData.length}")
      System.arraycopy(typedInputData, 0, outputData, outputOffset + i * featureSize, featureSize)
      i = i + 1
    }
  }

  private def decodeInt32(input: Tensor[ByteString], featureSize: Int): Unit = {
    val typedOutput = output.asInstanceOf[Tensor[Int]]
    val size = input.size().toSeq :+ featureSize
    typedOutput.resize(size.toArray)

    val outputData = typedOutput.storage().array()
    val outputOffset = typedOutput.storageOffset() - 1
    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1

    val dataSize = input.nElement()
    var i = 0
    while (i < dataSize) {
      val bytes = inputData(inputOffset + i).toByteArray
      val buffer = ByteBuffer.wrap(bytes)
      buffer.order(byteOrder)
      val typedInputData = buffer.asIntBuffer().array()
      require(typedInputData.length == featureSize,
        s"each element should have the same size, first elem size: $featureSize, " +
          s"${i}th elem size: ${typedInputData.length}")
      System.arraycopy(typedInputData, 0, outputData, outputOffset + i * featureSize, featureSize)
      i = i + 1
    }
  }

  private def decodeInt64(input: Tensor[ByteString], featureSize: Int): Unit = {
    val typedOutput = output.asInstanceOf[Tensor[Long]]
    val size = input.size().toSeq :+ featureSize
    typedOutput.resize(size.toArray)

    val outputData = typedOutput.storage().array()
    val outputOffset = typedOutput.storageOffset() - 1
    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1

    val dataSize = input.nElement()
    var i = 0
    while (i < dataSize) {
      val bytes = inputData(inputOffset + i).toByteArray
      val buffer = ByteBuffer.wrap(bytes)
      buffer.order(byteOrder)
      val typedInputData = buffer.asLongBuffer().array()
      require(typedInputData.length == featureSize,
        s"each element should have the same size, first elem size: $featureSize, " +
          s"${i}th elem size: ${typedInputData.length}")
      System.arraycopy(typedInputData, 0, outputData, outputOffset + i * featureSize, featureSize)
      i = i + 1
    }
  }

  private def decodeInt16(input: Tensor[ByteString], featureSize: Int): Unit = {
    val typedOutput = output.asInstanceOf[Tensor[Int]]
    val size = input.size().toSeq :+ featureSize
    typedOutput.resize(size.toArray)

    val outputData = typedOutput.storage().array()
    val outputOffset = typedOutput.storageOffset() - 1
    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1

    val dataSize = input.nElement()
    var i = 0
    while (i < dataSize) {
      val bytes = inputData(inputOffset + i).toByteArray
      val buffer = ByteBuffer.wrap(bytes)
      buffer.order(byteOrder)
      val typedInputData = buffer.asShortBuffer().array()
      require(typedInputData.length == featureSize,
        s"each element should have the same size, first elem size: $featureSize, " +
          s"${i}th elem size: ${typedInputData.length}")
      System.arraycopy(typedInputData, 0, outputData, outputOffset + i * featureSize, featureSize)
      i = i + 1
    }
  }

  private def decodeInt8(input: Tensor[ByteString], featureSize: Int): Unit = {
    val typedOutput = output.asInstanceOf[Tensor[Int]]
    val size = input.size().toSeq :+ featureSize
    typedOutput.resize(size.toArray)

    val outputData = typedOutput.storage().array()
    val outputOffset = typedOutput.storageOffset() - 1
    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1

    val dataSize = input.nElement()
    var i = 0
    while (i < dataSize) {
      val bytes = inputData(inputOffset + i).toByteArray
      val typedInputData = bytes
      require(typedInputData.length == featureSize,
        s"each element should have the same size, first elem size: $featureSize, " +
          s"${i}th elem size: ${typedInputData.length}")
      var j = 0
      while (j < featureSize) {
        outputData(outputOffset + i * featureSize + j) = typedInputData(j)
        j = j + 1
      }
      i = i + 1
    }
  }

  private def decodeUint8(input: Tensor[ByteString], featureSize: Int): Unit = {
    val typedOutput = output.asInstanceOf[Tensor[Int]]
    val size = input.size().toSeq :+ featureSize
    typedOutput.resize(size.toArray)

    val outputData = typedOutput.storage().array()
    val outputOffset = typedOutput.storageOffset() - 1
    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1

    val dataSize = input.nElement()
    var i = 0
    while (i < dataSize) {
      val bytes = inputData(inputOffset + i).toByteArray
      val typedInputData = bytes
      require(typedInputData.length == featureSize,
        s"each element should have the same size, first elem size: $featureSize, " +
          s"${i}th elem size: ${typedInputData.length}")
      var j = 0
      while (j < featureSize) {
        outputData(outputOffset + i * featureSize + j) =
          (typedInputData(j) & 0xff.toShort).toShort
        j = j + 1
      }
      i = i + 1
    }
  }

  override def updateGradInput(input: Tensor[ByteString], gradOutput: Activity):
  Tensor[ByteString] = {
    throw new IllegalArgumentException()
  }
}



