package com.intel.analytics.zoo.examples

import scala.io.Source
import java.io.File

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.layers.BERT

import scala.collection.mutable.ArrayBuffer

object convertBert {
  def main(args: Array[String]): Unit = {
//    val t = BERT.loadModel[Float]("/tmp/zoo-bert.model")
    val vocab = 30522
    val hiddenSize = 768
    val intermediateSize = 3072
    val seqLen = 512
    val preTrainModel = BERT[Float](vocab = vocab, hiddenSize = hiddenSize,
      intermediateSize = intermediateSize, maxPositionLen = seqLen)
    val testShape = Shape(List(Shape(1, seqLen), Shape(1, seqLen),
      Shape(1, seqLen), Shape(1, 1, 1, seqLen)))
    preTrainModel.build(testShape)
    val weight = preTrainModel.parameters()._1

    var i = 0

    val path = "/tmp/numpy/"
    val d = new File(path)
    require(d.exists && d.isDirectory)

    val buf = ArrayBuffer[Array[Float]]()
    var filename = path + "bert_embeddings_position_embeddings_0.out"
    for (line <- Source.fromFile(filename).getLines) {
      buf.append(line.split(" ").map(_.toFloat
      ))
    }
    var param = Tensor[Float](buf.flatten.toArray, Array(buf.size, buf.head.length))
    require(param.size.deep == weight(i).size().deep)
    weight(i).set(param)
    i += 1

    buf.clear()
    filename = path + "bert_embeddings_word_embeddings_0.out"
    for (line <- Source.fromFile(filename).getLines) {
      buf.append(line.split(" ").map(_.toFloat
      ))
    }
    param = Tensor[Float](buf.flatten.toArray, Array(buf.size, buf.head.length))
    require(param.size.deep == weight(i).size().deep)
    weight(i).set(param)
    i += 1

    buf.clear()
    filename = path + "bert_embeddings_token_type_embeddings_0.out"
    for (line <- Source.fromFile(filename).getLines) {
      buf.append(line.split(" ").map(_.toFloat
      ))
    }
    param = Tensor[Float](buf.flatten.toArray, Array(buf.size, buf.head.length))
    require(param.size.deep == weight(i).size().deep)
    weight(i).set(param)
    i += 1

    buf.clear()
    filename = path + "bert_embeddings_LayerNorm_gamma_0.out"
    for (line <- Source.fromFile(filename).getLines) {
      buf.append(line.split(" ").map(_.toFloat
      ))
    }
    param = Tensor[Float](buf.flatten.toArray, Array(buf.head.length, buf.size))
    require(param.size.deep == weight(i).size().deep)
    weight(i).set(param)
    i += 1

    buf.clear()
    filename = path + "bert_embeddings_LayerNorm_beta_0.out"
    for (line <- Source.fromFile(filename).getLines) {
      buf.append(line.split(" ").map(_.toFloat
      ))
    }
    param = Tensor[Float](buf.flatten.toArray, Array(buf.head.length, buf.size))
    require(param.size.deep == weight(i).size().deep)
    weight(i).set(param)
    i += 1

    // mapping block parameters
    var blockId = 0
    while (blockId < 12) {
      val prefix = s"bert_encoder_layer_${blockId}_"
      val qkv_w_files = Array("attention_self_query_kernel_0.out",
        "attention_self_key_kernel_0.out",
        "attention_self_value_kernel_0.out")
      val qkv_w_fileList = qkv_w_files.map(path + prefix + _)
      val qkv_w = qkv_w_fileList.map { file =>
        val buf = ArrayBuffer[Array[Float]]()
        for (line <- Source.fromFile(file).getLines) {
          buf.append(line.split(" ").map(_.toFloat
          ))
        }
        Tensor[Float](buf.flatten.toArray, Array(buf.size, buf.head.length))
      }
      weight(i).narrow(1, 1, hiddenSize).copy(qkv_w(0).t())
      weight(i).narrow(1, 1 + hiddenSize, hiddenSize).copy(qkv_w(1).t())
      weight(i).narrow(1, 1 + hiddenSize * 2, hiddenSize).copy(qkv_w(2).t())
      i += 1

      val qkv_b_files = Array(
        "attention_self_query_bias_0.out",
        "attention_self_key_bias_0.out",
        "attention_self_value_bias_0.out")
      val qkv_b_fileList = qkv_b_files.map(path + prefix + _)
      val qkv_b = qkv_b_fileList.map { file =>
        val buf = ArrayBuffer[Array[Float]]()
        for (line <- Source.fromFile(file).getLines) {
          buf.append(line.split(" ").map(_.toFloat
          ))
        }
        Tensor[Float](buf.flatten.toArray, Array(buf.size, buf.head.length)).squeeze(2)
      }
      weight(i).narrow(1, 1, hiddenSize).copy(qkv_b(0))
      weight(i).narrow(1, 1 + hiddenSize, hiddenSize).copy(qkv_b(1))
      weight(i).narrow(1, 1 + hiddenSize * 2, hiddenSize).copy(qkv_b(2))
      i += 1

      val files = Array(
        "attention_output_dense_kernel_0.out",
        "attention_output_dense_bias_0.out",
        "attention_output_LayerNorm_gamma_0.out",
        "attention_output_LayerNorm_beta_0.out",
        "intermediate_dense_kernel_0.out",
        "intermediate_dense_bias_0.out",
        "output_dense_kernel_0.out",
        "output_dense_bias_0.out",
        "output_LayerNorm_gamma_0.out",
        "output_LayerNorm_beta_0.out"
      )

      val fileList = files.map(path + prefix + _)
      fileList.foreach { file =>
        val buf = ArrayBuffer[Array[Float]]()
        for (line <- Source.fromFile(file).getLines) {
          buf.append(line.split(" ").map(_.toFloat
          ))
        }
        val param = Tensor[Float](buf.flatten.toArray, Array(buf.size, buf.head.length))
        if (file.contains("bias")) param.squeeze(2)
//        require(param.size.deep == weight(i).size().deep)
        if (file.contains("kernel")) {
          weight(i).set(param.t())
        } else weight(i).set(param)
        i += 1
      }
      blockId += 1
    }

    buf.clear()
    filename = path + "bert_pooler_dense_kernel_0.out"
    for (line <- Source.fromFile(filename).getLines) {
      buf.append(line.split(" ").map(_.toFloat
      ))
    }
    param = Tensor[Float](buf.flatten.toArray, Array(buf.size, buf.head.length))
    require(param.size.deep == weight(i).size().deep)
    weight(i).set(param.t())
    i += 1

    buf.clear()
    filename = path + "bert_pooler_dense_bias_0.out"
    for (line <- Source.fromFile(filename).getLines) {
      buf.append(line.split(" ").map(_.toFloat
      ))
    }
    param = Tensor[Float](buf.flatten.toArray, Array(buf.size, buf.head.length))
    weight(i).set(param)

    preTrainModel.saveModule("/tmp/zoo-bert.model", overWrite = true)
    println("convert done!")
  }
}
