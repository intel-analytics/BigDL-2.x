package com.intel.analytics.zoo.serving.serialization

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, ObjectInputStream, ObjectOutputStream}

class StreamSerializer {

}

object StreamSerializer {
  def objToBytes(obj: Object): Array[Byte] = {
    val bos = new ByteArrayOutputStream()
    val out = new ObjectOutputStream(bos)
    try {
      out.writeObject(obj)
      out.flush()
      bos.toByteArray()
    } catch {
      case e: Exception => e.printStackTrace()
        throw new Error("Ser error")
    } finally {
      try {
        bos.close()
      } catch {
        case e: Exception => // ignore close exception
      }
    }
  }
  def bytesToObj(bytes: Array[Byte]): Object = {
    val bis = new ByteArrayInputStream(bytes)
    val in = new ObjectInputStream(bis)
    try {
      in.readObject()
    } catch {
      case e: Exception => e.printStackTrace()
        throw new Error("Deser error")
    } finally {
      try {
        bis.close()
      } catch {
        case e: Exception => // ignore close exception
      }
    }
  }
}
