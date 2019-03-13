package com.intel.analytics.zoo.apps.streaming

import java.io.{File, PrintWriter}
import java.net.URI

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.fs.Path
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser

/**
  * Borrowed from https://github.com/intel-analytics/meph-streaming
  * The java program periodically write a text file to streamingPath, which contains 2 image paths.
  */
object ImagePathWriter {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)

    parser.parse(args, PathWriterParam()).foreach { params =>
      val path = new Path(params.imageSourcePath)
      val fs = FileSystem.get(path.toUri, new Configuration())
      val lists = if (params.imageSourcePath.contains("hdfs")) {
        // HDFS
        fs.listStatus(path).map(_.getPath.toString)
      } else {
        // Local files
        new File(params.imageSourcePath).listFiles().map(_.getAbsolutePath)
      }
      lists.grouped(10).zipWithIndex.foreach { case (batch, id) =>
        val batchPath = new Path(params.streamingPath, id + ".txt").toString
        if (params.imageSourcePath.contains("hdfs")) {
          // Write to HDFS
          val outstream = fs.create(new Path(batchPath), true)
          batch.foreach(line => outstream.writeBytes(line + "\n"))
          outstream.close()
        } else {
          // Write to local
          val pw = new PrintWriter(batchPath)
          batch.foreach(line => pw.println(line))
          pw.close()
        }
        println("wrote " + batchPath)
        Thread.sleep(4000)
      }
    }
  }

  private case class PathWriterParam(imageSourcePath: String = "",
                                     streamingPath: String = "file:///tmp/zoo/streaming")

  private val parser = new OptionParser[PathWriterParam]("PathWriterParam") {
    head("PathWriterParam")
    opt[String]("imageSourcePath")
      .text("folder that contains the source images, local file system only")
      .action((x, c) => c.copy(imageSourcePath = x))
      .required()
    opt[String]("streamingPath")
      .text("folder that used to store the streaming paths, local file system only, i.e. file:///path")
      .action((x, c) => c.copy(streamingPath = x))
  }
}
