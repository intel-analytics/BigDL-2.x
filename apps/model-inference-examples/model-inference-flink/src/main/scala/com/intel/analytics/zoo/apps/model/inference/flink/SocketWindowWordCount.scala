package com.intel.analytics.zoo.apps.model.inference.flink

import org.apache.flink.api.java.utils.ParameterTool
import org.apache.flink.streaming.api.scala._
import org.apache.flink.streaming.api.windowing.time.Time

object SocketWindowWordCount {

  case class WordWithCount(word: String, count: Long)

  def main(args: Array[String]) : Unit = {

    // the host and the port to connect to
    var hostname: String = "localhost"
    var port: Int = 0

    try {
      val params = ParameterTool.fromArgs(args)
      hostname = if (params.has("hostname")) params.get("hostname") else "localhost"
      port = params.getInt("port")
    } catch {
      case e: Exception => {
        System.err.println("No port specified. Please run 'SocketWindowWordCount " +
          "--hostname <hostname> --port <port>', where hostname (localhost by default) and port " +
          "is the address of the text server")
        System.err.println("To start a simple text server, run 'netcat -l <port>' " +
          "and type the input text into the command line")
        return
      }
    }

    println("start SocketWindowWordCount job...")

    // get the execution environment
    val env: StreamExecutionEnvironment = StreamExecutionEnvironment.getExecutionEnvironment

    // get input data by connecting to the socket
    val text: DataStream[String] = env.socketTextStream(hostname, port, '\n')

    // parse the data, group it, window it, and aggregate the counts
    val windowCounts: DataStream[WordWithCount] = text
      .flatMap { w => w.split("\\s") }
      .map { w => WordWithCount(w, 1) }
      .keyBy("word")
      .timeWindow(Time.seconds(5))
      .sum("count")

    windowCounts.map(wc => println(s"$wc")).setParallelism(1)

    env.execute("Socket Window WordCount")
  }

}
