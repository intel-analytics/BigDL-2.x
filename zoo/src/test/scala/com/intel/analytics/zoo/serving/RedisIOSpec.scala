/*
 * Copyright 2021 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.serving

import java.io._
import java.util
import java.util.AbstractMap.SimpleEntry
import java.util.concurrent.{ExecutorService, Executors}

import scala.util.control.Breaks._
import scala.collection.JavaConverters._
import com.intel.analytics.zoo.serving.http.{PredictionInputMessage, _}
import com.intel.analytics.zoo.serving.utils.Conventions
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import redis.clients.jedis.{Jedis, ScanParams, ScanResult, StreamEntryID}
import com.intel.analytics.zoo.serving.pipeline.{RedisUtils}
import org.apache.commons.io.IOUtils
import redis.embedded.exceptions.EmbeddedRedisException
import redis.embedded.{AbstractRedisInstance, Redis, RedisExecProvider, RedisServer, RedisServerBuilder}
import redis.embedded.util.OS
import scopt.OptionParser

//The RedisServer implementation is based on:
//  https://github.com/kstyrc/embedded-redis
class PrintReaderRunnable(val reader: BufferedReader) extends Runnable {
  override def run(): Unit = {
    try
      this.readLines()
    finally IOUtils.closeQuietly(this.reader)
  }

  def readLines(): Unit = {
    breakable{
      while ( {
        true
      }) {
        try {
          var line = this.reader.readLine()
          if (line != null) {
            System.out.println(line)
            break()
          }
        } catch {
          case var2: IOException =>
            var2.printStackTrace()
        }
        return
      }
    }
  }
}

abstract class AbstractRedisInstance protected(val port: Int) extends Redis {
  protected var args: util.List[String] = _
  private var active = false
  private var redisProcess: Process = _
  final private val executor = Executors.newSingleThreadExecutor

  override def isActive: Boolean = this.active

  @throws[EmbeddedRedisException]
  override def start(): Unit = {
    if (this.active) throw new EmbeddedRedisException(
      "This redis server instance is already running...")
    else try {
      this.redisProcess = this.createRedisProcessBuilder.start
      this.logErrors()
      this.awaitRedisServerReady()
      this.active = true
    } catch {
      case var2: IOException =>
        throw new EmbeddedRedisException("Failed to start Redis instance", var2)
    }
  }

  private def logErrors(): Unit = {
    val errorStream = this.redisProcess.getErrorStream
    val reader = new BufferedReader(new InputStreamReader(errorStream))
    val printReaderTask = new PrintReaderRunnable(reader)
    this.executor.submit(printReaderTask)
  }

  @throws[IOException]
  private def awaitRedisServerReady(): Unit = {
    val reader = new BufferedReader(new InputStreamReader(this.redisProcess.getInputStream))
    var outputLine = ""
    try
        do {
          outputLine = reader.readLine
          if (outputLine == null) throw new RuntimeException(
            "Can't start redis server. Check logs for details.")
        } while ( {
          !outputLine.matches(this.redisReadyPattern)
        })
    finally IOUtils.closeQuietly(reader)
  }

  protected def redisReadyPattern: String

  private def createRedisProcessBuilder = {
    val executable = new File(this.args.get(0).asInstanceOf[String])
    val pb = new ProcessBuilder(this.args)
    pb.directory(executable.getParentFile)
    pb
  }

  @throws[EmbeddedRedisException]
  override def stop(): Unit = {
    if (this.active) {
      this.redisProcess.destroy()
      this.tryWaitFor()
      this.active = false
    }
  }

  private def tryWaitFor(): Unit = {
    try
      this.redisProcess.waitFor
    catch {
      case var2: InterruptedException =>
        throw new EmbeddedRedisException("Failed to stop redis instance", var2)
    }
  }

  override def ports: util.List[Integer] = util.Arrays.asList(this.port)
}


object RedisServer {
  private val REDIS_READY_PATTERN = ".*The server is now ready to accept connections on port.*"
  private val DEFAULT_REDIS_PORT = 6379

  def builder : RedisServerBuilder = new RedisServerBuilder
}

class RedisServer @throws[IOException]
(port : Int) extends AbstractRedisInstance(port) {
  var executable: File = RedisExecProvider.defaultProvider.get
  this.args = util.Arrays.asList(executable.getAbsolutePath,
    "--port", Integer.toString(port.intValue))

  def this(executable: File, port: Integer) {
    this(port)
    this.executable = executable
    this.args = util.Arrays.asList(executable.getAbsolutePath,
      "--port", Integer.toString(port.intValue))
  }

  def this(redisExecProvider: RedisExecProvider, port: Integer) {
    this(port)
    this.executable = redisExecProvider.get.getAbsoluteFile
    this.args = util.Arrays.asList(redisExecProvider.get.getAbsolutePath,
      "--port", Integer.toString(port.intValue))
  }

  override protected def redisReadyPattern: String = ".*Ready to accept connections.*"
}

class RedisIOSpec(path : String) extends FlatSpec with Matchers with BeforeAndAfter with Supportive {
  val redisHost = "localhost"
  val redisPort = 6379
  val pathToRedisExecutable = path
  var redisServer: RedisServer = _
  var jedis: Jedis = _

  val inputHash = List("index1" -> "data1", "index2" -> "data2")
  val inputXStream = Map("name1" -> "A", "name2" -> "B").asJava

  before {
    val customProvider = RedisExecProvider.defaultProvider.`override`(OS.UNIX,
      pathToRedisExecutable)
    redisServer = new RedisServer(customProvider, redisPort)
    redisServer.start()

    jedis = new Jedis(redisHost, redisPort)
  }

  after {
    redisServer.stop()
  }

  "redisServer" should "works well" in {
    redisServer shouldNot be (null)
    jedis shouldNot be (null)
  }

  "redisServer" should "have correct output" in {
    xgroupCreate()
    xstreamWrite(inputXStream, "test")
    readRedis()
    hashmapWrite(inputHash)
    readRedis()
  }

  def xgroupCreate() : Unit = {
    println("Create Group <xstream>\n")
    try {
      jedis.xgroupCreate("test",
        "xstream", new StreamEntryID(0, 0), true)
    } catch {
      case e: Exception =>
        logger.info(s"xgroupCreate raise [$e], " +
          s"will not create new group.")
    }
  }

  def readRedis() : Unit = {
    println("Read group <xstream>")
    val response = jedis.xreadGroup(
      "xstream",
      "fake consumer",
      10,
      10,
      false,
      new SimpleEntry("test", StreamEntryID.UNRECEIVED_ENTRY))
    println(response)

    println("Read all keys")
    val params = new ScanParams
    params.`match`("*")

    // Use "0" to do a full iteration of the collection.
    val scanResult = jedis.scan("0", params)
    val keys = scanResult.getResult
    println(keys)
  }

  def hashmapWrite(value: List[(String, String)]) : Unit = {
    println("Write Hash Map to Redis")
    val ppl = jedis.pipelined()
    var cnt = 0
    value.foreach(v => {
      RedisUtils.writeHashMap(ppl, v._1, v._2, "test")
      if (v._2 != "NaN") {
        cnt += 1
      }
    })
    ppl.sync()
    logger.info(s"${cnt} valid records written to redis")
  }

  def xstreamWrite(hash: util.Map[String, String], streamID: String) : Unit = {
    println(s"Write to Redis stream ${streamID}")
    val ppl = jedis.pipelined()
    ppl.xadd(streamID, StreamEntryID.NEW_ENTRY, hash)
    ppl.sync()
    logger.info(s"${hash.size()} valid records written to redis")
  }

}

object RedisIOTest {
  // initialize the parser
  case class Config(path: String = null)
  val parser = new OptionParser[Config]("RedisIO test Usage") {
    opt[String]('p', "path")
      .text("Path to Redis Server Executable")
      .action((x, params) => params.copy(path = x))
      .required()
  }

  def main(args: Array[String]): Unit = {
    val arg = parser.parse(args, Config()).head
    val path = arg.path

    val redisIOMockInstance = new RedisIOSpec(path)
    redisIOMockInstance.execute()
  }

}


