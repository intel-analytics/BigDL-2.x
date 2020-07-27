/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.serving.utils

import java.io.{FileOutputStream, ObjectOutputStream, PrintWriter}
import java.util.concurrent.Executors

import org.apache.spark.sql.streaming._
import java.util.concurrent.TimeUnit.SECONDS

import org.apache.flink.core.execution.JobClient
import org.apache.spark.streaming.StreamingContext

object ClusterServingManager {
  /**
   * A Runnable that check the stop signal of serving
   * If signal detected, stop the streaming query
   * @param helper ClusterServing Helper
   * @param query StreamingQeury
   * @return
   */
  def queryTerminator(helper: ClusterServingHelper,
                      query: StreamingContext): Runnable = new Runnable {
    override def run(): Unit = {
      if (helper.checkStop()) {
        query.stop()
      }
    }
  }

  /**
   * Run the Terminator to periodically check
   * whether to end the streaming depending on the stop signal
   * @param helper ClusterServing Helper
   * @param query StreamingQeury
   * @return
   */
  def listenTermination(helper: ClusterServingHelper,
                        query: StreamingContext): Unit = {
    Executors.newSingleThreadScheduledExecutor.scheduleWithFixedDelay(
      queryTerminator(helper, query), 1, 1, SECONDS
    )

  }
  def writeObjectToFile(cli: JobClient): Unit = {
    try {
//      val fileOut = new FileOutputStream("/tmp/cluster-serving-job-id")
//      val objectOut = new ObjectOutputStream(fileOut)
//      objectOut.writeObject(obj)
//      objectOut.close()
      new PrintWriter("/tmp/cluster-serving-job-id") {
        write(cli.getJobID.toHexString)
        close
      }
      println("Cluster Serving Flink job id written to file.")
    }
    catch {
      case e: Exception =>
        e.printStackTrace()
        println("Failed to write job id written to file. You may not manager job by id now.")
    }
  }
}
