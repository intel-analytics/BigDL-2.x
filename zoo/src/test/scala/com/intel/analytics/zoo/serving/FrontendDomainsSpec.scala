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

package com.intel.analytics.zoo.serving

import java.util.concurrent.{CountDownLatch, Executors}

import com.google.common.util.concurrent.RateLimiter
import com.intel.analytics.zoo.serving.http._
import com.netflix.concurrency.limits.executors.BlockingAdaptiveExecutor
import com.netflix.concurrency.limits.limiter.SimpleLimiter
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class FrontendDomainsSpec extends FlatSpec with Matchers with BeforeAndAfter with Supportive {

  "ServingError" should "serialized as json" in {
    val message = "contentType not supported"
    val error = ServingError(message)
    error.toString should include (s""""error" : "$message"""")
  }

  "Feature" should "serialized and deserialized as json" in {
    val image1 = new ImageFeature("aW1hZ2UgYnl0ZXM=")
    val image2 = new ImageFeature("YXdlc29tZSBpbWFnZSBieXRlcw==")
    val instance1 = Map("image" -> image1, "caption" -> "seaside")
    val instance2 = Map("image" -> image2, "caption" -> "montians")
    val inputs = Instances(instance1, instance2)
    val json = timing("serialize")() {
      JsonUtil.toJson(inputs)
    }
    val obj = timing("deserialize")() {
      JsonUtil.fromJson(classOf[Instances], json)
    }
    println(obj)
  }

  "RateLimiter" should "work" in {
    val rateLimiter = RateLimiter.create(5)
    val executorService = Executors.newFixedThreadPool(5)
    val nTasks = 100
    val countDownLatch = new CountDownLatch(nTasks)
    val start = System.currentTimeMillis()
    List.range(0, nTasks).map(i => {
      executorService.submit(new Runnable {
        override def run(): Unit = {
          rateLimiter.acquire(1)
          Thread.sleep(1000)
          println(Thread.currentThread().getName() + " gets job " + i + " done")
          countDownLatch.countDown()
        }
      })
    })
    executorService.shutdown();
    countDownLatch.await();
    val end = System.currentTimeMillis();
    println("10 jobs gets done by 5 threads concurrently in " + (end - start) + " milliseconds")

    val executor = new BlockingAdaptiveExecutor(
      SimpleLimiter.newBuilder()
        .build())

    val countDownLatch2 = new CountDownLatch(nTasks)
    val start2 = System.currentTimeMillis()
    List.range(0, nTasks).map(i => {
      executor.execute(new Runnable {
        override def run(): Unit = {
          Thread.sleep(1000)
          println(Thread.currentThread().getName() + " gets job " + i + " done")
          countDownLatch2.countDown()
        }
      })
    })
    countDownLatch2.await();
    val end2 = System.currentTimeMillis();
    println("10 jobs gets done by 5 threads concurrently in " + (end2 - start2) + " milliseconds")



  }

}
