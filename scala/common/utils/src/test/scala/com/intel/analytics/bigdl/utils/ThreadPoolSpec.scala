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

package com.intel.analytics.bigdl.utils

import com.intel.analytics.bigdl.mkl.MKL
import com.intel.analytics.bigdl.mkl.hardware.Affinity
import org.scalatest.{FlatSpec, Matchers}

class ThreadPoolSpec extends FlatSpec with Matchers {

  "mkldnn backend" should "create omp threads and bind correctly" in {
    com.intel.analytics.bigdl.mkl.MklDnn.isLoaded
    val poolSize = 1
    val ompSize = 4

    val threadPool = new ThreadPool(poolSize)
    // backup the affinities
    val affinities = threadPool.invokeAndWait2( (0 until poolSize).map(i =>
      () => {
        Affinity.getAffinity()
      })).map(_.get()).toArray

    threadPool.setMKLThreadOfMklDnnBackend(ompSize)

    threadPool.invokeAndWait2( (0 until poolSize).map( i =>
      () => {
        Affinity.getAffinity.length should be (1)
        Affinity.getAffinity.head should be (0)
      }))

    // set back the affinities
    threadPool.invokeAndWait2( (0 until poolSize).map( i => () => {
      Affinity.setAffinity(affinities(i))
    }))

    threadPool.invokeAndWait2( (0 until poolSize).map( i =>
      () => {
        Affinity.getAffinity.zipWithIndex.foreach(ai => ai._1 should be (ai._2))
      }))

  }

  "mkldnn thread affinity binding" should "not influence other threads" in {
    val poolSize = 1
    val ompSize = 4

    val threadPool = new ThreadPool(poolSize)
    threadPool.setMKLThreadOfMklDnnBackend(ompSize)

    threadPool.invokeAndWait2( (0 until poolSize).map( i =>
      () => {
        Affinity.getAffinity.length should be (1)
        Affinity.getAffinity.head should be (0)
      }))

    val threadPool2 = new ThreadPool(poolSize)
    threadPool2.invokeAndWait2( (0 until poolSize).map(i => () => {
      println(Affinity.getAffinity.mkString("\t"))
      Affinity.getAffinity.length should not be (1)
    }))
  }
}
