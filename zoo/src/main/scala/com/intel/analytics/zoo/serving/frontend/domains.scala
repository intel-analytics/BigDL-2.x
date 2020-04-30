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

package com.intel.analytics.zoo.serving.frontend

import com.codahale.metrics.Timer

case class ServingTimerMetrics(
    name: String,
    count: Long,
    meanRate: Double,
    min: Long,
    max: Long,
    mean: Double,
    median: Double,
    stdDev: Double,
    _75thPercentile: Double,
    _95thPercentile: Double,
    _98thPercentile: Double,
    _99thPercentile: Double,
    _999thPercentile: Double
)

object ServingTimerMetrics {
  def apply(name: String, timer: Timer): ServingTimerMetrics =
    ServingTimerMetrics(
      name,
      timer.getCount,
      timer.getMeanRate,
      timer.getSnapshot.getMin/1000000,
      timer.getSnapshot.getMax/1000000,
      timer.getSnapshot.getMean/1000000,
      timer.getSnapshot.getMedian/1000000,
      timer.getSnapshot.getStdDev/1000000,
      timer.getSnapshot.get75thPercentile()/1000000,
      timer.getSnapshot.get95thPercentile()/1000000,
      timer.getSnapshot.get98thPercentile()/1000000,
      timer.getSnapshot.get99thPercentile()/1000000,
      timer.getSnapshot.get999thPercentile()/1000000
    )
}
