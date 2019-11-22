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

package com.intel.analytics.zoo.pipeline.inference

import com.intel.analytics.zoo.tensorboard.Summary
import com.intel.analytics.zoo.tensorboard.FileWriter
import com.intel.analytics.zoo.tensorboard.FileReader



<<<<<<< HEAD
class InferenceSummary(
                        logDir: String,
                        appName: String) extends Summary(logDir, appName) {
=======
class InferenceSummary(logDir: String,
                       appName: String) extends Summary(logDir, appName) {
>>>>>>> 504c982fc1f7a3d2ed00681e8677b4b0e03faf5e
  protected val folder = s"$logDir/$appName/inference"
  override val writer = new FileWriter(folder)


  /**
   * ReadScalar by tag name. Optional tag name is based on ValidationMethod, "Loss",
   * "Top1Accuracy" or "Top5Accuracy".
   *
   * @param tag tag name.
   * @return an array of triple.
   */
  override def readScalar(tag: String): Array[(Long, Float, Double)] = {
    FileReader.readScalar(folder, tag)
  }
}
object InferenceSummary {
  def apply(logDir: String,
            appName: String): InferenceSummary = {
    new InferenceSummary(logDir, appName)
  }
}