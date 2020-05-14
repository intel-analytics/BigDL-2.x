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

import java.nio.file.{Files, Paths}

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path

object FileUtils {
  /**
   * Use hadoop utils to copy file from remote to local
   * @param src remote path, could be hdfs, s3
   * @param dst local path
   */
  def copyToLocal(src: String, dst: String): Unit = {
    val conf = new Configuration()

    val srcPath = new Path(src)
    val fs = srcPath.getFileSystem(conf)

    val dstPath = new Path(dst)
    fs.copyToLocalFile(srcPath, dstPath)
  }
  /**
   * Check stop signal, return true if signal detected
   * @return
   */
  def checkStop(): Boolean = {
    if (!Files.exists(Paths.get("running"))) {
      return true
    }
    return false

  }
}
