package com.intel.analytics.zoo.serving.utils

import java.nio.file.{FileSystems, Path}
import java.nio.file.StandardWatchEventKinds._

object Watcher {
  def checkUpdate(path: Path): Unit = {
    val watcher = FileSystems.getDefault.newWatchService()
    val key = path.register(watcher, ENTRY_CREATE, ENTRY_MODIFY)
  }
}
