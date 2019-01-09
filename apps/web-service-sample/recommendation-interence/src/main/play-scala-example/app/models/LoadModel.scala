package models

import java.util.concurrent.TimeUnit

import akka.actor.{ActorSystem, Scheduler}
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule

import scala.concurrent.duration.Duration

trait LoadModel {

  var params = ModelParams(
    "./modelFiles/rnnModel",
    "./modelFiles/skuIndexer.zip",
    "./modelFiles/skuLookUp",
    scala.util.Properties.envOrElse("configEnvironmewnt", "dev")
  )

  val actorSystem = ActorSystem()
  val scheduler: Scheduler = actorSystem.scheduler
  private val task = new Runnable {
    def run(): Unit = {
      try {
//        ModelParams.downloadModel(params)
        ModelParams.refresh(params)
      }
      catch {
        case _: Exception => println("Model update has failed")
      }
    }
  }

  implicit val executor = actorSystem.dispatcher

  scheduler.schedule(
    initialDelay = Duration(5, TimeUnit.SECONDS),
    interval = Duration(5, TimeUnit.SECONDS),
    runnable = task)

  val mapper = new ObjectMapper
  mapper.registerModule(DefaultScalaModule)

}
