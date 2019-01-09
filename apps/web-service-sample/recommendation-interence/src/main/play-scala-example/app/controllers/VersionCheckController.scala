package controllers

import java.text.SimpleDateFormat

import javax.inject._
import models.LoadModel
import play.api.libs.json._
import play.api.mvc._

/**
  * This controller creates an `Action` to handle healthcheck.
  */

@Singleton
class VersionCheckController @Inject()(cc: ControllerComponents)
  extends AbstractController(cc) with LoadModel {

  /**
    * Create an Action to return @VersionCheckController status.
    * The configuration in the `routes` file means that this method
    * will be called when the application receives a `GET` request with
    * a path of `/VersionCheck`.
    */

  def index = Action {
    val df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS")
    val modelEpoch = params.recModelVersion.get
    val modelVersion = df.format(modelEpoch)
    val skuIndexerModelEpoch = params.skuIndexerModelVersion.get
    val skuIndexerModelVersion = df.format(skuIndexerModelEpoch)

    Ok(
      Json.obj(
        "status" -> "ok",
        "recModelVersion" -> modelVersion,
        "skuIndexerModelVersion" -> skuIndexerModelVersion
      )
    )
  }
}
