package controllers

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.LocalPredictor
import com.intel.analytics.bigdl.tensor.Tensor
import javax.inject._
import models.LoadModel
import play.api.libs.json._
import play.api.mvc._
import utilities.Helper._

import scala.collection.immutable.Map

/**
  * This controller creates an `Action` to handle recommendation prediction.
  */

@Singleton
class ModelController @Inject()(cc: ControllerComponents) extends AbstractController(cc) with LoadModel {

  /**
    * Create an Action to return @ModelController status.
    * The configuration in the `routes` file means that this method
    * will be called when the application receives a `POST` request with
    * a path of `/recModel`.
    */

  def recModel: Action[JsValue] = Action(parse.json) { request =>
    try {
      val requestJson = request.body.toString()
      val requestMap = mapper.readValue(requestJson, classOf[Map[String, Any]])
      val sku = requestMap("SKU_NUM").asInstanceOf[List[String]]
      sku.foreach(
        println(_)
      )
      val skuIndex = sku.map(x => leapTransform(
        requestString = x, inputCol = "SKU_NUM", outputCol = "SKU_INDEX", transformer = params.skuIndexerModel.get, mapper = mapper
      ).toFloat).reverse.padTo(10, 0f).reverse

      val inputSample = Array(Sample(Tensor(skuIndex.toArray, Array(10))))

      val predict = params.recModel match {
        case Some(_) =>
          val predict = LocalPredictor(params.recModel.get).predict(inputSample)
            .map { x =>
              val _output = x.toTensor[Float]
              val indices = _output.topk(11, 1, false)
              val predict = (1 to 20).map{
                i =>
                  val predict = indices._2.valueAt(i).toInt - 1
                  val probability = Math.exp(_output.valueAt(predict).toDouble)
                  Map(s"predict$i" -> revertStringIndex(predict.toString), s"probability$i" -> probability)
              }
              predict.toArray
            }.head
          predict
        case None => Map("predict" -> "N/A", "probability" -> 0.0)
      }

      val predictionJson = mapper.writeValueAsString(predict)

      Ok(Json.parse(predictionJson.toString))
    }

    catch{
      case _:Exception => BadRequest("Nah nah nah nah nah...this request contains bad characters...")
    }
  }
}
