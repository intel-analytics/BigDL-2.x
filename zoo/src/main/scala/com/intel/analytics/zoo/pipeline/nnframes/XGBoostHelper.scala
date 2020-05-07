package ml.dmlc.xgboost4j.scala.spark

object XGBoostHelper {
  def load(path: String, numClass: Int) = {
    import ml.dmlc.xgboost4j.scala.XGBoost
    val _booster = XGBoost.loadModel(path)
    new XGBoostClassificationModel("XGBClassifierModel", numClass, _booster)
  }
}
