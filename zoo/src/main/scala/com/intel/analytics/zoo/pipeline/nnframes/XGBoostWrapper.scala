package ml.dmlc.xgboost4j.scala.spark

object XGBoostWrapper {
  def load(path: String, numClass: Int) = {
    import ml.dmlc.xgboost4j.scala.XGBoost
    val _booster = XGBoost.loadModel(path)
    new XGBoostClassificationModel("nnXGBClassifierModel", numClass, _booster)
  }
}
