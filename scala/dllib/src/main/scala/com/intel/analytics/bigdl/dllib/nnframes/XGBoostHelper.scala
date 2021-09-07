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

package ml.dmlc.xgboost4j.scala.spark

object XGBoostHelper {
  def load(path: String, numClass: Int): XGBoostClassificationModel = {
    import ml.dmlc.xgboost4j.scala.XGBoost
    val _booster = XGBoost.loadModel(path)
    new XGBoostClassificationModel("XGBClassifierModel", numClass, _booster)
  }

  def load(path: String): XGBoostRegressionModel = {
    import ml.dmlc.xgboost4j.scala.XGBoost
    val _booster = XGBoost.loadModel(path)
    new XGBoostRegressionModel("XGBRegressorModel", _booster)
  }
}
