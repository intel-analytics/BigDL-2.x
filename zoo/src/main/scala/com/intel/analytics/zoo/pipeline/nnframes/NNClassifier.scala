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

package com.intel.analytics.zoo.pipeline.nnframes

import com.intel.analytics.bigdl.dataset.{Sample, Transformer}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.{Criterion, Module}
import com.intel.analytics.zoo.feature.common._
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import com.intel.analytics.zoo.pipeline.nnframes.NNModel.NNModelWriter
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostHelper,
XGBoostRegressionModel, XGBoostRegressor, XGBoostClassifier}
import org.apache.spark.ml.DefaultParamsWriterWrapper
import org.apache.spark.ml.adapter.SchemaUtils
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.param.{DoubleParam, ParamMap}
import org.apache.spark.ml.util.{Identifiable, MLReadable, MLReader}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types._
import org.json4s.DefaultFormats

import scala.reflect.ClassTag

/**
 * [[NNClassifier]] is a specialized [[NNEstimator]] that simplifies the data format for
 * classification tasks. It explicitly supports label column of DoubleType.
 * and the fitted [[NNClassifierModel]] will have the prediction column of DoubleType.
 *
 * @param model BigDL module to be optimized
 * @param criterion  BigDL criterion method
 */
class NNClassifier[T: ClassTag] private[zoo]  (
    @transient override val model: Module[T],
    override val criterion : Criterion[T],
    override val uid: String = Identifiable.randomUID("nnClassifier")
  )(implicit ev: TensorNumeric[T])
  extends NNEstimator[T](model, criterion) {

  override protected def wrapBigDLModel(m: Module[T]): NNClassifierModel[T] = {
    val classifierModel = new NNClassifierModel[T](m)
    val originBatchsize = classifierModel.getBatchSize
    copyValues(classifierModel.setParent(this)).setBatchSize(originBatchsize)
    val clonedTransformer = ToTuple() -> $(samplePreprocessing)
      .asInstanceOf[Preprocessing[(Any, Option[Any]), Sample[T]]].clonePreprocessing()
    classifierModel.setSamplePreprocessing(clonedTransformer)
  }

  override def transformSchema(schema : StructType): StructType = {
    validateParams(schema)
    SchemaUtils.appendColumn(schema, $(predictionCol), DoubleType)
  }

  override def copy(extra: ParamMap): NNClassifier[T] = {
    val newClassifier = new NNClassifier[T](
      model.cloneModule(),
      criterion.cloneCriterion(),
      this.uid
    )

    val copied = copyValues(
      newClassifier,
      extra)

    // optimMethod has internal states like steps and epochs, and
    // cannot be shared between estimators
    copied.setOptimMethod(this.getOptimMethod.clone())

    if (this.validationTrigger.isDefined) {
      copied.setValidation(
        validationTrigger.get, validationDF, validationMethods.clone(), validationBatchSize)
    }
    copied
  }
}

object NNClassifier {

  /**
   * Construct a [[NNClassifier]] with default Preprocessing, SeqToTensor
   *
   * @param model BigDL module to be optimized
   * @param criterion  BigDL criterion method
   */
  def apply[T: ClassTag](
      model: Module[T],
      criterion: Criterion[T]
    )(implicit ev: TensorNumeric[T]): NNClassifier[T] = {
    new NNClassifier(model, criterion)
        .setSamplePreprocessing(FeatureLabelPreprocessing(SeqToTensor(), ScalarToTensor()))
  }

  /**
   * Construct a [[NNClassifier]] with a feature size. The constructor is useful
   * when the feature column contains the following data types:
   * Float, Double, Int, Array[Float], Array[Double], Array[Int] and MLlib Vector. The feature
   * data are converted to Tensors with the specified sizes before sending to the model.
   *
   * @param model BigDL module to be optimized
   * @param criterion  BigDL criterion method
   * @param featureSize The size (Tensor dimensions) of the feature data. e.g. an image may be with
   *                    width * height = 28 * 28, featureSize = Array(28, 28).
   */
  def apply[T: ClassTag](
      model: Module[T],
      criterion: Criterion[T],
      featureSize: Array[Int]
    )(implicit ev: TensorNumeric[T]): NNClassifier[T] = {
    new NNClassifier(model, criterion)
        .setSamplePreprocessing(
          FeatureLabelPreprocessing(SeqToTensor(featureSize), ScalarToTensor()))
  }


  /**
   * Construct a [[NNClassifier]] with multiple input sizes. The constructor is useful
   * when the feature column and label column contains the following data types:
   * Float, Double, Int, Array[Float], Array[Double], Array[Int] and MLlib Vector. The feature
   * data are converted to Tensors with the specified sizes before sending to the model.
   *
   * This API is used for multi-input model, where user need to specify the tensor sizes for
   * each of the model input.
   *
   * @param model module to be optimized
   * @param criterion  criterion method
   * @param featureSize The sizes (Tensor dimensions) of the feature data.
   */
  def apply[T: ClassTag](
      model: Module[T],
      criterion: Criterion[T],
      featureSize : Array[Array[Int]]
    )(implicit ev: TensorNumeric[T]): NNClassifier[T] = {
    new NNClassifier(model, criterion)
      .setSamplePreprocessing(FeatureLabelPreprocessing(
        SeqToMultipleTensors(featureSize), ScalarToTensor()
      )
    )
  }

  /**
   * Construct a [[NNClassifier]] with a feature Preprocessing.
   *
   * @param model BigDL module to be optimized
   * @param criterion  BigDL criterion method
   * @param featurePreprocessing Preprocessing[F, Tensor[T] ].
   */
  def apply[F, T: ClassTag](
      model: Module[T],
      criterion: Criterion[T],
      featurePreprocessing: Preprocessing[F, Tensor[T]]
    )(implicit ev: TensorNumeric[T]): NNClassifier[T] = {
    new NNClassifier(model, criterion)
        .setSamplePreprocessing(
          FeatureLabelPreprocessing(featurePreprocessing, ScalarToTensor()))
  }
}

/**
 * [[NNClassifierModel]] is a specialized [[NNModel]] for classification tasks.
 * The prediction column will have the data type of Double.
 *
 * @param model trained BigDL models to use in prediction.
 */
class NNClassifierModel[T: ClassTag] private[zoo] (
    @transient override val model: Module[T],
    override val uid: String = Identifiable.randomUID("nnClassifierModel")
  )(implicit ev: TensorNumeric[T]) extends NNModel[T](model) {

  /**
   * Param for threshold in binary classification prediction.
   *
   * The threshold applies to the raw output of the model. If the output is greater than
   * threshold, then predict 1, else 0. A high threshold encourages the model to predict 0
   * more often; a low threshold encourages the model to predict 1 more often.
   *
   * Note: the param is different from the one in Spark ProbabilisticClassifier which is compared
   * against estimated probability.
   *
   * Default is 0.5.
   */
  final val threshold = new DoubleParam(this, "threshold", "threshold in binary" +
    " classification prediction")

  def getThreshold: Double = $(threshold)

  def setThreshold(value: Double): this.type = {
    set(threshold, value)
  }
  setDefault(threshold, 0.5)

  protected override def outputToPrediction(output: Tensor[T]): Any = {
    if (output.size().deep == Array(1).deep) {
      val raw = ev.toType[Double](output.toArray().head)
      if (raw > 0.5) 1.0 else 0.0
    } else {
      ev.toType[Double](output.max(1)._2.valueAt(1))
    }
  }

  override def transformSchema(schema : StructType): StructType = {
    SchemaUtils.appendColumn(schema, $(predictionCol), DoubleType)
  }

  override def copy(extra: ParamMap): NNClassifierModel[T] = {
    val copied = new NNClassifierModel(model.cloneModule(), uid).setParent(parent)
    copyValues(copied, extra).asInstanceOf[NNClassifierModel[T]]
  }
}

object NNClassifierModel extends MLReadable[NNClassifierModel[_]] {

  /**
   * Construct a [[NNClassifierModel]] with default Preprocessing, SeqToTensor
   *
   * @param model BigDL module to be optimized
   */
  def apply[T: ClassTag](
      model: Module[T]
    )(implicit ev: TensorNumeric[T]): NNClassifierModel[T] = {
    new NNClassifierModel(model)
      .setSamplePreprocessing(SeqToTensor() -> TensorToSample())
  }

  /**
   * Construct a [[NNClassifierModel]] with a feature size. The constructor is useful
   * when the feature column contains the following data types:
   * Float, Double, Int, Array[Float], Array[Double], Array[Int] and MLlib Vector. The feature
   * data are converted to Tensors with the specified sizes before sending to the model.
   *
   * @param model BigDL module to be optimized
   * @param featureSize The size (Tensor dimensions) of the feature data. e.g. an image may be with
   *                    width * height = 28 * 28, featureSize = Array(28, 28).
   */
  def apply[T: ClassTag](
      model: Module[T],
      featureSize : Array[Int]
    )(implicit ev: TensorNumeric[T]): NNClassifierModel[T] = {
    new NNClassifierModel(model)
      .setSamplePreprocessing(SeqToTensor(featureSize) -> TensorToSample())
  }

  /**
   * Construct a [[NNClassifierModel]] with sizes of multiple model inputs. The constructor is
   * useful when the feature column contains the following data types:
   * Float, Double, Int, Array[Float], Array[Double], Array[Int] and MLlib Vector. The feature
   * data are converted to Tensors with the specified sizes before sending to the model.
   *
   * This API is used for multi-input model, where user need to specify the tensor sizes for
   * each of the model input.
   *
   * @param model model to be used, which should be a multi-input model.
   * @param featureSize The sizes (Tensor dimensions) of the feature data.
   */
  def apply[T: ClassTag](
      model: Module[T],
      featureSize : Array[Array[Int]]
    )(implicit ev: TensorNumeric[T]): NNClassifierModel[T] = {
    new NNClassifierModel(model)
      .setSamplePreprocessing(SeqToMultipleTensors(featureSize) -> MultiTensorsToSample())
  }

  /**
   * Construct a [[NNClassifierModel]] with a feature Preprocessing.
   *
   * @param model BigDL module to be optimized
   * @param featurePreprocessing Preprocessing[F, Tensor[T] ].
   */
  def apply[F, T: ClassTag](
      model: Module[T],
      featurePreprocessing: Preprocessing[F, Tensor[T]]
    )(implicit ev: TensorNumeric[T]): NNClassifierModel[T] = {
    new NNClassifierModel(model).setSamplePreprocessing(featurePreprocessing -> TensorToSample())
  }

  private[nnframes] class NNClassifierModelReader() extends MLReader[NNClassifierModel[_]] {
    import scala.language.existentials
    implicit val format: DefaultFormats.type = DefaultFormats
    override def load(path: String): NNClassifierModel[_] = {
      val (meta, model, typeTag, feaTran) = NNModel.getMetaAndModel(path, sc)
      val nnModel = typeTag match {
        case "TensorDouble" =>
          new NNClassifierModel[Double](model.asInstanceOf[Module[Double]], meta.uid)
            .setSamplePreprocessing(feaTran.asInstanceOf[Preprocessing[Any, Sample[Double]]])
        case "TensorFloat" =>
          new NNClassifierModel[Float](model.asInstanceOf[Module[Float]], meta.uid)
            .setSamplePreprocessing(feaTran.asInstanceOf[Preprocessing[Any, Sample[Float]]])
        case _ =>
          throw new Exception("Only support float and double for now")
      }

      DefaultParamsWriterWrapper.getAndSetParams(nnModel, meta)
      nnModel
    }
  }

  class NNClassifierModelWriter[T: ClassTag](
      instance: NNClassifierModel[T])(implicit ev: TensorNumeric[T])
    extends NNModelWriter[T](instance)

  override def read: MLReader[NNClassifierModel[_]] = {
    new NNClassifierModel.NNClassifierModelReader
  }
}

class XGBClassifier () {
  private val model = new XGBoostClassifier()
  model.setNthread(EngineRef.getCoreNumber())
  model.setMaxBins(256)
  def setFeaturesCol(featuresColName: String): this.type = {
    model.setFeaturesCol(featuresColName)
    this
  }

  def fit(df: DataFrame): XGBClassifierModel = {
    df.repartition(EngineRef.getNodeNumber())
    val xgbmodel = model.fit(df)
    new XGBClassifierModel(xgbmodel)
  }

  def setNthread(value: Int): this.type = {
    model.setNthread(value)
    this
  }

  def setNumRound(value: Int): this.type = {
    model.setNumRound(value)
    this
  }

  def setNumWorkers(value: Int): this.type = {
    model.setNumWorkers(value)
    this
  }

  def setEta(value: Int): this.type = {
    model.setEta(value)
    this
  }

  def setGamma(value: Int): this.type = {
    model.setGamma(value)
    this
  }

  def setMaxDepth(value: Int): this.type = {
    model.setMaxDepth(value)
    this
  }

  def setMissing(value: Float): this.type = {
    model.setMissing(value)
    this
  }

}
/**
 * [[XGBClassifierModel]] is a trained XGBoost classification model.
 * The prediction column will have the prediction results.
 *
 * @param model trained XGBoostClassificationModel to use in prediction.
 */
class XGBClassifierModel private[zoo](
   val model: XGBoostClassificationModel) {
  private var featuresCols: Array[String] = null
  private var predictionCol: String = null

  def setFeaturesCol(featuresColName: Array[String]): this.type = {
    require(featuresColName.length > 1, "Please set a valid feature columns")
    featuresCols = featuresColName
    this
  }

  def setPredictionCol(value: String): this.type = {
    predictionCol = value
    this
  }

  def setInferBatchSize(value: Int): this.type = {
    model.setInferBatchSize(value)
    this
  }

  def transform(dataset: DataFrame): DataFrame = {
    require(featuresCols!=None, "Please set feature columns before transform")
    val featureVectorAssembler = new VectorAssembler()
      .setInputCols(featuresCols)
      .setOutputCol("featureAssembledVector")
    val assembledDF = featureVectorAssembler.transform(dataset)
    import org.apache.spark.sql.functions.{col, udf}
    import org.apache.spark.ml.linalg.Vector
    val asDense = udf((v: Vector) => v.toDense)
    val xgbInput = assembledDF.withColumn("DenseFeatures", asDense(col("featureAssembledVector")))
    model.setFeaturesCol("DenseFeatures")
    var output = model.transform(xgbInput).drop("DenseFeatures", "featureAssembledVector")
    if(predictionCol != null) {
      output = output.withColumnRenamed("prediction", predictionCol)
    }
    output
  }
}

object XGBClassifierModel {
  def load(path: String, numClass: Int): XGBClassifierModel = {
    new XGBClassifierModel(XGBoostHelper.load(path, numClass))
  }
}

/**
 * [[XGBRegressor]] xgboost wrapper of XGBRegressor.
 */
class XGBRegressor () {

  private val model = new XGBoostRegressor()
  model.setNthread(EngineRef.getCoreNumber())
  model.setMaxBins(256)

  def setLabelCol(labelColName : String) : this.type = {
    model.setLabelCol(labelColName)
    this
  }

  def setFeaturesCol(featuresColName: String): this.type = {
    model.setFeaturesCol(featuresColName)
    this
  }

  def fit(df: DataFrame): XGBRegressorModel = {
    df.repartition(EngineRef.getNodeNumber())
    val xgbModel = model.fit(df)
    new XGBRegressorModel(xgbModel)
  }

  def setNumRound(value: Int): this.type = {
    model.setNumRound(value)
    this
  }

  def setNumWorkers(value: Int): this.type = {
    model.setNumWorkers(value)
    this
  }

  def setNthread(value: Int): this.type = {
    model.setNthread(value)
    this
  }

  def setSilent(value: Int): this.type = {
    model.setSilent(value)
    this
  }

  def setMissing(value: Float): this.type = {
    model.setMissing(value)
    this
  }

  def setCheckpointPath(value: String): this.type = {
    model.setCheckpointPath(value)
    this
  }

  def setCheckpointInterval(value: Int): this.type = {
    model.setCheckpointInterval(value)
    this
  }

  def setSeed(value: Long): this.type = {
    model.setSeed(value)
    this
  }

  def setEta(value: Double): this.type = {
    model.setEta(value)
    this
  }

  def setGamma(value: Double): this.type = {
    model.setGamma(value)
    this
  }

  def setMaxDepth(value: Int): this.type = {
    model.setMaxDepth(value)
    this
  }

  def setMinChildWeight(value: Double): this.type = {
    model.setMinChildWeight(value)
    this
  }

  def setMaxDeltaStep(value: Double): this.type = {
    model.setMaxDeltaStep(value)
    this
  }

  def setColsampleBytree(value: Double): this.type = {
    model.setColsampleBytree(value)
    this
  }

  def setColsampleBylevel(value: Double): this.type = {
    model.setColsampleBylevel(value)
    this
  }

  def setLambda(value: Double): this.type = {
    model.setLambda(value)
    this
  }

  def setAlpha(value: Double): this.type = {
    model.setAlpha(value)
    this
  }

  def setTreeMethod(value: String): this.type = {
    model.setTreeMethod(value)
    this
  }

  def setGrowPolicy(value: String): this.type = {
    model.setGrowPolicy(value)
    this
  }

  def setMaxBins(value: Int): this.type = {
    model.setMaxBins(value)
    this
  }

  def setMaxLeaves(value: Int): this.type = {
    model.setMaxLeaves(value)
    this
  }

  def setSketchEps(value: Double): this.type = {
    model.setSketchEps(value)
    this
  }

  def setScalePosWeight(value: Double): this.type = {
    model.setScalePosWeight(value)
    this
  }

  def setSampleType(value: String): this.type = {
    model.setSampleType(value)
    this
  }

  def setNormalizeType(value: String): this.type = {
    model.setNormalizeType(value)
    this
  }

  def setRateDrop(value: Double): this.type = {
    model.setRateDrop(value)
    this
  }

  def setSkipDrop(value: Double): this.type = {
    model.setSkipDrop(value)
    this
  }

  def setLambdaBias(value: Double): this.type = {
    model.setLambdaBias(value)
    this
  }

  def setObjective(value: String): this.type = {
    model.setObjective(value)
    this
  }

  def setObjectiveType(value: String): this.type = {
    model.setObjectiveType(value)
    this
  }

  def setSubsample(value: Double): this.type = {
    model.setSubsample(value)
    this
  }

  def setBaseScore(value: Double): this.type = {
    model.setBaseScore(value)
    this
  }

  def setEvalMetric(value: String): this.type = {
    model.setEvalMetric(value)
    this
  }

  def setNumEarlyStoppingRounds(value: Int): this.type = {
    model.setNumEarlyStoppingRounds(value)
    this
  }

  def setMaximizeEvaluationMetrics(value: Boolean): this.type = {
    model.setMaximizeEvaluationMetrics(value)
    this
  }
}

/**
 * [[XGBRegressorModel]] xgboost wrapper of XGBRegressorModel.
 */
class XGBRegressorModel private[zoo](val model: XGBoostRegressionModel) {
  var predictionCol: String = null
  var featuresCol: String = "features"
  var featurearray: Array[String] = Array("features")
  def setPredictionCol(value: String): this.type = {
    predictionCol = value
    this
  }

  def setInferBatchSize(value: Int): this.type = {
    model.setInferBatchSize(value)
    this
  }

  def setFeaturesCol(value: String): this.type = {
    model.setFeaturesCol(value)
    featuresCol = value
    this
  }

  def transform(dataset: DataFrame): DataFrame = {
    val featureVectorAssembler = new VectorAssembler()
      .setInputCols(featurearray)
      .setOutputCol("featureAssembledVector")
    val assembledDF = featureVectorAssembler.transform(dataset)
    import org.apache.spark.sql.functions.{col, udf}
    import org.apache.spark.ml.linalg.Vector
    val asDense = udf((v: Vector) => v.toDense)
    val xgbInput = assembledDF.withColumn("DenseFeatures", asDense(col("featureAssembledVector")))
    model.setFeaturesCol("DenseFeatures")
    var output = model.transform(xgbInput).drop("DenseFeatures", "featureAssembledVector")
    if(predictionCol != null) {
      output = output.withColumnRenamed("prediction", predictionCol)
    }
    output
  }

  def save(path: String): Unit = {
    model.write.overwrite().save(path)
  }
}

object XGBRegressorModel {
  /**
   * Load pretrained Zoo XGBRegressorModel.
   */
  def load(path: String): XGBRegressorModel = {
    new XGBRegressorModel(XGBoostRegressionModel.load(path))
  }

  /**
   * Load pretrained xgboost XGBoostRegressionModel.
   */
  def loadFromXGB(path: String): XGBRegressorModel = {
    new XGBRegressorModel(XGBoostHelper.load(path))
  }
}
