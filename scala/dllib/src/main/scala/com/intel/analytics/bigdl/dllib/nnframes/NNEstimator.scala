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

import com.intel.analytics.bigdl.dataset.{Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.{Criterion, Module}
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.tensor.{Tensor, DoubleType => TensorDouble, FloatType => TensorFloat}
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleLoader
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import org.apache.spark.ml.adapter.{HasFeaturesCol, HasPredictionCol, SchemaUtils}
import org.apache.spark.ml.{DLEstimatorBase, DLTransformerBase, VectorCompatibility, DefaultParamsWriterWrapper}
import org.apache.spark.ml.param._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.ml.util._
import org.apache.spark.rdd.RDD
import org.json4s.{DefaultFormats, JObject}
import org.json4s.JsonDSL._

import scala.reflect.ClassTag

private[nnframes] trait HasBatchSize extends Params {

   final val batchSize: IntParam = new IntParam(this, "batchSize", "batchSize")

  def getBatchSize: Int = $(batchSize)
}

private[nnframes] trait TrainingParams[@specialized(Float, Double) T] extends Params {
  /**
   * When to stop the training, passed in a [[Trigger]]. E.g. Trigger.maxIterations
   */
  final val endWhen = new Param[Trigger](this, "endWhen", "Trigger to stop the training")

  def getEndWhen: Trigger = $(endWhen)

  /**
   * learning rate for the optimizer in the NNEstimator.
   * Default: 0.001
   */
  final val learningRate = new DoubleParam(
    this, "learningRate", "learningRate", ParamValidators.gt(0))

  def getLearningRate: Double = $(learningRate)

  /**
   * learning rate decay for each iteration.
   * Default: 0
   */
  final val learningRateDecay = new DoubleParam(this, "learningRateDecay", "learningRateDecay")

  def getLearningRateDecay: Double = $(learningRateDecay)

  /**
   * Number of max Epoch for the training, an epoch refers to a traverse over the training data
   * Default: 50
   */
  final val maxEpoch = new IntParam(this, "maxEpoch", "number of max Epoch", ParamValidators.gt(0))

  def getMaxEpoch: Int = $(maxEpoch)

  /**
   * optimization method to be used. BigDL supports many optimization methods like Adam,
   * SGD and LBFGS. Refer to package com.intel.analytics.bigdl.optim for all the options.
   * Default: SGD
   */
  final val optimMethod = new Param[OptimMethod[T]](this, "optimMethod", "optimMethod")

  def getOptimMethod: OptimMethod[T] = $(optimMethod)
}

/**
 * Common trait for NNEstimator and NNModel
 */
private[nnframes] trait NNParams[@specialized(Float, Double) T] extends HasFeaturesCol
  with HasPredictionCol with VectorCompatibility with HasBatchSize {

  setDefault(batchSize -> 1)

  /**
   * Validate if feature and label columns are of supported data types.
   * Default: 0
   */
  protected def validateDataType(schema: StructType, colName: String): Unit = {
    val dataTypes = Seq(
      new ArrayType(DoubleType, false),
      new ArrayType(DoubleType, true),
      new ArrayType(FloatType, false),
      new ArrayType(FloatType, true),
      DoubleType,
      FloatType,
      NNImageSchema.floatSchema
    ) ++ validVectorTypes

    // TODO use SchemaUtils.checkColumnTypes after convert to 2.0
    val actualDataType = schema(colName).dataType
    require(dataTypes.exists(actualDataType.equals),
      s"Column $colName must be of type equal to one of the following types: " +
        s"${dataTypes.mkString("[", ", ", "]")} but was actually of type $actualDataType.")
  }

  /**
   * Get conversion function to extract data from original DataFrame
   * Default: 0
   */
  protected def getConvertFunc(colType: DataType): (Row, Int) => Seq[AnyVal] = {
    colType match {
      case ArrayType(DoubleType, false) =>
        (row: Row, index: Int) => row.getSeq[Double](index)
      case ArrayType(DoubleType, true) =>
        (row: Row, index: Int) => row.getSeq[Double](index)
      case ArrayType(FloatType, false) =>
        (row: Row, index: Int) => row.getSeq[Float](index)
      case ArrayType(FloatType, true) =>
        (row: Row, index: Int) => row.getSeq[Float](index)
      case DoubleType =>
        (row: Row, index: Int) => Seq[Double](row.getDouble(index))
      case FloatType =>
        (row: Row, index: Int) => Seq[Float](row.getFloat(index))
      case NNImageSchema.floatSchema =>
        (row: Row, index: Int) => row.getAs[Row](index).getSeq[Float](5)
      case _ =>
        if (colType.typeName.contains("vector")) {
          (row: Row, index: Int) => getVectorSeq(row, colType, index)
        } else {
          throw new IllegalArgumentException(
            s"$colType is not a supported (Unexpected path).")
        }
    }
  }
}

/**
 * [[NNEstimator]] helps to train a BigDL Model with the Spark
 * ML Estimator/Transfomer pattern, thus Spark users can conveniently fit BigDL into Spark
 * ML pipeline. In Addition, it supports image schema
 *
 * [[NNEstimator]] supports feature and label data in the format of
 * Array[Double], Array[Float], org.apache.spark.mllib.linalg.{Vector, VectorUDT},
 * org.apache.spark.ml.linalg.{Vector, VectorUDT}, Double and Float.
 *
 * User should specify the feature data dimensions and label data dimensions via the constructor
 * parameters featureSize and labelSize respectively. Internally the feature and label data are
 * converted to BigDL tensors, to further train a BigDL model efficiently.
 *
 * For details usage, please refer to examples in package
 * com.intel.analytics.zoo.pipeline.example.nnframes
 *
 * @param model BigDL module to be optimized
 * @param criterion  BigDL criterion method
 * @param featureSize The size (Tensor dimensions) of the feature data. e.g. an image may be with
 *                    width * height = 28 * 28, featureSize = Array(28, 28).
 * @param labelSize The size (Tensor dimensions) of the label data.
 */
class NNEstimator[T: ClassTag](
    @transient val model: Module[T],
    val criterion : Criterion[T],
    val featureSize : Array[Int],
    val labelSize : Array[Int],
    override val uid: String = Identifiable.randomUID("nnEstimator")
  )(implicit ev: TensorNumeric[T])
  extends DLEstimatorBase[NNEstimator[T], NNModel[T]] with NNParams[T] with TrainingParams[T] {

  def setFeaturesCol(featuresColName: String): this.type = set(featuresCol, featuresColName)

  def setLabelCol(labelColName : String) : this.type = set(labelCol, labelColName)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def setBatchSize(value: Int): this.type = set(batchSize, value)

  def setEndWhen(trigger: Trigger): this.type = set(endWhen, trigger)

  def setLearningRate(value: Double): this.type = set(learningRate, value)
  setDefault(learningRate -> 1e-3)

  def setLearningRateDecay(value: Double): this.type = set(learningRateDecay, value)
  setDefault(learningRateDecay -> 0.0)

  def setMaxEpoch(value: Int): this.type = set(maxEpoch, value)
  setDefault(maxEpoch -> 50)

  def setOptimMethod(value: OptimMethod[T]): this.type = set(optimMethod, value)
  set(optimMethod, new SGD[T])

  @transient private var trainSummary: Option[TrainSummary] = None

  def getTrainSummary: Option[TrainSummary] = trainSummary

  /**
   * Statistics (LearningRate, Loss, Throughput, Parameters) collected during training for the
   * training data, which can be used for visualization via Tensorboard.
   * Use setTrainSummary to enable train logger. Then the log will be saved to
   * logDir/appName/train as specified by the parameters of TrainSummary.
   *
   * Default: Not enabled
   */
  def setTrainSummary(value: TrainSummary): this.type = {
    this.trainSummary = Some(value)
    this
  }

  @transient private var validationSummary: Option[ValidationSummary] = None

  /**
   * Statistics (LearningRate, Loss, Throughput, Parameters) collected during training for the
   * validation data if validation data is set, which can be used for visualization via
   * Tensorboard. Use setValidationSummary to enable validation logger. Then the log will be
   * saved to logDir/appName/ as specified by the parameters of validationSummary.
   *
   * Default: None
   */
  def getValidationSummary: Option[ValidationSummary] = validationSummary

  /**
   * Enable validation Summary
   */
  def setValidationSummary(value: ValidationSummary): this.type = {
    this.validationSummary = Some(value)
    this
  }

  @transient protected var validationTrigger: Option[Trigger] = None
  @transient protected var validationDF: DataFrame = _
  @transient protected var validationMethods: Array[ValidationMethod[T]] = _
  @transient protected var validationBatchSize: Int = 0
  /**
   * Set a validate evaluation during training
   *
   * @param trigger how often to evaluation validation set
   * @param validationDF validate data set
   * @param vMethods a set of validation method [[ValidationMethod]]
   * @param batchSize batch size for validation
   * @return this optimizer
   */
  def setValidation(trigger: Trigger, validationDF: DataFrame,
                    vMethods : Array[ValidationMethod[T]], batchSize: Int)
  : this.type = {
    this.validationTrigger = Some(trigger)
    this.validationDF = validationDF
    this.validationMethods = vMethods
    this.validationBatchSize = batchSize
    this
  }

  def getValidation: Option[(Trigger, DataFrame, Array[ValidationMethod[T]], Int)] = {
    if (validationTrigger.isDefined) {
      Some(validationTrigger.get, validationDF, validationMethods, validationBatchSize)
    }
    else {
      None
    }
  }

  protected def validateParams(schema : StructType): Unit = {
    validateDataType(schema, $(featuresCol))
    validateDataType(schema, $(labelCol))
    if(isSet(endWhen) && isSet(maxEpoch)) {
      throw new IllegalArgumentException(s"endWhen and maxEpoch cannot be both set")
    }
    if (validationTrigger.isEmpty && validationSummary.isDefined) {
      throw new IllegalArgumentException(
        s"validationSummary is only valid if validation data is set.")
    }
  }

  override def transformSchema(schema : StructType): StructType = {
    validateParams(schema)
    SchemaUtils.appendColumn(schema, $(predictionCol), ArrayType(DoubleType, false))
  }

  protected override def internalFit(dataFrame: DataFrame): NNModel[T] = {
    val localFeatureCol = $(featuresCol)
    val localLabelCol = $(labelCol)

    def getSamples(dataFrame: DataFrame): RDD[Sample[T]] = {
      val featureType = dataFrame.schema(localFeatureCol).dataType
      val featureColIndex = dataFrame.schema.fieldIndex(localFeatureCol)
      val labelType = dataFrame.schema(localLabelCol).dataType
      val labelColIndex = dataFrame.schema.fieldIndex(localLabelCol)

      val featureFunc = getConvertFunc(featureType)
      val labelFunc = getConvertFunc(labelType)

      val featureAndLabel: RDD[(Seq[AnyVal], Seq[AnyVal])] = dataFrame.rdd.map { row =>
        val features = featureFunc(row, featureColIndex)
        val labels = labelFunc(row, labelColIndex)
        (features, labels)
      }

      val samples = featureAndLabel.map { case (f, l) =>
        // convert feature and label data type to the same type with model
        // TODO: investigate to reduce memory consumption during conversion.
        val feature = f.head match {
          case dd: Double => f.asInstanceOf[Seq[Double]].map(ev.fromType(_))
          case ff: Float => f.asInstanceOf[Seq[Float]].map(ev.fromType(_))
        }
        val label = l.head match {
          case dd: Double => l.asInstanceOf[Seq[Double]].map(ev.fromType(_))
          case ff: Float => l.asInstanceOf[Seq[Float]].map(ev.fromType(_))
        }
        (feature, label)
      }.map { case (feature, label) =>
        Sample(Tensor(feature.toArray, featureSize), Tensor(label.toArray, labelSize))
      }
      samples
    }

    val trainingSamples = getSamples(dataFrame)
    val state = T("learningRate" -> $(learningRate), "learningRateDecay" -> $(learningRateDecay))
    val endTrigger = if (isSet(endWhen)) $(endWhen) else Trigger.maxEpoch($(maxEpoch))
    val optimizer = Optimizer(model, trainingSamples, criterion, $(batchSize))
      .setState(state)
      .setOptimMethod($(optimMethod))
      .setEndWhen(endTrigger)

    if (validationTrigger.isDefined) {
      val validationSamples = getSamples(validationDF)
      optimizer.setValidation(
        validationTrigger.get,
        validationSamples,
        validationMethods,
        validationBatchSize)
      if (this.validationSummary.isDefined) {
        optimizer.setValidationSummary(this.validationSummary.get)
      }
    }

    if (this.trainSummary.isDefined) {
      optimizer.setTrainSummary(this.trainSummary.get)
    }

    val optimizedModel = optimizer.optimize()
    wrapBigDLModel(optimizedModel, featureSize)
  }

  /**
   * sub classes can extend the method and return required model for different transform tasks
   */
  protected def wrapBigDLModel(m: Module[T], featureSize: Array[Int]): NNModel[T] = {
    val dlModel = new NNModel[T](m, featureSize)
    copyValues(dlModel.setParent(this))
  }

  /**
   * Return a deep copy for DLEstimator.
   * Note that trainSummary and validationSummary will not be copied to the new instance since
   * currently they are not thread-safe.
   */
  override def copy(extra: ParamMap): NNEstimator[T] = {
    val copied = copyValues(
      new NNEstimator(
        model.cloneModule(),
        criterion.cloneCriterion(),
        featureSize.clone(),
        labelSize.clone(),
        this.uid
      ),
      extra)

    if (this.validationTrigger.isDefined) {
      copied.setValidation(
        validationTrigger.get, validationDF, validationMethods.clone(), validationBatchSize)
    }
    copied
  }
}

/**
 * [[NNModel]] helps embed a BigDL model into a Spark Transformer, thus Spark users can
 * conveniently merge BigDL into Spark ML pipeline.
 * [[NNModel]] supports feature data in the format of
 * Array[Double], Array[Float], org.apache.spark.mllib.linalg.{Vector, VectorUDT},
 * org.apache.spark.ml.linalg.{Vector, VectorUDT}, Double and Float.
 * Internally [[NNModel]] use features column as storage of the feature data, and create
 * Tensors according to the constructor parameter featureSize.
 *
 * [[NNModel]] is compatible with both spark 1.5-plus and 2.0 by extending ML Transformer.
 * @param model trainned BigDL models to use in prediction.
 * @param featureSize The size (Tensor dimensions) of the feature data. (e.g. an image may be with
 * featureSize = 28 * 28).
 */
class NNModel[T: ClassTag](
    @transient val model: Module[T],
    var featureSize : Array[Int],
    override val uid: String = "DLModel")(implicit ev: TensorNumeric[T])
  extends DLTransformerBase[NNModel[T]] with NNParams[T]
    with HasBatchSize with MLWritable {

  def setFeaturesCol(featuresColName: String): this.type = set(featuresCol, featuresColName)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def setFeatureSize(value: Array[Int]): this.type = {
    this.featureSize = value
    this
  }

  def setBatchSize(value: Int): this.type = set(batchSize, value)

  def getFeatureSize: Array[Int] = this.featureSize

  /**
   * Perform a prediction on featureCol, and write result to the predictionCol.
   */
  protected override def internalTransform(dataFrame: DataFrame): DataFrame = {
    val featureType = dataFrame.schema($(featuresCol)).dataType
    val featureColIndex = dataFrame.schema.fieldIndex($(featuresCol))
    val featureFunc = getConvertFunc(featureType)
    val sc = dataFrame.sqlContext.sparkContext
    val modelBroadCast = ModelBroadcast[T]().broadcast(sc, model.evaluate())
    val localBatchSize = $(batchSize)
    val transformerBC = sc.broadcast(SampleToMiniBatch[T](localBatchSize))

    val resultRDD = dataFrame.rdd.mapPartitions { rowIter =>
      val localModel = modelBroadCast.value()

      val transformer = transformerBC.value.cloneTransformer()
      rowIter.grouped(localBatchSize).flatMap { rowBatch =>
        val samples = rowBatch.map { row =>
          val features = featureFunc(row, featureColIndex)
          val featureBuffer = features.head match {
            case dd: Double => features.asInstanceOf[Seq[Double]].map(ev.fromType(_))
            case ff: Float => features.asInstanceOf[Seq[Float]].map(ev.fromType(_))
          }
          Sample(Tensor(featureBuffer.toArray, featureSize))
        }.toIterator
        val predictions = transformer(samples).flatMap { batch =>
          val batchResult = localModel.forward(batch.getInput()).toTensor.squeeze()
          if (batchResult.size().length == 2) {
            batchResult.split(1).map(outputToPrediction)
          } else if (batchResult.size().length == 1) {
            Array(outputToPrediction(batchResult))
          } else {
            throw new RuntimeException(
              "unexpected batchResult dimension: " + batchResult.size().mkString(", "))
          }
        }
        rowBatch.toIterator.zip(predictions).map { case (row, predict) =>
          Row.fromSeq(row.toSeq ++ Seq(predict))
        }
      }
    }

    val resultSchema = transformSchema(dataFrame.schema)
    dataFrame.sqlContext.createDataFrame(resultRDD, resultSchema)
  }

  protected def outputToPrediction(output: Tensor[T]): Any = {
    output.clone().storage().array().map(ev.toType[Double])
  }

  override def transformSchema(schema : StructType): StructType = {
    validateDataType(schema, $(featuresCol))
    SchemaUtils.appendColumn(schema, $(predictionCol), ArrayType(DoubleType, containsNull = false))
  }

  override def copy(extra: ParamMap): NNModel[T] = {
    val copied = new NNModel(model.cloneModule(), featureSize.clone(), uid).setParent(parent)
    copyValues(copied, extra)
  }

  override def write: MLWriter = new NNModel.NNModelWriter[T](this)
}

object NNModel extends MLReadable[NNModel[_]] {

  import scala.language.existentials
  implicit val format: DefaultFormats.type = DefaultFormats

  private[nnframes] class NNModelReader() extends MLReader[NNModel[_]] {
    override def load(path: String): NNModel[_] = {
      val (meta, model, typeTag) = NNModel.getMetaAndModel(path, sc)
      val featureSize = (meta.metadata \ "featureSize").extract[Seq[Int]].toArray
      val nnModel = typeTag match {
        case "TensorDouble" =>
          new NNModel[Double](model.asInstanceOf[Module[Double]], featureSize)
        case "TensorFloat" =>
          new NNModel[Float](model.asInstanceOf[Module[Float]], featureSize)
        case _ =>
          throw new Exception("Only support float and double for now")
      }

      DefaultParamsWriterWrapper.getAndSetParams(nnModel, meta)
      nnModel
    }
  }

  private[nnframes] def getMetaAndModel(path: String, sc: SparkContext) = {
    val meta = DefaultParamsWriterWrapper.loadMetadata(path, sc)
    val (modulePath, weightPath) =
      new Path(path, "module").toString -> new Path(path, "weight").toString
    val typeTag = (meta.metadata \ "tensorDataType").extract[String]

    val model = typeTag match {
      case "TensorDouble" =>
        ModuleLoader.loadFromFile[Double](modulePath, weightPath)
      case "TensorFloat" =>
        ModuleLoader.loadFromFile[Float](modulePath, weightPath)
      case _ =>
        throw new Exception("Only support float and double for now")
    }
    (meta, model, typeTag)
  }

  class NNModelWriter[@specialized(Float, Double) T: ClassTag](
    instance: NNModel[T])(implicit ev: TensorNumeric[T]) extends MLWriter {
    override protected def saveImpl(path: String): Unit = {
      val extraMetaData: JObject = "featureSize" -> instance.featureSize.toSeq
      NNModel.saveImpl[T](instance, instance.model,
        path, sc, shouldOverwrite, Some(extraMetaData))
    }
  }

  /**
   * Helper method for saving a NNModel to disk.
   * For compatibility with spark ml pipeline, TensorDataType is stored separately in extraMetadata.
   *
   * @tparam T TensorDataType
   * @param instance  NNModel
   * @param path  Path to which to save the NNModel.
   * @param extraMetadata  Metadata such as featureSize.
   */
  private[nnframes] def saveImpl[@specialized(Float, Double) T: ClassTag](
      instance: NNModel[T],
      module: Module[T],
      path: String,
      sc: SparkContext,
      isOverWrite: Boolean = false,
      extraMetadata: Option[JObject] = None)(implicit ev: TensorNumeric[T]): Unit = {
    val tensorDataType = ev.getType() match {
      case TensorDouble => "TensorDouble"
      case TensorFloat => "TensorFloat"
      case _ => throw new Exception("Only support Double and Float for now")
    }

    val extra = extraMetadata.getOrElse(JObject()) ~ ("tensorDataType" -> tensorDataType)
    DefaultParamsWriterWrapper.saveMetadata(instance, path, sc, Option(extra))
    val (modulePath, weightPath) =
      new Path(path, "module").toString -> new Path(path, "weight").toString
    module.saveModule(modulePath, weightPath, isOverWrite)
  }

  override def read: MLReader[NNModel[_]] = new NNModelReader

  override def load(path: String): NNModel[_] = read.load(path)
}

