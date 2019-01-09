package models

import java.io.File
import java.nio.file.Paths
import java.util.concurrent.locks.{ReadWriteLock, ReentrantReadWriteLock}

import com.amazonaws.auth.DefaultAWSCredentialsProviderChain
import com.amazonaws.services.s3.AmazonS3ClientBuilder
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.Engine
import ml.combust.bundle.BundleFile
import ml.combust.mleap.runtime.MleapSupport._
import ml.combust.mleap.runtime.frame.Transformer
import resource.managed


case class ModelParams(
                        recModelPath: String,
                        skuIndexerModelPath: String,
                        skuLookUpPath: String,
                        env: String,
                        bucketName: String,
                        var recModel: Option[AbstractModule[Activity, Activity, Float]],
                        var recModelVersion: Option[Long],
                        var skuIndexerModel: Option[Transformer],
                        var skuIndexerModelVersion: Option[Long],
                        var skuLookUp: Option[List[(String, String)]]
                 )

object ModelParams {

  private val s3client = AmazonS3ClientBuilder.standard().withCredentials(new DefaultAWSCredentialsProviderChain()).build()
  private val lock: ReadWriteLock = new ReentrantReadWriteLock()
  private val currentDir = Paths.get(".").toAbsolutePath.toString

  System.setProperty("bigdl.localMode", "true")
  println("BigDL localMode is: " + System.getProperty("bigdl.localMode"))
  Engine.init

  def apply(
             recModelPath: String,
             skuIndexerModelPath: String,
             skuLookUpPath: String,
             env: String
           ): ModelParams = ModelParams(
    recModelPath,
    skuIndexerModelPath,
    skuLookUpPath,
    env,
    if (env == "prod" || env == "canary") "ecomdatascience-p" else "ecomdatascience-np",
    loadBigDL(recModelPath),
    loadVersion(recModelPath),
    loadMleap(skuIndexerModelPath),
    loadVersion(skuIndexerModelPath),
    loadLookUp(skuLookUpPath)
  )

//  def downloadModel(params: ModelParams): Any = {
//    lock.writeLock().lock()
//    try {
//      val file = s3client.getObject(new GetObjectRequest(params.bucketName, "lu/subLatest.zip")).getObjectContent
//      println("Downloading the file")
//      val outputStream = new FileOutputStream(s"${params.currentDir}${params.subModelPath}")
//      IOUtils.copy(file, outputStream)
//      println("Download has completed")
//    }
//    catch {
//      case e: Exception => println(s"Cannot download model at ${params.env}-${params.bucketName}-${params.subModelPath}"); None
//    }
//    finally { lock.writeLock().unlock() }
//  }

  def loadBigDL(path: String) = {
    if (new File(currentDir+ "/" + path).exists()) {
      lock.readLock().lock()
      try {
//        Some(ZooModel.loadModel[Float](path).asInstanceOf[Module[Float]])
        Some(Module.loadModule[Float](path))
      }
          catch {
            case e: Exception => println(s"Cannot load bigDL model at $currentDir$path"); None
          }
      finally { lock.readLock().unlock() }
    }
    else {
      println(s"Cannot update bigDL model at $currentDir$path")
      None
    }
  }

  def loadMleap(modelPath: String): Option[Transformer] = {
    if (new File(currentDir + "/" + modelPath).exists()) {
      lock.readLock().lock()
      try Some(
        (for (bundleFile <- managed(BundleFile(s"jar:file:$currentDir/$modelPath"))) yield {
          bundleFile.loadMleapBundle().get
        }).opt.get.root
      )
      catch {
        case _: Exception => println(s"Cannot load Mleap model at $currentDir$modelPath"); None;
      }
      finally {
        lock.readLock().unlock()
      }
    }
    else {
      println(s"Cannot update Mleap model at $currentDir$modelPath")
      None
    }
  }

  def loadLookUp(filePath: String): Option[List[(String, String)]] = {
    lock.readLock().lock()
    try Some(
      scala.io.Source.fromFile(filePath).getLines().filter(_.nonEmpty)
        .map(_.split(" "))
        .map(x => (x(0), x(1))).toList
    )
    catch {
      case _: Exception => println(s"Cannot load lookUp file at $filePath"); None;
    }
    finally { lock.readLock().unlock() }
  }

  def loadVersion(path: String): Option[Long] = {
    lock.readLock().lock()
    try Some(new File(currentDir + "/" + path).lastModified())
    catch {
      case _: Exception => println(s"Cannot load model version at $currentDir$path"); None
    }
    finally { lock.readLock().unlock() }
  }

  def refresh(params: ModelParams): ModelParams = {
    params.recModel = loadBigDL(params.recModelPath)
    params.recModelVersion = loadVersion(params.recModelPath)
    params.skuIndexerModel = loadMleap(params.skuIndexerModelPath)
    params.skuIndexerModelVersion = loadVersion(params.skuIndexerModelPath)

    params
  }

}