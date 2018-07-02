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
package com.intel.analytics.zoo.pipeline.api.net

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.nn.{Container, Graph, StaticGraph}
import com.intel.analytics.bigdl.serialization.Bigdl.BigDLModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer._
import com.intel.analytics.zoo.pipeline.api.keras.layers.KerasLayerWrapper
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import org.json4s._
import org.json4s.jackson.JsonMethods._

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.reflect.ClassTag
import scala.reflect.io.Path


class GraphNet[T: ClassTag](graph: Graph[T])(implicit ev: TensorNumeric[T])
  extends Container[Activity, Activity, T] with NetUtils[T, GraphNet[T]] {

  // need to refer this object to make the register effective
  GraphNet

  private val labor = graph
  modules.append(labor)

  def getSubModules(): List[AbstractModule[Activity, Activity, T]] = {
    this.labor.modules.toList
  }

  val outputNodes = NetUtils.getGraphOutputs(graph)

  val inputNodes = graph.inputs

  override def updateOutput(input: Activity): Activity = {
    output = labor.updateOutput(input)
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    gradInput = labor.updateGradInput(input, gradOutput)
    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    labor.accGradParameters(input, gradOutput)
  }

  override def node(name: String): ModuleNode[T] = this.graph.node(name)

  override def newGraph(output: String): GraphNet[T] = {
    newGraph(Seq(output))
  }


  override def newGraph(outputs: Seq[String]): GraphNet[T] = {
    val inputs = graph.inputs
    val variables = NetUtils.getGraphVariables(graph)
      .asInstanceOf[Option[(Array[Tensor[T]], Array[Tensor[T]])]]

    graph match {
      case g: StaticGraph[T] =>
        val newGraph = Graph(inputs.toArray, nodes(outputs)
          .map(_.removeNextEdges()).toArray, variables)
        new GraphNet[T](newGraph)
      case g =>
        val newGraph = NetUtils.dynamic[T](inputs.toArray, nodes(outputs)
          .map(_.removeNextEdges()).toArray,
          variables, NetUtils.getGenerateBackward(g))
        new GraphNet[T](newGraph)
    }
  }

  override def toKeras(): KerasLayer[Activity, Activity, T] = {
    new KerasLayerWrapper[T](this)
  }
}

object GraphNet extends ContainerSerializable {

  ModuleSerializer.registerModule(
    "com.intel.analytics.zoo.pipeline.api.net.GraphNet",
    GraphNet)


  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
                                              builder: BigDLModule.Builder)
                                             (implicit ev: TensorNumeric[T]): Unit = {
    val labor = context.moduleData.module.
      asInstanceOf[GraphNet[T]].labor
    val subModule = ModuleSerializer.serialize(SerializeContext(ModuleData(labor,
      new ArrayBuffer[String](), new ArrayBuffer[String]()), context.storages,
      context.storageType, _copyWeightAndBias))
    builder.addSubModules(subModule.bigDLModule)
  }

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
      (implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {
    val subProtoModules = context.bigdlModule.getSubModulesList.asScala
    val subModules = subProtoModules.map(module => {
      val subModuleData = ModuleSerializer.load(DeserializeContext(module,
        context.storages, context.storageType, _copyWeightAndBias))
      subModuleData.module
    })
    val tGraph = subModules.head.asInstanceOf[StaticGraph[T]]
    tGraph
  }
}


object NetUtils {
  private[zoo] def getGraphOutputs[T](graph: Graph[T]): Seq[ModuleNode[T]] = {
    KerasUtils.invokeMethod(graph, "outputs").asInstanceOf[Seq[ModuleNode[T]]]
  }

  private[zoo] def getGraphVariables[T](graph: Graph[T]) = {
    KerasUtils.invokeMethod(graph, "variables")
      .asInstanceOf[Option[(Array[Tensor[T]], Array[Tensor[T]])]]
  }

  private[zoo] def getGenerateBackward[T](graph: Graph[T]): Boolean = {
    KerasUtils.invokeMethod(graph, "generateBackward").asInstanceOf[Boolean]
  }

  private[zoo] def dynamic[T](
       input : Array[ModuleNode[T]],
       output : Array[ModuleNode[T]],
       variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None,
       generateBackward: Boolean = true)
       (implicit ev: TensorNumeric[T], ev2: ClassTag[T]): Graph[T] = {
    import scala.reflect.runtime.{universe => ru}
    val m = ru.runtimeMirror(Graph.getClass.getClassLoader)
    val mirror = m.reflect(Graph)
    val dynamic = mirror.symbol.typeSignature
      .member(ru.newTermName("dynamic"))
      .filter(_.asMethod.paramss.flatten.length == 6)

    val result = mirror.reflectMethod(dynamic.asMethod)(input, output,
      variables, generateBackward, ev2, ev)

    result.asInstanceOf[Graph[T]]
  }

  implicit val formats = DefaultFormats

  private[zoo] def processTFFolder(folder: String): (String, Seq[String], Seq[String]) = {
    val folderPath = Path(folder)
    if (!folderPath.exists) {
      throw new IllegalArgumentException(s"$folder does not exist")
    }

    val modelPath = folderPath / Path("frozen_inference_graph.pb")
    if (!modelPath.exists) {
      throw new IllegalArgumentException(
        s"${modelPath.path} does not exist")
    }
    val metaPath = folderPath / Path("graph_meta.json")
    if (!metaPath.exists) {
      throw new IllegalArgumentException(
        s"${metaPath.path} does not exist")
    }

    val jsonStr = Source.fromFile(metaPath.jfile).getLines().mkString

    val meta = parse(jsonStr).camelizeKeys.extract[Meta]
    (modelPath.toString(), meta.inputNames, meta.outputNames)
  }

  private[zoo] def removePort(nodes: Seq[String]): Seq[String] = {
    nodes.map(node => if (node contains ":") node.split(":")(0) else node)
  }
}

private[zoo] case class Meta(inputNames: Array[String], outputNames: Array[String])


trait NetUtils[T, D <: Module[T] with NetUtils[T, D]] {

  /**
   * Return the nodes in the graph as specified by the names
   */
  def nodes(names: Seq[String]): Seq[ModuleNode[T]] = {
    names.map(node)
  }

  /**
   * Return the node in the graph as specified by the name
   */
  def node(name: String): ModuleNode[T]

  /**
   * Freeze the model from the bottom up to the layers
   * specified by names (inclusive).
   *
   * This is useful for finetune a model
   */
  def freezeUpTo(names: String*): this.type = {
    dfs(nodes(names)).foreach(_.element.freeze())
    this
  }

  /**
   * Specify a node as output and return a new graph using
   * the existing nodes
   */
  def newGraph(output: String): D

  /**
   * Specify a seq of nodes as output and return a new graph using
   * the existing nodes
   */
  def newGraph(outputs: Seq[String]): D

  def toKeras(): KerasLayer[Activity, Activity, T]

  private def dfs(endPoints: Seq[ModuleNode[T]]): Iterator[ModuleNode[T]] = {
    new Iterator[ModuleNode[T]] {
      private val stack = new mutable.Stack[ModuleNode[T]]()
      endPoints.map(stack.push)
      private val visited = new mutable.HashSet[ModuleNode[T]]()

      override def hasNext: Boolean = stack.nonEmpty

      override def next(): ModuleNode[T] = {
        require(hasNext, "No more elements in the graph")
        val node = stack.pop()
        visited.add(node)
        val nextNodes = node.prevNodes
        // to preserve order
        val nodesSet = mutable.LinkedHashSet[ModuleNode[T]]()
        nextNodes.foreach(nodesSet.add)
        nodesSet.filter(!visited.contains(_))
          .filter(!stack.contains(_)).foreach(stack.push)
        node
      }
    }
  }
}
