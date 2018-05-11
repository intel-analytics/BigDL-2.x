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
package com.intel.analytics.bigdl

import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.nn.{DynamicGraph, Graph, StaticGraph}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.pipeline.api.keras.layers.KerasLayerWrapper

import scala.collection.mutable
import scala.reflect.ClassTag


object NetUtils {

  type GraphWithUtils[T] = Graph[T] with NetUtils[T, _ <: Graph[T] with NetUtils[T, _]]

  def getOutputs[T](graph: Graph[T]): Seq[ModuleNode[T]] = {
    graph.outputs
  }

  def withGraphUtils[T: ClassTag](graph: Graph[T])
        (implicit ev: TensorNumeric[T]):
  Graph[T] with NetUtils[T, _ <: GraphWithUtils[T]] = {
    val inputs = graph.inputs
    val outputs = graph.outputs
    val variables = graph.variables

    graph match {

      case g: StaticGraph[T] =>
        new StaticNetWithUtils[T](inputs, outputs, variables)
      case g: DynamicGraph[T] =>
        new DynamicNetWithUtils[T](inputs, outputs, variables, g.generateBackward)
    }
  }

}

class StaticNetWithUtils[T: ClassTag](
   private val _inputs : Seq[ModuleNode[T]],
   private val _outputs : Seq[ModuleNode[T]],
   private val _variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None)
                                     (implicit ev: TensorNumeric[T])
  extends StaticGraph[T](_inputs, _outputs, _variables)
    with NetUtils[T, StaticNetWithUtils[T]] {

  override def newGraph(output: String): StaticNetWithUtils[T] = {
    new StaticNetWithUtils[T](inputs, nodes(Seq(output)), _variables)
  }

  override def newGraph(outputs: Seq[String]): StaticNetWithUtils[T] = {
    new StaticNetWithUtils[T](inputs, nodes(outputs))
  }

  override def toKeras() = {
    new KerasLayerWrapper[T](this)
  }

}

class DynamicNetWithUtils[T: ClassTag](
  _inputs : Seq[ModuleNode[T]],
  _outputs : Seq[ModuleNode[T]],
  _variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None,
  generateBackward: Boolean = true)(implicit ev: TensorNumeric[T])
  extends DynamicGraph[T](_inputs, _outputs, _variables, generateBackward)
    with NetUtils[T, DynamicNetWithUtils[T]] {

  override def newGraph(output: String): DynamicNetWithUtils[T] = {
    new DynamicNetWithUtils[T](
      inputs,
      nodes(Seq(output)).map(_.removeNextEdges()),
      _variables, generateBackward)
  }

  override def newGraph(outputs: Seq[String]): DynamicNetWithUtils[T] = {
    new DynamicNetWithUtils[T](
      inputs,
      nodes(outputs).map(_.removeNextEdges()), _variables, generateBackward)
  }

  override def toKeras() = {
    new KerasLayerWrapper[T](this)
  }
}


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
