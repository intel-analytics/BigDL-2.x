/*
 * Copyright 2016 The BigDL Authors.
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
package com.intel.analytics.bigdl.utils.tf

import java.io.{DataInputStream, InputStream, FileReader => JFileReader}
import java.nio.ByteOrder
import java.util
import java.util.List

import com.google.protobuf.{CodedInputStream, TextFormat}
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Graph
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.tf.TensorflowToBigDL._
import com.intel.analytics.bigdl.utils.{DirectedGraph, FileReader, Node}
import org.tensorflow.framework.{GraphDef, NodeDef}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object TensorflowLoader{

  type Context[T] = mutable.HashMap[String, (Tensor[T], Tensor[T], Option[Seq[(Int, Int)]])]

  /**
   * Load tensorflow model from a prototxt file
   * @param graphPrototxt where is the tensorflow protobuf file
   * @param inputs input node names
   * @param outputs output node names
   * @param byteOrder file byteOrder
   * @return
   */
  def load[T: ClassTag](graphPrototxt: String, inputs: Seq[String], outputs: Seq[String],
        byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): Module[T] = {
    // Get node list
    val nodeList = parse(graphPrototxt)

    // Construct tf node graph
    val (tfGraph, adjustedInputs) =
      buildTFGraph(nodeList, outputs, (node: NodeDef) => inputs.contains(node.getName))

    // Build BigDL model from the tf node graph
    buildBigDLModel(tfGraph, adjustedInputs, outputs, byteOrder, graphPrototxt)
  }

  /**
   * Parse a tensorflow model protobuf binary file, read a list of op nodes from it
   * @param graphProtoTxt where is the tf protobuf file
   * @return
   */
  private[bigdl] def parse(graphProtoTxt: String) : List[NodeDef] = {
    var fr: FileReader = null
    var in: InputStream = null
    try {
      fr = FileReader(graphProtoTxt)
      in = fr.open()
      val reader = CodedInputStream.newInstance(new DataInputStream(in))
      reader.setSizeLimit(0x7fffffff)

      val graph = GraphDef.parseFrom(reader)
      graph.getNodeList
    } finally {
      if (fr != null) fr.close()
      if (in != null) in.close()
    }

  }

  /**
   * Parse a tensorflow model protobuf text file, read a list of op nodes from it
   * @param graphProtoTxt where is the tf protobuf file
   * @return
   */
  private[bigdl] def parseTxt(graphProtoTxt: String) : List[NodeDef] = {
    val f = new java.io.File(graphProtoTxt)
    require(f.exists(), graphProtoTxt + " does not exists")

    val reader = new JFileReader(f)

    val graph = GraphDef.newBuilder()

    TextFormat.merge(reader, graph)

    graph.build().getNodeList
  }

  /**
   * Build tf ops graph from a given node list
   * @param nodes
   * @param outputNodeNames
   * @return
   */
  private[bigdl] def buildTFGraph(nodes : List[NodeDef], outputNodeNames: Seq[String],
                                  isInput: (NodeDef) => Boolean = (_: NodeDef) => false)
  : (DirectedGraph[NodeDef], Seq[String]) = {
    import scala.collection.JavaConverters._
    var name2Node = nodes.asScala.map(n => n.getName -> new Node(n)).toMap

    // Process node with multiple tensor output, each tensor is regarded as a node
    nodes.asScala
      .flatMap(_.getInputList.asScala)
      .filter(_.split(TENSOR_SEPARATOR).length > 1)
      .foreach { nameWithChannel =>
        val name = nameWithChannel.split(TENSOR_SEPARATOR).head
        val tfNode = NodeDef.newBuilder(name2Node(name).element)
          .setName(nameWithChannel).build()
        name2Node += nameWithChannel -> new Node(tfNode)
      }

    // Build graph
    val outputNodes = if (outputNodeNames == null) {
      name2Node.valuesIterator.filter(_.nextNodes.isEmpty).toArray
    } else {
      val results = name2Node.valuesIterator.toArray.filter(n =>
        outputNodeNames.contains(n.element.getName))
      require(results.length == outputNodeNames.length, "Invalid outputNode names")
      results
    }

    def connect(nodes: Seq[Node[NodeDef]]): Seq[String] = {
      var inputCounter = 0
      val queue = new mutable.Queue[Node[NodeDef]]()
      val visited = mutable.Set[Node[NodeDef]]()
      val inputs = new mutable.ArrayBuffer[String]()

      // Do a BFS to connect the nodes
      queue.enqueue(nodes: _*)
      while(queue.nonEmpty) {
        val node = queue.dequeue()
        if (!visited(node)) {
          visited += node
          if (!isInput(node.element) && !node.element.getInputList.isEmpty) {
            // continue to traverse
            node.element.getInputList.asScala.foreach { preNodeName =>
              // It is tricky here, remove the first char in the name of control dep node
              val preNode = if (preNodeName.charAt(0) == '^') {
                val name = preNodeName.substring(1)
                val preNode = name2Node(name)
                val dependencyNode = Node(NodeDef.newBuilder()
                  .setOp("DependencyNode")
                  .addInput(preNode.element.getName)
                  .setName(s"depends_on_${preNode.element.getName}")
                  .build())
                preNode -> dependencyNode -> node
                preNode
              } else {
                val preNode = name2Node(preNodeName)
                preNode -> node
                preNode
              }
              queue.enqueue(preNode)
            }
          } else {
            if (isInput(node.element) && node.element.getOp != "Placeholder") {
              // if the predefined input node is not a Placeholder, add one to match the Input node
              val name = s"input$inputCounter"
              val placeholder = NodeDef.newBuilder()
                .setName(name)
                .setOp("Placeholder").build()
              inputCounter = inputCounter + 1
              val n = Node(placeholder)
              n -> node
              inputs += name
            } else if (node.element.getOp == "Placeholder") {
              inputs += node.element.getName
            }
          }
        }

      }
      inputs
    }

    val inputs = connect(outputNodes)

    val dummyOutput = new Node[NodeDef](null)
    outputNodes.foreach(_ -> dummyOutput)
    (dummyOutput.graph(reverse = true), inputs)
  }

  private[bigdl] def buildBigDLModel[T: ClassTag](
      tfGraph: DirectedGraph[NodeDef],
      inputs: Seq[String],
      outputs: Seq[String],
      byteOrder: ByteOrder,
      graphPrototxt: String,
      ctx: Option[Context[T]] = None
    )(implicit ev: TensorNumeric[T]): Module[T] = {
    import scala.collection.JavaConverters._

    // Map from tensorflow node to the converted BigDL node
    val convertedNode = new mutable.HashMap[Node[NodeDef],
      Node[AbstractModule[Activity, Activity, T]]]()
    val nameToNode =
      new mutable.HashMap[String, Node[AbstractModule[Activity, Activity, T]]]()
    val context = ctx.getOrElse(
      new mutable.HashMap[String, (Tensor[T], Tensor[T], Option[Seq[(Int, Int)]])])

    // BFS to keep the input order same
    tfGraph.BFS.foreach(n => {
      if (n.element == null) {
        // Dummy node, skip
      } else if (convertedNode.get(n).isDefined) {
        // converted node, skip
      } else {
        val errorMsg =
          s"""
            | Cannot convert the given tensorflow operation graph to BigDL model. The convert fails
            | at node ${n.element.getName}.
            | To investigate the model. Please use the dump_tf_graph.py to dump the graph, then use
            | Tensorboard to visualize the model.
            |
            | python dump_tf_graph.py $graphPrototxt
            | tensorboard --logdir ./log
            |
            | You can find the dump_tf_graph.py in the bin folder of the dist package, or scripts
            | folder in the source code.
          """.stripMargin

        val (module, nodes, inputNodes) =
          extract[T](n.graph(reverse = true), context, byteOrder).getOrElse(
            throw new UnsupportedOperationException(errorMsg)
          )

        // set name
        if (nodes.size() == 1) {
          // Use tf operation name if one to one map
          module.setName(removeColon(nodes.get(0).element.getName()))
        } else {
          // Many to one map
          val name = removeColon(findCommonPrefix(nodes.asScala.map(_.element.getName)))
          if (name == "") {
            // Use a name combine nodes
            module.setName(s"[${nodes.asScala.map(_.element.getName).map(_.replaceAll("/", "\\\\"))
              .map(removeColon(_)).mkString(", ")}]")
          } else {
            // Use the common name
            module.setName(name + "/" + module.getName())
          }
        }
        val node = new Node(module)
        nodes.asScala.foreach(m => {
          convertedNode(m) = node
          nameToNode(m.element.getName) = node
        })

        // These two pieces of code are all necessary
        val nextNodes = n.nextNodes.filter(
          n => n.element != null &&
            convertedNode.contains(n) && !context.contains(n.element.getName)
        ).map(convertedNode(_)).filter(_ != node)
        nextNodes.foreach(node -> _)

        val preNodes = inputNodes.flatMap(_.prevNodes)
          .filter(n => n.element != null && convertedNode.contains(n)
            && !context.contains(n.element.getName))
          .map(convertedNode(_)).filter(_ != node)
        preNodes.foreach(_ -> node)
      }
    })

    val inputNodes = inputs
      .map(n => nameToNode.getOrElse(n, throw new IllegalArgumentException(s"Can't find node $n")))
    val outputNodes = outputs
      .map(n => nameToNode.getOrElse(n, throw new IllegalArgumentException(s"Can't find node $n")))


    val weights = ArrayBuffer[Tensor[T]]()
    val gradients = ArrayBuffer[Tensor[T]]()
    for ((weight, grad, _) <- context.values) {
      weights += weight
      gradients += grad
    }

    Graph(inputNodes.toArray, outputNodes.toArray, Some((weights.toArray, gradients.toArray)))
  }

  /**
   * Extract one module and the corresponding node list from the given graph
   * @param graph
   * @return
   */
  private[bigdl] def extract[T: ClassTag](graph: DirectedGraph[NodeDef],
      context: Context[T], byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): Option[(
    AbstractModule[Activity, Activity, T],
      List[Node[NodeDef]],
      Seq[Node[NodeDef]]
    )] = {

    var i = 0
    while(i < patterns.length) {
      val (result, inputs) = matchGraph(graph, patterns(i).topology)
      if (result.size != 0) {
        // get model
        return Some(patterns(i).layer(graph, context, byteOrder), result, inputs)
      }
      i += 1
    }
    None
  }

  private def matchGraph(graph: DirectedGraph[NodeDef], pattern: DirectedGraph[String])
      : (List[Node[NodeDef]], Seq[Node[NodeDef]]) = {
    require(graph.reverse && pattern.reverse, "Must pass in reversed graph")
    val patternToGraph = new mutable.HashMap[Node[String], Node[NodeDef]]()
    val inputs = new ArrayBuffer[Node[NodeDef]]()
    patternToGraph(pattern.source) = graph.source

    pattern.BFS.foreach(patternNode => {
      if (patternNode.element != N_INPUT_PLACEHOLDER && patternNode.element != INPUT_PLACEHOLDER) {
        // Normal operation node
        if (patternToGraph.get(patternNode).isEmpty) return (util.Collections.emptyList(), Seq())

        val graphNode = patternToGraph(patternNode)
        // Operation type should match
        if (patternNode.element != graphNode.element.getOp) return (
          util.Collections.emptyList(), Seq())

        // Prev nodes number should be same except for the Ninput case
        val patternInputLength = patternNode.prevNodes.length
        val graphInputLength = graphNode.prevNodes.
          filterNot(_.element.getOp == "DependencyNode").length
        if (patternInputLength != graphInputLength &&
          !patternNode.prevNodes.exists(_.element == N_INPUT_PLACEHOLDER)) {
          return (util.Collections.emptyList(), Seq())
        }

        var i = 0
        var direction = 0
        var j = 0
        while (i < patternNode.prevNodes.length) {
          if (patternNode.prevNodes(i).element == N_INPUT_PLACEHOLDER) {
            require(patternNode.prevNodes.count(_.element == N_INPUT_PLACEHOLDER) == 1,
              s"only support one $N_INPUT_PLACEHOLDER ")
            direction = 1
            // skip the left input nodes of graphNode,
            // once we find the placeholder, we start from another side
            if (!inputs.contains(graphNode)) {
              inputs.append(graphNode)
            }
          } else if (patternNode.prevNodes(i).element == INPUT_PLACEHOLDER) {
            // skip input placeholder
            if (!inputs.contains(graphNode)) {
              inputs.append(graphNode)
            }
          } else {
            val posPattern = { if (direction == 0) i else patternNode.prevNodes.length - 1 - j}
            val posGraph = { if (direction == 0) i else graphNode.prevNodes.length - 1 - j}
            val pn = patternNode.prevNodes(posPattern)
            val gn = graphNode.prevNodes(posGraph)
            if (patternToGraph.contains(pn)) {
              if (!patternToGraph(pn).eq(gn)) return (util.Collections.emptyList(), Seq())
            } else {
              patternToGraph(pn) = gn
            }
            if (direction == 1) j += 1
          }
          i += 1
        }
      }
    })
    import scala.collection.JavaConverters._
    return (patternToGraph.valuesIterator.toList.asJava, inputs)
  }

  private def findCommonPrefix(data: Seq[String]): String = {
    if (data.length == 0) return ""
    var shortest = data(0).length
    data.foreach(s => if (s.length < shortest) shortest = s.length)
    var prefix = ""
    var i = 0
    while(i < shortest) {
      var c = data(0).charAt(i)
      data.foreach(s => if (c != s.charAt(i)) return removeLast(prefix))
      prefix += c
      i += 1
    }

    return removeLast(prefix)
  }

  private def removeLast(s: String): String = {
    if (s.length == 0) return s
    if (s.charAt(s.length - 1) == '/') s.substring(0, s.length - 1) else s
  }

  private def removeColon(s: String): String = {
    s.replaceAll(":", "")
  }
}
