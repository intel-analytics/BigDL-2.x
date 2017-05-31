// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: log_memory.proto

package org.tensorflow.framework;

public interface MemoryLogTensorAllocationOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.MemoryLogTensorAllocation)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <pre>
   * Process-unique step id.
   * </pre>
   *
   * <code>optional int64 step_id = 1;</code>
   */
  long getStepId();

  /**
   * <pre>
   * Name of the kernel making the allocation as set in GraphDef,
   * e.g., "affine2/weights/Assign".
   * </pre>
   *
   * <code>optional string kernel_name = 2;</code>
   */
  java.lang.String getKernelName();
  /**
   * <pre>
   * Name of the kernel making the allocation as set in GraphDef,
   * e.g., "affine2/weights/Assign".
   * </pre>
   *
   * <code>optional string kernel_name = 2;</code>
   */
  com.google.protobuf.ByteString
      getKernelNameBytes();

  /**
   * <pre>
   * Allocated tensor details.
   * </pre>
   *
   * <code>optional .tensorflow.TensorDescription tensor = 3;</code>
   */
  boolean hasTensor();
  /**
   * <pre>
   * Allocated tensor details.
   * </pre>
   *
   * <code>optional .tensorflow.TensorDescription tensor = 3;</code>
   */
  org.tensorflow.framework.TensorDescription getTensor();
  /**
   * <pre>
   * Allocated tensor details.
   * </pre>
   *
   * <code>optional .tensorflow.TensorDescription tensor = 3;</code>
   */
  org.tensorflow.framework.TensorDescriptionOrBuilder getTensorOrBuilder();
}
