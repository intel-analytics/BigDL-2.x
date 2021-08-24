// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: PSI.proto

package com.intel.analytics.zoo.ppml.psi.generated;

public interface DownloadResponseOrBuilder extends
    // @@protoc_insertion_point(interface_extends:DownloadResponse)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>string task_id = 1;</code>
   * @return The taskId.
   */
  String getTaskId();
  /**
   * <code>string task_id = 1;</code>
   * @return The bytes for taskId.
   */
  com.google.protobuf.ByteString
      getTaskIdBytes();

  /**
   * <code>.SIGNAL status = 2;</code>
   * @return The enum numeric value on the wire for status.
   */
  int getStatusValue();
  /**
   * <code>.SIGNAL status = 2;</code>
   * @return The status.
   */
  SIGNAL getStatus();

  /**
   * <code>int32 split = 3;</code>
   * @return The split.
   */
  int getSplit();

  /**
   * <code>int32 num_split = 4;</code>
   * @return The numSplit.
   */
  int getNumSplit();

  /**
   * <code>int32 split_length = 5;</code>
   * @return The splitLength.
   */
  int getSplitLength();

  /**
   * <code>int32 total_length = 6;</code>
   * @return The totalLength.
   */
  int getTotalLength();

  /**
   * <code>repeated string intersection = 7;</code>
   * @return A list containing the intersection.
   */
  java.util.List<String>
      getIntersectionList();
  /**
   * <code>repeated string intersection = 7;</code>
   * @return The count of intersection.
   */
  int getIntersectionCount();
  /**
   * <code>repeated string intersection = 7;</code>
   * @param index The index of the element to return.
   * @return The intersection at the given index.
   */
  String getIntersection(int index);
  /**
   * <code>repeated string intersection = 7;</code>
   * @param index The index of the value to return.
   * @return The bytes of the intersection at the given index.
   */
  com.google.protobuf.ByteString
      getIntersectionBytes(int index);
}
