// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: PSI.proto

package com.intel.analytics.zoo.ppml.psi.generated;

/**
 * Protobuf type {@code UploadRequest}
 */
public final class UploadRequest extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:UploadRequest)
        com.intel.analytics.zoo.ppml.psi.generated.UploadRequestOrBuilder {
private static final long serialVersionUID = 0L;
  // Use UploadRequest.newBuilder() to construct.
  private UploadRequest(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private UploadRequest() {
    taskId_ = "";
    clientId_ = "";
    hashedID_ = com.google.protobuf.LazyStringArrayList.EMPTY;
  }

  @Override
  @SuppressWarnings({"unused"})
  protected Object newInstance(
      UnusedPrivateParameter unused) {
    return new com.intel.analytics.zoo.ppml.psi.generated.UploadRequest();
  }

  @Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private UploadRequest(
      com.google.protobuf.CodedInputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    this();
    if (extensionRegistry == null) {
      throw new NullPointerException();
    }
    int mutable_bitField0_ = 0;
    com.google.protobuf.UnknownFieldSet.Builder unknownFields =
        com.google.protobuf.UnknownFieldSet.newBuilder();
    try {
      boolean done = false;
      while (!done) {
        int tag = input.readTag();
        switch (tag) {
          case 0:
            done = true;
            break;
          case 10: {
            String s = input.readStringRequireUtf8();

            taskId_ = s;
            break;
          }
          case 18: {
            String s = input.readStringRequireUtf8();

            clientId_ = s;
            break;
          }
          case 24: {

            split_ = input.readInt32();
            break;
          }
          case 32: {

            numSplit_ = input.readInt32();
            break;
          }
          case 40: {

            splitLength_ = input.readInt32();
            break;
          }
          case 48: {

            totalLength_ = input.readInt32();
            break;
          }
          case 58: {
            String s = input.readStringRequireUtf8();
            if (!((mutable_bitField0_ & 0x00000001) != 0)) {
              hashedID_ = new com.google.protobuf.LazyStringArrayList();
              mutable_bitField0_ |= 0x00000001;
            }
            hashedID_.add(s);
            break;
          }
          default: {
            if (!parseUnknownField(
                input, unknownFields, extensionRegistry, tag)) {
              done = true;
            }
            break;
          }
        }
      }
    } catch (com.google.protobuf.InvalidProtocolBufferException e) {
      throw e.setUnfinishedMessage(this);
    } catch (java.io.IOException e) {
      throw new com.google.protobuf.InvalidProtocolBufferException(
          e).setUnfinishedMessage(this);
    } finally {
      if (((mutable_bitField0_ & 0x00000001) != 0)) {
        hashedID_ = hashedID_.getUnmodifiableView();
      }
      this.unknownFields = unknownFields.build();
      makeExtensionsImmutable();
    }
  }
  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return com.intel.analytics.zoo.ppml.psi.generated.PSIProto.internal_static_UploadRequest_descriptor;
  }

  @Override
  protected FieldAccessorTable
      internalGetFieldAccessorTable() {
    return com.intel.analytics.zoo.ppml.psi.generated.PSIProto.internal_static_UploadRequest_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            com.intel.analytics.zoo.ppml.psi.generated.UploadRequest.class, com.intel.analytics.zoo.ppml.psi.generated.UploadRequest.Builder.class);
  }

  public static final int TASK_ID_FIELD_NUMBER = 1;
  private volatile Object taskId_;
  /**
   * <code>string task_id = 1;</code>
   * @return The taskId.
   */
  @Override
  public String getTaskId() {
    Object ref = taskId_;
    if (ref instanceof String) {
      return (String) ref;
    } else {
      com.google.protobuf.ByteString bs = 
          (com.google.protobuf.ByteString) ref;
      String s = bs.toStringUtf8();
      taskId_ = s;
      return s;
    }
  }
  /**
   * <code>string task_id = 1;</code>
   * @return The bytes for taskId.
   */
  @Override
  public com.google.protobuf.ByteString
      getTaskIdBytes() {
    Object ref = taskId_;
    if (ref instanceof String) {
      com.google.protobuf.ByteString b = 
          com.google.protobuf.ByteString.copyFromUtf8(
              (String) ref);
      taskId_ = b;
      return b;
    } else {
      return (com.google.protobuf.ByteString) ref;
    }
  }

  public static final int CLIENT_ID_FIELD_NUMBER = 2;
  private volatile Object clientId_;
  /**
   * <code>string client_id = 2;</code>
   * @return The clientId.
   */
  @Override
  public String getClientId() {
    Object ref = clientId_;
    if (ref instanceof String) {
      return (String) ref;
    } else {
      com.google.protobuf.ByteString bs = 
          (com.google.protobuf.ByteString) ref;
      String s = bs.toStringUtf8();
      clientId_ = s;
      return s;
    }
  }
  /**
   * <code>string client_id = 2;</code>
   * @return The bytes for clientId.
   */
  @Override
  public com.google.protobuf.ByteString
      getClientIdBytes() {
    Object ref = clientId_;
    if (ref instanceof String) {
      com.google.protobuf.ByteString b = 
          com.google.protobuf.ByteString.copyFromUtf8(
              (String) ref);
      clientId_ = b;
      return b;
    } else {
      return (com.google.protobuf.ByteString) ref;
    }
  }

  public static final int SPLIT_FIELD_NUMBER = 3;
  private int split_;
  /**
   * <code>int32 split = 3;</code>
   * @return The split.
   */
  @Override
  public int getSplit() {
    return split_;
  }

  public static final int NUM_SPLIT_FIELD_NUMBER = 4;
  private int numSplit_;
  /**
   * <code>int32 num_split = 4;</code>
   * @return The numSplit.
   */
  @Override
  public int getNumSplit() {
    return numSplit_;
  }

  public static final int SPLIT_LENGTH_FIELD_NUMBER = 5;
  private int splitLength_;
  /**
   * <code>int32 split_length = 5;</code>
   * @return The splitLength.
   */
  @Override
  public int getSplitLength() {
    return splitLength_;
  }

  public static final int TOTAL_LENGTH_FIELD_NUMBER = 6;
  private int totalLength_;
  /**
   * <code>int32 total_length = 6;</code>
   * @return The totalLength.
   */
  @Override
  public int getTotalLength() {
    return totalLength_;
  }

  public static final int HASHEDID_FIELD_NUMBER = 7;
  private com.google.protobuf.LazyStringList hashedID_;
  /**
   * <code>repeated string hashedID = 7;</code>
   * @return A list containing the hashedID.
   */
  public com.google.protobuf.ProtocolStringList
      getHashedIDList() {
    return hashedID_;
  }
  /**
   * <code>repeated string hashedID = 7;</code>
   * @return The count of hashedID.
   */
  public int getHashedIDCount() {
    return hashedID_.size();
  }
  /**
   * <code>repeated string hashedID = 7;</code>
   * @param index The index of the element to return.
   * @return The hashedID at the given index.
   */
  public String getHashedID(int index) {
    return hashedID_.get(index);
  }
  /**
   * <code>repeated string hashedID = 7;</code>
   * @param index The index of the value to return.
   * @return The bytes of the hashedID at the given index.
   */
  public com.google.protobuf.ByteString
      getHashedIDBytes(int index) {
    return hashedID_.getByteString(index);
  }

  private byte memoizedIsInitialized = -1;
  @Override
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    memoizedIsInitialized = 1;
    return true;
  }

  @Override
  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    if (!getTaskIdBytes().isEmpty()) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 1, taskId_);
    }
    if (!getClientIdBytes().isEmpty()) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 2, clientId_);
    }
    if (split_ != 0) {
      output.writeInt32(3, split_);
    }
    if (numSplit_ != 0) {
      output.writeInt32(4, numSplit_);
    }
    if (splitLength_ != 0) {
      output.writeInt32(5, splitLength_);
    }
    if (totalLength_ != 0) {
      output.writeInt32(6, totalLength_);
    }
    for (int i = 0; i < hashedID_.size(); i++) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 7, hashedID_.getRaw(i));
    }
    unknownFields.writeTo(output);
  }

  @Override
  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (!getTaskIdBytes().isEmpty()) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(1, taskId_);
    }
    if (!getClientIdBytes().isEmpty()) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(2, clientId_);
    }
    if (split_ != 0) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt32Size(3, split_);
    }
    if (numSplit_ != 0) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt32Size(4, numSplit_);
    }
    if (splitLength_ != 0) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt32Size(5, splitLength_);
    }
    if (totalLength_ != 0) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt32Size(6, totalLength_);
    }
    {
      int dataSize = 0;
      for (int i = 0; i < hashedID_.size(); i++) {
        dataSize += computeStringSizeNoTag(hashedID_.getRaw(i));
      }
      size += dataSize;
      size += 1 * getHashedIDList().size();
    }
    size += unknownFields.getSerializedSize();
    memoizedSize = size;
    return size;
  }

  @Override
  public boolean equals(final Object obj) {
    if (obj == this) {
     return true;
    }
    if (!(obj instanceof com.intel.analytics.zoo.ppml.psi.generated.UploadRequest)) {
      return super.equals(obj);
    }
    com.intel.analytics.zoo.ppml.psi.generated.UploadRequest other = (com.intel.analytics.zoo.ppml.psi.generated.UploadRequest) obj;

    if (!getTaskId()
        .equals(other.getTaskId())) return false;
    if (!getClientId()
        .equals(other.getClientId())) return false;
    if (getSplit()
        != other.getSplit()) return false;
    if (getNumSplit()
        != other.getNumSplit()) return false;
    if (getSplitLength()
        != other.getSplitLength()) return false;
    if (getTotalLength()
        != other.getTotalLength()) return false;
    if (!getHashedIDList()
        .equals(other.getHashedIDList())) return false;
    if (!unknownFields.equals(other.unknownFields)) return false;
    return true;
  }

  @Override
  public int hashCode() {
    if (memoizedHashCode != 0) {
      return memoizedHashCode;
    }
    int hash = 41;
    hash = (19 * hash) + getDescriptor().hashCode();
    hash = (37 * hash) + TASK_ID_FIELD_NUMBER;
    hash = (53 * hash) + getTaskId().hashCode();
    hash = (37 * hash) + CLIENT_ID_FIELD_NUMBER;
    hash = (53 * hash) + getClientId().hashCode();
    hash = (37 * hash) + SPLIT_FIELD_NUMBER;
    hash = (53 * hash) + getSplit();
    hash = (37 * hash) + NUM_SPLIT_FIELD_NUMBER;
    hash = (53 * hash) + getNumSplit();
    hash = (37 * hash) + SPLIT_LENGTH_FIELD_NUMBER;
    hash = (53 * hash) + getSplitLength();
    hash = (37 * hash) + TOTAL_LENGTH_FIELD_NUMBER;
    hash = (53 * hash) + getTotalLength();
    if (getHashedIDCount() > 0) {
      hash = (37 * hash) + HASHEDID_FIELD_NUMBER;
      hash = (53 * hash) + getHashedIDList().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static com.intel.analytics.zoo.ppml.psi.generated.UploadRequest parseFrom(
      java.nio.ByteBuffer data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static com.intel.analytics.zoo.ppml.psi.generated.UploadRequest parseFrom(
      java.nio.ByteBuffer data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static com.intel.analytics.zoo.ppml.psi.generated.UploadRequest parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static com.intel.analytics.zoo.ppml.psi.generated.UploadRequest parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static com.intel.analytics.zoo.ppml.psi.generated.UploadRequest parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static com.intel.analytics.zoo.ppml.psi.generated.UploadRequest parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static com.intel.analytics.zoo.ppml.psi.generated.UploadRequest parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static com.intel.analytics.zoo.ppml.psi.generated.UploadRequest parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static com.intel.analytics.zoo.ppml.psi.generated.UploadRequest parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static com.intel.analytics.zoo.ppml.psi.generated.UploadRequest parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static com.intel.analytics.zoo.ppml.psi.generated.UploadRequest parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static com.intel.analytics.zoo.ppml.psi.generated.UploadRequest parseFrom(
      com.google.protobuf.CodedInputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }

  @Override
  public Builder newBuilderForType() { return newBuilder(); }
  public static Builder newBuilder() {
    return DEFAULT_INSTANCE.toBuilder();
  }
  public static Builder newBuilder(com.intel.analytics.zoo.ppml.psi.generated.UploadRequest prototype) {
    return DEFAULT_INSTANCE.toBuilder().mergeFrom(prototype);
  }
  @Override
  public Builder toBuilder() {
    return this == DEFAULT_INSTANCE
        ? new Builder() : new Builder().mergeFrom(this);
  }

  @Override
  protected Builder newBuilderForType(
      BuilderParent parent) {
    Builder builder = new Builder(parent);
    return builder;
  }
  /**
   * Protobuf type {@code UploadRequest}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:UploadRequest)
      com.intel.analytics.zoo.ppml.psi.generated.UploadRequestOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return com.intel.analytics.zoo.ppml.psi.generated.PSIProto.internal_static_UploadRequest_descriptor;
    }

    @Override
    protected FieldAccessorTable
        internalGetFieldAccessorTable() {
      return com.intel.analytics.zoo.ppml.psi.generated.PSIProto.internal_static_UploadRequest_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              com.intel.analytics.zoo.ppml.psi.generated.UploadRequest.class, com.intel.analytics.zoo.ppml.psi.generated.UploadRequest.Builder.class);
    }

    // Construct using com.intel.analytics.zoo.ppml.psi.generated.UploadRequest.newBuilder()
    private Builder() {
      maybeForceBuilderInitialization();
    }

    private Builder(
        BuilderParent parent) {
      super(parent);
      maybeForceBuilderInitialization();
    }
    private void maybeForceBuilderInitialization() {
      if (com.google.protobuf.GeneratedMessageV3
              .alwaysUseFieldBuilders) {
      }
    }
    @Override
    public Builder clear() {
      super.clear();
      taskId_ = "";

      clientId_ = "";

      split_ = 0;

      numSplit_ = 0;

      splitLength_ = 0;

      totalLength_ = 0;

      hashedID_ = com.google.protobuf.LazyStringArrayList.EMPTY;
      bitField0_ = (bitField0_ & ~0x00000001);
      return this;
    }

    @Override
    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return com.intel.analytics.zoo.ppml.psi.generated.PSIProto.internal_static_UploadRequest_descriptor;
    }

    @Override
    public com.intel.analytics.zoo.ppml.psi.generated.UploadRequest getDefaultInstanceForType() {
      return com.intel.analytics.zoo.ppml.psi.generated.UploadRequest.getDefaultInstance();
    }

    @Override
    public com.intel.analytics.zoo.ppml.psi.generated.UploadRequest build() {
      com.intel.analytics.zoo.ppml.psi.generated.UploadRequest result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    @Override
    public com.intel.analytics.zoo.ppml.psi.generated.UploadRequest buildPartial() {
      com.intel.analytics.zoo.ppml.psi.generated.UploadRequest result = new com.intel.analytics.zoo.ppml.psi.generated.UploadRequest(this);
      int from_bitField0_ = bitField0_;
      result.taskId_ = taskId_;
      result.clientId_ = clientId_;
      result.split_ = split_;
      result.numSplit_ = numSplit_;
      result.splitLength_ = splitLength_;
      result.totalLength_ = totalLength_;
      if (((bitField0_ & 0x00000001) != 0)) {
        hashedID_ = hashedID_.getUnmodifiableView();
        bitField0_ = (bitField0_ & ~0x00000001);
      }
      result.hashedID_ = hashedID_;
      onBuilt();
      return result;
    }

    @Override
    public Builder clone() {
      return super.clone();
    }
    @Override
    public Builder setField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        Object value) {
      return super.setField(field, value);
    }
    @Override
    public Builder clearField(
        com.google.protobuf.Descriptors.FieldDescriptor field) {
      return super.clearField(field);
    }
    @Override
    public Builder clearOneof(
        com.google.protobuf.Descriptors.OneofDescriptor oneof) {
      return super.clearOneof(oneof);
    }
    @Override
    public Builder setRepeatedField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        int index, Object value) {
      return super.setRepeatedField(field, index, value);
    }
    @Override
    public Builder addRepeatedField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        Object value) {
      return super.addRepeatedField(field, value);
    }
    @Override
    public Builder mergeFrom(com.google.protobuf.Message other) {
      if (other instanceof com.intel.analytics.zoo.ppml.psi.generated.UploadRequest) {
        return mergeFrom((com.intel.analytics.zoo.ppml.psi.generated.UploadRequest)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(com.intel.analytics.zoo.ppml.psi.generated.UploadRequest other) {
      if (other == com.intel.analytics.zoo.ppml.psi.generated.UploadRequest.getDefaultInstance()) return this;
      if (!other.getTaskId().isEmpty()) {
        taskId_ = other.taskId_;
        onChanged();
      }
      if (!other.getClientId().isEmpty()) {
        clientId_ = other.clientId_;
        onChanged();
      }
      if (other.getSplit() != 0) {
        setSplit(other.getSplit());
      }
      if (other.getNumSplit() != 0) {
        setNumSplit(other.getNumSplit());
      }
      if (other.getSplitLength() != 0) {
        setSplitLength(other.getSplitLength());
      }
      if (other.getTotalLength() != 0) {
        setTotalLength(other.getTotalLength());
      }
      if (!other.hashedID_.isEmpty()) {
        if (hashedID_.isEmpty()) {
          hashedID_ = other.hashedID_;
          bitField0_ = (bitField0_ & ~0x00000001);
        } else {
          ensureHashedIDIsMutable();
          hashedID_.addAll(other.hashedID_);
        }
        onChanged();
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    @Override
    public final boolean isInitialized() {
      return true;
    }

    @Override
    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      com.intel.analytics.zoo.ppml.psi.generated.UploadRequest parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (com.intel.analytics.zoo.ppml.psi.generated.UploadRequest) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private Object taskId_ = "";
    /**
     * <code>string task_id = 1;</code>
     * @return The taskId.
     */
    public String getTaskId() {
      Object ref = taskId_;
      if (!(ref instanceof String)) {
        com.google.protobuf.ByteString bs =
            (com.google.protobuf.ByteString) ref;
        String s = bs.toStringUtf8();
        taskId_ = s;
        return s;
      } else {
        return (String) ref;
      }
    }
    /**
     * <code>string task_id = 1;</code>
     * @return The bytes for taskId.
     */
    public com.google.protobuf.ByteString
        getTaskIdBytes() {
      Object ref = taskId_;
      if (ref instanceof String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (String) ref);
        taskId_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }
    /**
     * <code>string task_id = 1;</code>
     * @param value The taskId to set.
     * @return This builder for chaining.
     */
    public Builder setTaskId(
        String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  
      taskId_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>string task_id = 1;</code>
     * @return This builder for chaining.
     */
    public Builder clearTaskId() {
      
      taskId_ = getDefaultInstance().getTaskId();
      onChanged();
      return this;
    }
    /**
     * <code>string task_id = 1;</code>
     * @param value The bytes for taskId to set.
     * @return This builder for chaining.
     */
    public Builder setTaskIdBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  checkByteStringIsUtf8(value);
      
      taskId_ = value;
      onChanged();
      return this;
    }

    private Object clientId_ = "";
    /**
     * <code>string client_id = 2;</code>
     * @return The clientId.
     */
    public String getClientId() {
      Object ref = clientId_;
      if (!(ref instanceof String)) {
        com.google.protobuf.ByteString bs =
            (com.google.protobuf.ByteString) ref;
        String s = bs.toStringUtf8();
        clientId_ = s;
        return s;
      } else {
        return (String) ref;
      }
    }
    /**
     * <code>string client_id = 2;</code>
     * @return The bytes for clientId.
     */
    public com.google.protobuf.ByteString
        getClientIdBytes() {
      Object ref = clientId_;
      if (ref instanceof String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (String) ref);
        clientId_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }
    /**
     * <code>string client_id = 2;</code>
     * @param value The clientId to set.
     * @return This builder for chaining.
     */
    public Builder setClientId(
        String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  
      clientId_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>string client_id = 2;</code>
     * @return This builder for chaining.
     */
    public Builder clearClientId() {
      
      clientId_ = getDefaultInstance().getClientId();
      onChanged();
      return this;
    }
    /**
     * <code>string client_id = 2;</code>
     * @param value The bytes for clientId to set.
     * @return This builder for chaining.
     */
    public Builder setClientIdBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  checkByteStringIsUtf8(value);
      
      clientId_ = value;
      onChanged();
      return this;
    }

    private int split_ ;
    /**
     * <code>int32 split = 3;</code>
     * @return The split.
     */
    @Override
    public int getSplit() {
      return split_;
    }
    /**
     * <code>int32 split = 3;</code>
     * @param value The split to set.
     * @return This builder for chaining.
     */
    public Builder setSplit(int value) {
      
      split_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>int32 split = 3;</code>
     * @return This builder for chaining.
     */
    public Builder clearSplit() {
      
      split_ = 0;
      onChanged();
      return this;
    }

    private int numSplit_ ;
    /**
     * <code>int32 num_split = 4;</code>
     * @return The numSplit.
     */
    @Override
    public int getNumSplit() {
      return numSplit_;
    }
    /**
     * <code>int32 num_split = 4;</code>
     * @param value The numSplit to set.
     * @return This builder for chaining.
     */
    public Builder setNumSplit(int value) {
      
      numSplit_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>int32 num_split = 4;</code>
     * @return This builder for chaining.
     */
    public Builder clearNumSplit() {
      
      numSplit_ = 0;
      onChanged();
      return this;
    }

    private int splitLength_ ;
    /**
     * <code>int32 split_length = 5;</code>
     * @return The splitLength.
     */
    @Override
    public int getSplitLength() {
      return splitLength_;
    }
    /**
     * <code>int32 split_length = 5;</code>
     * @param value The splitLength to set.
     * @return This builder for chaining.
     */
    public Builder setSplitLength(int value) {
      
      splitLength_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>int32 split_length = 5;</code>
     * @return This builder for chaining.
     */
    public Builder clearSplitLength() {
      
      splitLength_ = 0;
      onChanged();
      return this;
    }

    private int totalLength_ ;
    /**
     * <code>int32 total_length = 6;</code>
     * @return The totalLength.
     */
    @Override
    public int getTotalLength() {
      return totalLength_;
    }
    /**
     * <code>int32 total_length = 6;</code>
     * @param value The totalLength to set.
     * @return This builder for chaining.
     */
    public Builder setTotalLength(int value) {
      
      totalLength_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>int32 total_length = 6;</code>
     * @return This builder for chaining.
     */
    public Builder clearTotalLength() {
      
      totalLength_ = 0;
      onChanged();
      return this;
    }

    private com.google.protobuf.LazyStringList hashedID_ = com.google.protobuf.LazyStringArrayList.EMPTY;
    private void ensureHashedIDIsMutable() {
      if (!((bitField0_ & 0x00000001) != 0)) {
        hashedID_ = new com.google.protobuf.LazyStringArrayList(hashedID_);
        bitField0_ |= 0x00000001;
       }
    }
    /**
     * <code>repeated string hashedID = 7;</code>
     * @return A list containing the hashedID.
     */
    public com.google.protobuf.ProtocolStringList
        getHashedIDList() {
      return hashedID_.getUnmodifiableView();
    }
    /**
     * <code>repeated string hashedID = 7;</code>
     * @return The count of hashedID.
     */
    public int getHashedIDCount() {
      return hashedID_.size();
    }
    /**
     * <code>repeated string hashedID = 7;</code>
     * @param index The index of the element to return.
     * @return The hashedID at the given index.
     */
    public String getHashedID(int index) {
      return hashedID_.get(index);
    }
    /**
     * <code>repeated string hashedID = 7;</code>
     * @param index The index of the value to return.
     * @return The bytes of the hashedID at the given index.
     */
    public com.google.protobuf.ByteString
        getHashedIDBytes(int index) {
      return hashedID_.getByteString(index);
    }
    /**
     * <code>repeated string hashedID = 7;</code>
     * @param index The index to set the value at.
     * @param value The hashedID to set.
     * @return This builder for chaining.
     */
    public Builder setHashedID(
        int index, String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  ensureHashedIDIsMutable();
      hashedID_.set(index, value);
      onChanged();
      return this;
    }
    /**
     * <code>repeated string hashedID = 7;</code>
     * @param value The hashedID to add.
     * @return This builder for chaining.
     */
    public Builder addHashedID(
        String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  ensureHashedIDIsMutable();
      hashedID_.add(value);
      onChanged();
      return this;
    }
    /**
     * <code>repeated string hashedID = 7;</code>
     * @param values The hashedID to add.
     * @return This builder for chaining.
     */
    public Builder addAllHashedID(
        Iterable<String> values) {
      ensureHashedIDIsMutable();
      com.google.protobuf.AbstractMessageLite.Builder.addAll(
          values, hashedID_);
      onChanged();
      return this;
    }
    /**
     * <code>repeated string hashedID = 7;</code>
     * @return This builder for chaining.
     */
    public Builder clearHashedID() {
      hashedID_ = com.google.protobuf.LazyStringArrayList.EMPTY;
      bitField0_ = (bitField0_ & ~0x00000001);
      onChanged();
      return this;
    }
    /**
     * <code>repeated string hashedID = 7;</code>
     * @param value The bytes of the hashedID to add.
     * @return This builder for chaining.
     */
    public Builder addHashedIDBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  checkByteStringIsUtf8(value);
      ensureHashedIDIsMutable();
      hashedID_.add(value);
      onChanged();
      return this;
    }
    @Override
    public final Builder setUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.setUnknownFields(unknownFields);
    }

    @Override
    public final Builder mergeUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.mergeUnknownFields(unknownFields);
    }


    // @@protoc_insertion_point(builder_scope:UploadRequest)
  }

  // @@protoc_insertion_point(class_scope:UploadRequest)
  private static final com.intel.analytics.zoo.ppml.psi.generated.UploadRequest DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new com.intel.analytics.zoo.ppml.psi.generated.UploadRequest();
  }

  public static com.intel.analytics.zoo.ppml.psi.generated.UploadRequest getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  private static final com.google.protobuf.Parser<com.intel.analytics.zoo.ppml.psi.generated.UploadRequest>
      PARSER = new com.google.protobuf.AbstractParser<com.intel.analytics.zoo.ppml.psi.generated.UploadRequest>() {
    @Override
    public com.intel.analytics.zoo.ppml.psi.generated.UploadRequest parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return new com.intel.analytics.zoo.ppml.psi.generated.UploadRequest(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<com.intel.analytics.zoo.ppml.psi.generated.UploadRequest> parser() {
    return PARSER;
  }

  @Override
  public com.google.protobuf.Parser<com.intel.analytics.zoo.ppml.psi.generated.UploadRequest> getParserForType() {
    return PARSER;
  }

  @Override
  public com.intel.analytics.zoo.ppml.psi.generated.UploadRequest getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

