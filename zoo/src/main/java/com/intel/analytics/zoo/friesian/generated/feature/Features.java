// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: feature.proto

package com.intel.analytics.zoo.friesian.generated.feature;

/**
 * Protobuf type {@code feature.Features}
 */
public  final class Features extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:feature.Features)
    FeaturesOrBuilder {
private static final long serialVersionUID = 0L;
  // Use Features.newBuilder() to construct.
  private Features(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private Features() {
    b64Feature_ = com.google.protobuf.LazyStringArrayList.EMPTY;
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private Features(
      com.google.protobuf.CodedInputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    this();
    if (extensionRegistry == null) {
      throw new java.lang.NullPointerException();
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
            java.lang.String s = input.readStringRequireUtf8();
            if (!((mutable_bitField0_ & 0x00000001) == 0x00000001)) {
              b64Feature_ = new com.google.protobuf.LazyStringArrayList();
              mutable_bitField0_ |= 0x00000001;
            }
            b64Feature_.add(s);
            break;
          }
          default: {
            if (!parseUnknownFieldProto3(
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
      if (((mutable_bitField0_ & 0x00000001) == 0x00000001)) {
        b64Feature_ = b64Feature_.getUnmodifiableView();
      }
      this.unknownFields = unknownFields.build();
      makeExtensionsImmutable();
    }
  }
  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return FeatureProto.internal_static_feature_Features_descriptor;
  }

  @java.lang.Override
  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return FeatureProto.internal_static_feature_Features_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            Features.class, Features.Builder.class);
  }

  public static final int B64FEATURE_FIELD_NUMBER = 1;
  private com.google.protobuf.LazyStringList b64Feature_;
  /**
   * <code>repeated string b64Feature = 1;</code>
   */
  public com.google.protobuf.ProtocolStringList
      getB64FeatureList() {
    return b64Feature_;
  }
  /**
   * <code>repeated string b64Feature = 1;</code>
   */
  public int getB64FeatureCount() {
    return b64Feature_.size();
  }
  /**
   * <code>repeated string b64Feature = 1;</code>
   */
  public java.lang.String getB64Feature(int index) {
    return b64Feature_.get(index);
  }
  /**
   * <code>repeated string b64Feature = 1;</code>
   */
  public com.google.protobuf.ByteString
      getB64FeatureBytes(int index) {
    return b64Feature_.getByteString(index);
  }

  private byte memoizedIsInitialized = -1;
  @java.lang.Override
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    memoizedIsInitialized = 1;
    return true;
  }

  @java.lang.Override
  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    for (int i = 0; i < b64Feature_.size(); i++) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 1, b64Feature_.getRaw(i));
    }
    unknownFields.writeTo(output);
  }

  @java.lang.Override
  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    {
      int dataSize = 0;
      for (int i = 0; i < b64Feature_.size(); i++) {
        dataSize += computeStringSizeNoTag(b64Feature_.getRaw(i));
      }
      size += dataSize;
      size += 1 * getB64FeatureList().size();
    }
    size += unknownFields.getSerializedSize();
    memoizedSize = size;
    return size;
  }

  @java.lang.Override
  public boolean equals(final java.lang.Object obj) {
    if (obj == this) {
     return true;
    }
    if (!(obj instanceof Features)) {
      return super.equals(obj);
    }
    Features other = (Features) obj;

    boolean result = true;
    result = result && getB64FeatureList()
        .equals(other.getB64FeatureList());
    result = result && unknownFields.equals(other.unknownFields);
    return result;
  }

  @java.lang.Override
  public int hashCode() {
    if (memoizedHashCode != 0) {
      return memoizedHashCode;
    }
    int hash = 41;
    hash = (19 * hash) + getDescriptor().hashCode();
    if (getB64FeatureCount() > 0) {
      hash = (37 * hash) + B64FEATURE_FIELD_NUMBER;
      hash = (53 * hash) + getB64FeatureList().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static Features parseFrom(
      java.nio.ByteBuffer data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static Features parseFrom(
      java.nio.ByteBuffer data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static Features parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static Features parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static Features parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static Features parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static Features parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static Features parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static Features parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static Features parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static Features parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static Features parseFrom(
      com.google.protobuf.CodedInputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }

  @java.lang.Override
  public Builder newBuilderForType() { return newBuilder(); }
  public static Builder newBuilder() {
    return DEFAULT_INSTANCE.toBuilder();
  }
  public static Builder newBuilder(Features prototype) {
    return DEFAULT_INSTANCE.toBuilder().mergeFrom(prototype);
  }
  @java.lang.Override
  public Builder toBuilder() {
    return this == DEFAULT_INSTANCE
        ? new Builder() : new Builder().mergeFrom(this);
  }

  @java.lang.Override
  protected Builder newBuilderForType(
      com.google.protobuf.GeneratedMessageV3.BuilderParent parent) {
    Builder builder = new Builder(parent);
    return builder;
  }
  /**
   * Protobuf type {@code feature.Features}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:feature.Features)
          FeaturesOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return FeatureProto.internal_static_feature_Features_descriptor;
    }

    @java.lang.Override
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return FeatureProto.internal_static_feature_Features_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              Features.class, Features.Builder.class);
    }

    // Construct using com.intel.analytics.zoo.grpc.generated.feature.Features.newBuilder()
    private Builder() {
      maybeForceBuilderInitialization();
    }

    private Builder(
        com.google.protobuf.GeneratedMessageV3.BuilderParent parent) {
      super(parent);
      maybeForceBuilderInitialization();
    }
    private void maybeForceBuilderInitialization() {
      if (com.google.protobuf.GeneratedMessageV3
              .alwaysUseFieldBuilders) {
      }
    }
    @java.lang.Override
    public Builder clear() {
      super.clear();
      b64Feature_ = com.google.protobuf.LazyStringArrayList.EMPTY;
      bitField0_ = (bitField0_ & ~0x00000001);
      return this;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return FeatureProto.internal_static_feature_Features_descriptor;
    }

    @java.lang.Override
    public Features getDefaultInstanceForType() {
      return Features.getDefaultInstance();
    }

    @java.lang.Override
    public Features build() {
      Features result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    @java.lang.Override
    public Features buildPartial() {
      Features result = new Features(this);
      int from_bitField0_ = bitField0_;
      if (((bitField0_ & 0x00000001) == 0x00000001)) {
        b64Feature_ = b64Feature_.getUnmodifiableView();
        bitField0_ = (bitField0_ & ~0x00000001);
      }
      result.b64Feature_ = b64Feature_;
      onBuilt();
      return result;
    }

    @java.lang.Override
    public Builder clone() {
      return (Builder) super.clone();
    }
    @java.lang.Override
    public Builder setField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        java.lang.Object value) {
      return (Builder) super.setField(field, value);
    }
    @java.lang.Override
    public Builder clearField(
        com.google.protobuf.Descriptors.FieldDescriptor field) {
      return (Builder) super.clearField(field);
    }
    @java.lang.Override
    public Builder clearOneof(
        com.google.protobuf.Descriptors.OneofDescriptor oneof) {
      return (Builder) super.clearOneof(oneof);
    }
    @java.lang.Override
    public Builder setRepeatedField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        int index, java.lang.Object value) {
      return (Builder) super.setRepeatedField(field, index, value);
    }
    @java.lang.Override
    public Builder addRepeatedField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        java.lang.Object value) {
      return (Builder) super.addRepeatedField(field, value);
    }
    @java.lang.Override
    public Builder mergeFrom(com.google.protobuf.Message other) {
      if (other instanceof Features) {
        return mergeFrom((Features)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(Features other) {
      if (other == Features.getDefaultInstance()) return this;
      if (!other.b64Feature_.isEmpty()) {
        if (b64Feature_.isEmpty()) {
          b64Feature_ = other.b64Feature_;
          bitField0_ = (bitField0_ & ~0x00000001);
        } else {
          ensureB64FeatureIsMutable();
          b64Feature_.addAll(other.b64Feature_);
        }
        onChanged();
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    @java.lang.Override
    public final boolean isInitialized() {
      return true;
    }

    @java.lang.Override
    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      Features parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (Features) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private com.google.protobuf.LazyStringList b64Feature_ = com.google.protobuf.LazyStringArrayList.EMPTY;
    private void ensureB64FeatureIsMutable() {
      if (!((bitField0_ & 0x00000001) == 0x00000001)) {
        b64Feature_ = new com.google.protobuf.LazyStringArrayList(b64Feature_);
        bitField0_ |= 0x00000001;
       }
    }
    /**
     * <code>repeated string b64Feature = 1;</code>
     */
    public com.google.protobuf.ProtocolStringList
        getB64FeatureList() {
      return b64Feature_.getUnmodifiableView();
    }
    /**
     * <code>repeated string b64Feature = 1;</code>
     */
    public int getB64FeatureCount() {
      return b64Feature_.size();
    }
    /**
     * <code>repeated string b64Feature = 1;</code>
     */
    public java.lang.String getB64Feature(int index) {
      return b64Feature_.get(index);
    }
    /**
     * <code>repeated string b64Feature = 1;</code>
     */
    public com.google.protobuf.ByteString
        getB64FeatureBytes(int index) {
      return b64Feature_.getByteString(index);
    }
    /**
     * <code>repeated string b64Feature = 1;</code>
     */
    public Builder setB64Feature(
        int index, java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  ensureB64FeatureIsMutable();
      b64Feature_.set(index, value);
      onChanged();
      return this;
    }
    /**
     * <code>repeated string b64Feature = 1;</code>
     */
    public Builder addB64Feature(
        java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  ensureB64FeatureIsMutable();
      b64Feature_.add(value);
      onChanged();
      return this;
    }
    /**
     * <code>repeated string b64Feature = 1;</code>
     */
    public Builder addAllB64Feature(
        java.lang.Iterable<java.lang.String> values) {
      ensureB64FeatureIsMutable();
      com.google.protobuf.AbstractMessageLite.Builder.addAll(
          values, b64Feature_);
      onChanged();
      return this;
    }
    /**
     * <code>repeated string b64Feature = 1;</code>
     */
    public Builder clearB64Feature() {
      b64Feature_ = com.google.protobuf.LazyStringArrayList.EMPTY;
      bitField0_ = (bitField0_ & ~0x00000001);
      onChanged();
      return this;
    }
    /**
     * <code>repeated string b64Feature = 1;</code>
     */
    public Builder addB64FeatureBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  checkByteStringIsUtf8(value);
      ensureB64FeatureIsMutable();
      b64Feature_.add(value);
      onChanged();
      return this;
    }
    @java.lang.Override
    public final Builder setUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.setUnknownFieldsProto3(unknownFields);
    }

    @java.lang.Override
    public final Builder mergeUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.mergeUnknownFields(unknownFields);
    }


    // @@protoc_insertion_point(builder_scope:feature.Features)
  }

  // @@protoc_insertion_point(class_scope:feature.Features)
  private static final Features DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new Features();
  }

  public static Features getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  private static final com.google.protobuf.Parser<Features>
      PARSER = new com.google.protobuf.AbstractParser<Features>() {
    @java.lang.Override
    public Features parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return new Features(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<Features> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<Features> getParserForType() {
    return PARSER;
  }

  @java.lang.Override
  public Features getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

