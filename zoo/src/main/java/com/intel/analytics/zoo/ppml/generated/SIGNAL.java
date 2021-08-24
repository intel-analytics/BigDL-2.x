// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: PSI.proto

package com.intel.analytics.zoo.ppml.generated;

/**
 * Protobuf enum {@code SIGNAL}
 */
public enum SIGNAL
    implements com.google.protobuf.ProtocolMessageEnum {
  /**
   * <code>SUCCESS = 0;</code>
   */
  SUCCESS(0),
  /**
   * <code>WAIT = 1;</code>
   */
  WAIT(1),
  /**
   * <code>TIMEOUT = 2;</code>
   */
  TIMEOUT(2),
  /**
   * <code>EMPTY_INPUT = 3;</code>
   */
  EMPTY_INPUT(3),
  /**
   * <code>ERROR = 4;</code>
   */
  ERROR(4),
  UNRECOGNIZED(-1),
  ;

  /**
   * <code>SUCCESS = 0;</code>
   */
  public static final int SUCCESS_VALUE = 0;
  /**
   * <code>WAIT = 1;</code>
   */
  public static final int WAIT_VALUE = 1;
  /**
   * <code>TIMEOUT = 2;</code>
   */
  public static final int TIMEOUT_VALUE = 2;
  /**
   * <code>EMPTY_INPUT = 3;</code>
   */
  public static final int EMPTY_INPUT_VALUE = 3;
  /**
   * <code>ERROR = 4;</code>
   */
  public static final int ERROR_VALUE = 4;


  public final int getNumber() {
    if (this == UNRECOGNIZED) {
      throw new IllegalArgumentException(
          "Can't get the number of an unknown enum value.");
    }
    return value;
  }

  /**
   * @param value The numeric wire value of the corresponding enum entry.
   * @return The enum associated with the given numeric wire value.
   * @deprecated Use {@link #forNumber(int)} instead.
   */
  @Deprecated
  public static com.intel.analytics.zoo.ppml.psi.generated.SIGNAL valueOf(int value) {
    return forNumber(value);
  }

  /**
   * @param value The numeric wire value of the corresponding enum entry.
   * @return The enum associated with the given numeric wire value.
   */
  public static com.intel.analytics.zoo.ppml.psi.generated.SIGNAL forNumber(int value) {
    switch (value) {
      case 0: return SUCCESS;
      case 1: return WAIT;
      case 2: return TIMEOUT;
      case 3: return EMPTY_INPUT;
      case 4: return ERROR;
      default: return null;
    }
  }

  public static com.google.protobuf.Internal.EnumLiteMap<com.intel.analytics.zoo.ppml.psi.generated.SIGNAL>
      internalGetValueMap() {
    return internalValueMap;
  }
  private static final com.google.protobuf.Internal.EnumLiteMap<
          com.intel.analytics.zoo.ppml.psi.generated.SIGNAL> internalValueMap =
        new com.google.protobuf.Internal.EnumLiteMap<com.intel.analytics.zoo.ppml.psi.generated.SIGNAL>() {
          public com.intel.analytics.zoo.ppml.psi.generated.SIGNAL findValueByNumber(int number) {
            return com.intel.analytics.zoo.ppml.psi.generated.SIGNAL.forNumber(number);
          }
        };

  public final com.google.protobuf.Descriptors.EnumValueDescriptor
      getValueDescriptor() {
    if (this == UNRECOGNIZED) {
      throw new IllegalStateException(
          "Can't get the descriptor of an unrecognized enum value.");
    }
    return getDescriptor().getValues().get(ordinal());
  }
  public final com.google.protobuf.Descriptors.EnumDescriptor
      getDescriptorForType() {
    return getDescriptor();
  }
  public static final com.google.protobuf.Descriptors.EnumDescriptor
      getDescriptor() {
    return com.intel.analytics.zoo.ppml.psi.generated.PSIProto.getDescriptor().getEnumTypes().get(0);
  }

  private static final com.intel.analytics.zoo.ppml.psi.generated.SIGNAL[] VALUES = values();

  public static com.intel.analytics.zoo.ppml.psi.generated.SIGNAL valueOf(
      com.google.protobuf.Descriptors.EnumValueDescriptor desc) {
    if (desc.getType() != getDescriptor()) {
      throw new IllegalArgumentException(
        "EnumValueDescriptor is not for this type.");
    }
    if (desc.getIndex() == -1) {
      return UNRECOGNIZED;
    }
    return VALUES[desc.getIndex()];
  }

  private final int value;

  private SIGNAL(int value) {
    this.value = value;
  }

  // @@protoc_insertion_point(enum_scope:SIGNAL)
}

