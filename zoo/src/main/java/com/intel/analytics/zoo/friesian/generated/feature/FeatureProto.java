// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: feature.proto

package com.intel.analytics.zoo.friesian.generated.feature;

public final class FeatureProto {
  private FeatureProto() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_feature_IDs_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_feature_IDs_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_feature_Features_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_feature_Features_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_feature_ServerMessage_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_feature_ServerMessage_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n\rfeature.proto\022\007feature\032\033google/protobu" +
      "f/empty.proto\"\021\n\003IDs\022\n\n\002ID\030\001 \003(\005\"\036\n\010Feat" +
      "ures\022\022\n\nb64Feature\030\001 \003(\t\"\034\n\rServerMessag" +
      "e\022\013\n\003str\030\001 \001(\t2\274\001\n\016FeatureService\0224\n\017get" +
      "UserFeatures\022\014.feature.IDs\032\021.feature.Fea" +
      "tures\"\000\0224\n\017getItemFeatures\022\014.feature.IDs" +
      "\032\021.feature.Features\"\000\022>\n\ngetMetrics\022\026.go" +
      "ogle.protobuf.Empty\032\026.feature.ServerMess" +
      "age\"\000BF\n.com.intel.analytics.zoo.grpc.ge" +
      "nerated.featureB\014FeatureProtoP\001\242\002\003RTGb\006p" +
      "roto3"
    };
    com.google.protobuf.Descriptors.FileDescriptor.InternalDescriptorAssigner assigner =
        new com.google.protobuf.Descriptors.FileDescriptor.    InternalDescriptorAssigner() {
          public com.google.protobuf.ExtensionRegistry assignDescriptors(
              com.google.protobuf.Descriptors.FileDescriptor root) {
            descriptor = root;
            return null;
          }
        };
    com.google.protobuf.Descriptors.FileDescriptor
      .internalBuildGeneratedFileFrom(descriptorData,
        new com.google.protobuf.Descriptors.FileDescriptor[] {
          com.google.protobuf.EmptyProto.getDescriptor(),
        }, assigner);
    internal_static_feature_IDs_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_feature_IDs_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_feature_IDs_descriptor,
        new java.lang.String[] { "ID", });
    internal_static_feature_Features_descriptor =
      getDescriptor().getMessageTypes().get(1);
    internal_static_feature_Features_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_feature_Features_descriptor,
        new java.lang.String[] { "B64Feature", });
    internal_static_feature_ServerMessage_descriptor =
      getDescriptor().getMessageTypes().get(2);
    internal_static_feature_ServerMessage_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_feature_ServerMessage_descriptor,
        new java.lang.String[] { "Str", });
    com.google.protobuf.EmptyProto.getDescriptor();
  }

  // @@protoc_insertion_point(outer_class_scope)
}
