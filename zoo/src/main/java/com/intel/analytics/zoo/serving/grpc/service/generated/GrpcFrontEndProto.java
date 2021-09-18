// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: frontEndGRPC.proto

package com.intel.analytics.zoo.serving.grpc.service.generated;

public final class GrpcFrontEndProto {
  private GrpcFrontEndProto() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_grpc_Empty_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_grpc_Empty_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_grpc_GetModelsWithNameReq_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_grpc_GetModelsWithNameReq_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_grpc_GetModelsWithNameAndVersionReq_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_grpc_GetModelsWithNameAndVersionReq_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_grpc_StringReply_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_grpc_StringReply_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_grpc_MetricsReply_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_grpc_MetricsReply_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_grpc_MetricsReply_Metric_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_grpc_MetricsReply_Metric_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_grpc_InferenceModelGRPCMetaData_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_grpc_InferenceModelGRPCMetaData_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_grpc_ClusterServingGRPCMetaData_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_grpc_ClusterServingGRPCMetaData_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_grpc_ModelsReply_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_grpc_ModelsReply_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_grpc_PredictReq_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_grpc_PredictReq_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_grpc_PredictReply_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_grpc_PredictReply_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    String[] descriptorData = {
      "\n\022frontEndGRPC.proto\022\004grpc\"\007\n\005Empty\")\n\024G" +
      "etModelsWithNameReq\022\021\n\tmodelName\030\001 \001(\t\"I" +
      "\n\036GetModelsWithNameAndVersionReq\022\021\n\tmode" +
      "lName\030\001 \001(\t\022\024\n\014modelVersion\030\002 \001(\t\"\036\n\013Str" +
      "ingReply\022\017\n\007message\030\001 \001(\t\"\265\002\n\014MetricsRep" +
      "ly\022*\n\007metrics\030\001 \003(\0132\031.grpc.MetricsReply." +
      "Metric\032\370\001\n\006Metric\022\014\n\004name\030\001 \001(\t\022\r\n\005count" +
      "\030\002 \001(\003\022\020\n\010meanRate\030\003 \001(\001\022\013\n\003min\030\004 \001(\003\022\013\n" +
      "\003max\030\005 \001(\003\022\014\n\004mean\030\006 \001(\001\022\016\n\006median\030\007 \001(\001" +
      "\022\016\n\006stdDev\030\010 \001(\001\022\026\n\016Percentile75th\030\t \001(\001",
      "\022\026\n\016Percentile95th\030\n \001(\001\022\026\n\016Percentile98" +
      "th\030\013 \001(\001\022\026\n\016Percentile99th\030\014 \001(\001\022\027\n\017Perc" +
      "entile999th\030\r \001(\001\"\307\001\n\032InferenceModelGRPC" +
      "MetaData\022\021\n\tmodelName\030\001 \001(\t\022\024\n\014modelVers" +
      "ion\030\002 \001(\t\022\021\n\tmodelPath\030\003 \001(\t\022\021\n\tmodelTyp" +
      "e\030\004 \001(\t\022\022\n\nweightPath\030\005 \001(\t\022\032\n\022modelConC" +
      "urrentNum\030\006 \001(\005\022\030\n\020inputCompileType\030\007 \001(" +
      "\t\022\020\n\010features\030\010 \001(\t\"\260\002\n\032ClusterServingGR" +
      "PCMetaData\022\021\n\tmodelName\030\001 \001(\t\022\024\n\014modelVe" +
      "rsion\030\002 \001(\t\022\021\n\tredisHost\030\003 \001(\t\022\021\n\tredisP",
      "ort\030\004 \001(\t\022\027\n\017redisInputQueue\030\005 \001(\t\022\030\n\020re" +
      "disOutputQueue\030\006 \001(\t\022\022\n\ntimeWindow\030\007 \001(\005" +
      "\022\023\n\013countWindow\030\010 \001(\005\022\032\n\022redisSecureEnab" +
      "led\030\t \001(\010\022\033\n\023redisTrustStorePath\030\n \001(\t\022\034" +
      "\n\024redisTrustStoreToken\030\013 \001(\t\022\020\n\010features" +
      "\030\014 \001(\t\"\223\001\n\013ModelsReply\022A\n\027inferenceModel" +
      "MetaDatas\030\001 \003(\0132 .grpc.InferenceModelGRP" +
      "CMetaData\022A\n\027clusterServingMetaDatas\030\002 \003" +
      "(\0132 .grpc.ClusterServingGRPCMetaData\"D\n\n" +
      "PredictReq\022\021\n\tmodelName\030\001 \001(\t\022\024\n\014modelVe",
      "rsion\030\002 \001(\t\022\r\n\005input\030\003 \001(\t\" \n\014PredictRep" +
      "ly\022\020\n\010response\030\001 \001(\t2\365\002\n\023FrontEndGRPCSer" +
      "vice\022(\n\004Ping\022\013.grpc.Empty\032\021.grpc.StringR" +
      "eply\"\000\022/\n\nGetMetrics\022\013.grpc.Empty\032\022.grpc" +
      ".MetricsReply\"\000\0220\n\014GetAllModels\022\013.grpc.E" +
      "mpty\032\021.grpc.ModelsReply\"\000\022D\n\021GetModelsWi" +
      "thName\022\032.grpc.GetModelsWithNameReq\032\021.grp" +
      "c.ModelsReply\"\000\022X\n\033GetModelsWithNameAndV" +
      "ersion\022$.grpc.GetModelsWithNameAndVersio" +
      "nReq\032\021.grpc.ModelsReply\"\000\0221\n\007Predict\022\020.g",
      "rpc.PredictReq\032\022.grpc.PredictReply\"\000BT\n6" +
      "com.intel.analytics.zoo.serving.grpc.ser" +
      "vice.generatedB\021GrpcFrontEndProtoP\001\242\002\004gr" +
      "pcb\006proto3"
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
        }, assigner);
    internal_static_grpc_Empty_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_grpc_Empty_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_grpc_Empty_descriptor,
        new String[] { });
    internal_static_grpc_GetModelsWithNameReq_descriptor =
      getDescriptor().getMessageTypes().get(1);
    internal_static_grpc_GetModelsWithNameReq_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_grpc_GetModelsWithNameReq_descriptor,
        new String[] { "ModelName", });
    internal_static_grpc_GetModelsWithNameAndVersionReq_descriptor =
      getDescriptor().getMessageTypes().get(2);
    internal_static_grpc_GetModelsWithNameAndVersionReq_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_grpc_GetModelsWithNameAndVersionReq_descriptor,
        new String[] { "ModelName", "ModelVersion", });
    internal_static_grpc_StringReply_descriptor =
      getDescriptor().getMessageTypes().get(3);
    internal_static_grpc_StringReply_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_grpc_StringReply_descriptor,
        new String[] { "Message", });
    internal_static_grpc_MetricsReply_descriptor =
      getDescriptor().getMessageTypes().get(4);
    internal_static_grpc_MetricsReply_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_grpc_MetricsReply_descriptor,
        new String[] { "Metrics", });
    internal_static_grpc_MetricsReply_Metric_descriptor =
      internal_static_grpc_MetricsReply_descriptor.getNestedTypes().get(0);
    internal_static_grpc_MetricsReply_Metric_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_grpc_MetricsReply_Metric_descriptor,
        new String[] { "Name", "Count", "MeanRate", "Min", "Max", "Mean", "Median", "StdDev", "Percentile75Th", "Percentile95Th", "Percentile98Th", "Percentile99Th", "Percentile999Th", });
    internal_static_grpc_InferenceModelGRPCMetaData_descriptor =
      getDescriptor().getMessageTypes().get(5);
    internal_static_grpc_InferenceModelGRPCMetaData_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_grpc_InferenceModelGRPCMetaData_descriptor,
        new String[] { "ModelName", "ModelVersion", "ModelPath", "ModelType", "WeightPath", "ModelConCurrentNum", "InputCompileType", "Features", });
    internal_static_grpc_ClusterServingGRPCMetaData_descriptor =
      getDescriptor().getMessageTypes().get(6);
    internal_static_grpc_ClusterServingGRPCMetaData_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_grpc_ClusterServingGRPCMetaData_descriptor,
        new String[] { "ModelName", "ModelVersion", "RedisHost", "RedisPort", "RedisInputQueue", "RedisOutputQueue", "TimeWindow", "CountWindow", "RedisSecureEnabled", "RedisTrustStorePath", "RedisTrustStoreToken", "Features", });
    internal_static_grpc_ModelsReply_descriptor =
      getDescriptor().getMessageTypes().get(7);
    internal_static_grpc_ModelsReply_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_grpc_ModelsReply_descriptor,
        new String[] { "InferenceModelMetaDatas", "ClusterServingMetaDatas", });
    internal_static_grpc_PredictReq_descriptor =
      getDescriptor().getMessageTypes().get(8);
    internal_static_grpc_PredictReq_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_grpc_PredictReq_descriptor,
        new String[] { "ModelName", "ModelVersion", "Input", });
    internal_static_grpc_PredictReply_descriptor =
      getDescriptor().getMessageTypes().get(9);
    internal_static_grpc_PredictReply_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_grpc_PredictReply_descriptor,
        new String[] { "Response", });
  }

  // @@protoc_insertion_point(outer_class_scope)
}
