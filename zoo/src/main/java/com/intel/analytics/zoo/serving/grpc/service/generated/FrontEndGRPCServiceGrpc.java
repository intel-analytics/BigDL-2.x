package com.intel.analytics.zoo.serving.grpc.service.generated;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.37.0)",
    comments = "Source: frontEndGRPC.proto")
public final class FrontEndGRPCServiceGrpc {

  private FrontEndGRPCServiceGrpc() {}

  public static final String SERVICE_NAME = "grpc.FrontEndGRPCService";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.zoo.serving.grpc.service.generated.Empty,
      com.intel.analytics.zoo.serving.grpc.service.generated.StringReply> getPingMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "Ping",
      requestType = com.intel.analytics.zoo.serving.grpc.service.generated.Empty.class,
      responseType = com.intel.analytics.zoo.serving.grpc.service.generated.StringReply.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.zoo.serving.grpc.service.generated.Empty,
      com.intel.analytics.zoo.serving.grpc.service.generated.StringReply> getPingMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.zoo.serving.grpc.service.generated.Empty, com.intel.analytics.zoo.serving.grpc.service.generated.StringReply> getPingMethod;
    if ((getPingMethod = FrontEndGRPCServiceGrpc.getPingMethod) == null) {
      synchronized (FrontEndGRPCServiceGrpc.class) {
        if ((getPingMethod = FrontEndGRPCServiceGrpc.getPingMethod) == null) {
          FrontEndGRPCServiceGrpc.getPingMethod = getPingMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.zoo.serving.grpc.service.generated.Empty, com.intel.analytics.zoo.serving.grpc.service.generated.StringReply>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "Ping"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.zoo.serving.grpc.service.generated.Empty.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.zoo.serving.grpc.service.generated.StringReply.getDefaultInstance()))
              .setSchemaDescriptor(new FrontEndGRPCServiceMethodDescriptorSupplier("Ping"))
              .build();
        }
      }
    }
    return getPingMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.zoo.serving.grpc.service.generated.Empty,
      com.intel.analytics.zoo.serving.grpc.service.generated.MetricsReply> getGetMetricsMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "GetMetrics",
      requestType = com.intel.analytics.zoo.serving.grpc.service.generated.Empty.class,
      responseType = com.intel.analytics.zoo.serving.grpc.service.generated.MetricsReply.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.zoo.serving.grpc.service.generated.Empty,
      com.intel.analytics.zoo.serving.grpc.service.generated.MetricsReply> getGetMetricsMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.zoo.serving.grpc.service.generated.Empty, com.intel.analytics.zoo.serving.grpc.service.generated.MetricsReply> getGetMetricsMethod;
    if ((getGetMetricsMethod = FrontEndGRPCServiceGrpc.getGetMetricsMethod) == null) {
      synchronized (FrontEndGRPCServiceGrpc.class) {
        if ((getGetMetricsMethod = FrontEndGRPCServiceGrpc.getGetMetricsMethod) == null) {
          FrontEndGRPCServiceGrpc.getGetMetricsMethod = getGetMetricsMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.zoo.serving.grpc.service.generated.Empty, com.intel.analytics.zoo.serving.grpc.service.generated.MetricsReply>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "GetMetrics"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.zoo.serving.grpc.service.generated.Empty.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.zoo.serving.grpc.service.generated.MetricsReply.getDefaultInstance()))
              .setSchemaDescriptor(new FrontEndGRPCServiceMethodDescriptorSupplier("GetMetrics"))
              .build();
        }
      }
    }
    return getGetMetricsMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.zoo.serving.grpc.service.generated.Empty,
      com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply> getGetAllModelsMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "GetAllModels",
      requestType = com.intel.analytics.zoo.serving.grpc.service.generated.Empty.class,
      responseType = com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.zoo.serving.grpc.service.generated.Empty,
      com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply> getGetAllModelsMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.zoo.serving.grpc.service.generated.Empty, com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply> getGetAllModelsMethod;
    if ((getGetAllModelsMethod = FrontEndGRPCServiceGrpc.getGetAllModelsMethod) == null) {
      synchronized (FrontEndGRPCServiceGrpc.class) {
        if ((getGetAllModelsMethod = FrontEndGRPCServiceGrpc.getGetAllModelsMethod) == null) {
          FrontEndGRPCServiceGrpc.getGetAllModelsMethod = getGetAllModelsMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.zoo.serving.grpc.service.generated.Empty, com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "GetAllModels"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.zoo.serving.grpc.service.generated.Empty.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply.getDefaultInstance()))
              .setSchemaDescriptor(new FrontEndGRPCServiceMethodDescriptorSupplier("GetAllModels"))
              .build();
        }
      }
    }
    return getGetAllModelsMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.zoo.serving.grpc.service.generated.GetModelsWithNameReq,
      com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply> getGetModelsWithNameMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "GetModelsWithName",
      requestType = com.intel.analytics.zoo.serving.grpc.service.generated.GetModelsWithNameReq.class,
      responseType = com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.zoo.serving.grpc.service.generated.GetModelsWithNameReq,
      com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply> getGetModelsWithNameMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.zoo.serving.grpc.service.generated.GetModelsWithNameReq, com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply> getGetModelsWithNameMethod;
    if ((getGetModelsWithNameMethod = FrontEndGRPCServiceGrpc.getGetModelsWithNameMethod) == null) {
      synchronized (FrontEndGRPCServiceGrpc.class) {
        if ((getGetModelsWithNameMethod = FrontEndGRPCServiceGrpc.getGetModelsWithNameMethod) == null) {
          FrontEndGRPCServiceGrpc.getGetModelsWithNameMethod = getGetModelsWithNameMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.zoo.serving.grpc.service.generated.GetModelsWithNameReq, com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "GetModelsWithName"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.zoo.serving.grpc.service.generated.GetModelsWithNameReq.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply.getDefaultInstance()))
              .setSchemaDescriptor(new FrontEndGRPCServiceMethodDescriptorSupplier("GetModelsWithName"))
              .build();
        }
      }
    }
    return getGetModelsWithNameMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.zoo.serving.grpc.service.generated.GetModelsWithNameAndVersionReq,
      com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply> getGetModelsWithNameAndVersionMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "GetModelsWithNameAndVersion",
      requestType = com.intel.analytics.zoo.serving.grpc.service.generated.GetModelsWithNameAndVersionReq.class,
      responseType = com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.zoo.serving.grpc.service.generated.GetModelsWithNameAndVersionReq,
      com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply> getGetModelsWithNameAndVersionMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.zoo.serving.grpc.service.generated.GetModelsWithNameAndVersionReq, com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply> getGetModelsWithNameAndVersionMethod;
    if ((getGetModelsWithNameAndVersionMethod = FrontEndGRPCServiceGrpc.getGetModelsWithNameAndVersionMethod) == null) {
      synchronized (FrontEndGRPCServiceGrpc.class) {
        if ((getGetModelsWithNameAndVersionMethod = FrontEndGRPCServiceGrpc.getGetModelsWithNameAndVersionMethod) == null) {
          FrontEndGRPCServiceGrpc.getGetModelsWithNameAndVersionMethod = getGetModelsWithNameAndVersionMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.zoo.serving.grpc.service.generated.GetModelsWithNameAndVersionReq, com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "GetModelsWithNameAndVersion"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.zoo.serving.grpc.service.generated.GetModelsWithNameAndVersionReq.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply.getDefaultInstance()))
              .setSchemaDescriptor(new FrontEndGRPCServiceMethodDescriptorSupplier("GetModelsWithNameAndVersion"))
              .build();
        }
      }
    }
    return getGetModelsWithNameAndVersionMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.zoo.serving.grpc.service.generated.PredictReq,
      com.intel.analytics.zoo.serving.grpc.service.generated.PredictReply> getPredictMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "Predict",
      requestType = com.intel.analytics.zoo.serving.grpc.service.generated.PredictReq.class,
      responseType = com.intel.analytics.zoo.serving.grpc.service.generated.PredictReply.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.zoo.serving.grpc.service.generated.PredictReq,
      com.intel.analytics.zoo.serving.grpc.service.generated.PredictReply> getPredictMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.zoo.serving.grpc.service.generated.PredictReq, com.intel.analytics.zoo.serving.grpc.service.generated.PredictReply> getPredictMethod;
    if ((getPredictMethod = FrontEndGRPCServiceGrpc.getPredictMethod) == null) {
      synchronized (FrontEndGRPCServiceGrpc.class) {
        if ((getPredictMethod = FrontEndGRPCServiceGrpc.getPredictMethod) == null) {
          FrontEndGRPCServiceGrpc.getPredictMethod = getPredictMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.zoo.serving.grpc.service.generated.PredictReq, com.intel.analytics.zoo.serving.grpc.service.generated.PredictReply>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "Predict"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.zoo.serving.grpc.service.generated.PredictReq.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.zoo.serving.grpc.service.generated.PredictReply.getDefaultInstance()))
              .setSchemaDescriptor(new FrontEndGRPCServiceMethodDescriptorSupplier("Predict"))
              .build();
        }
      }
    }
    return getPredictMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static FrontEndGRPCServiceStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<FrontEndGRPCServiceStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<FrontEndGRPCServiceStub>() {
        @Override
        public FrontEndGRPCServiceStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new FrontEndGRPCServiceStub(channel, callOptions);
        }
      };
    return FrontEndGRPCServiceStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static FrontEndGRPCServiceBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<FrontEndGRPCServiceBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<FrontEndGRPCServiceBlockingStub>() {
        @Override
        public FrontEndGRPCServiceBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new FrontEndGRPCServiceBlockingStub(channel, callOptions);
        }
      };
    return FrontEndGRPCServiceBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static FrontEndGRPCServiceFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<FrontEndGRPCServiceFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<FrontEndGRPCServiceFutureStub>() {
        @Override
        public FrontEndGRPCServiceFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new FrontEndGRPCServiceFutureStub(channel, callOptions);
        }
      };
    return FrontEndGRPCServiceFutureStub.newStub(factory, channel);
  }

  /**
   */
  public static abstract class FrontEndGRPCServiceImplBase implements io.grpc.BindableService {

    /**
     * <pre>
     *ping port
     * </pre>
     */
    public void ping(com.intel.analytics.zoo.serving.grpc.service.generated.Empty request,
        io.grpc.stub.StreamObserver<com.intel.analytics.zoo.serving.grpc.service.generated.StringReply> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getPingMethod(), responseObserver);
    }

    /**
     * <pre>
     *metrics port
     * </pre>
     */
    public void getMetrics(com.intel.analytics.zoo.serving.grpc.service.generated.Empty request,
        io.grpc.stub.StreamObserver<com.intel.analytics.zoo.serving.grpc.service.generated.MetricsReply> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetMetricsMethod(), responseObserver);
    }

    /**
     * <pre>
     *get models port
     * </pre>
     */
    public void getAllModels(com.intel.analytics.zoo.serving.grpc.service.generated.Empty request,
        io.grpc.stub.StreamObserver<com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetAllModelsMethod(), responseObserver);
    }

    /**
     * <pre>
     *get models with model name port
     * </pre>
     */
    public void getModelsWithName(com.intel.analytics.zoo.serving.grpc.service.generated.GetModelsWithNameReq request,
        io.grpc.stub.StreamObserver<com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetModelsWithNameMethod(), responseObserver);
    }

    /**
     * <pre>
     *get models with model name port
     * </pre>
     */
    public void getModelsWithNameAndVersion(com.intel.analytics.zoo.serving.grpc.service.generated.GetModelsWithNameAndVersionReq request,
        io.grpc.stub.StreamObserver<com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetModelsWithNameAndVersionMethod(), responseObserver);
    }

    /**
     * <pre>
     *predict
     * </pre>
     */
    public void predict(com.intel.analytics.zoo.serving.grpc.service.generated.PredictReq request,
        io.grpc.stub.StreamObserver<com.intel.analytics.zoo.serving.grpc.service.generated.PredictReply> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getPredictMethod(), responseObserver);
    }

    @Override public final io.grpc.ServerServiceDefinition bindService() {
      return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
          .addMethod(
            getPingMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.zoo.serving.grpc.service.generated.Empty,
                com.intel.analytics.zoo.serving.grpc.service.generated.StringReply>(
                  this, METHODID_PING)))
          .addMethod(
            getGetMetricsMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.zoo.serving.grpc.service.generated.Empty,
                com.intel.analytics.zoo.serving.grpc.service.generated.MetricsReply>(
                  this, METHODID_GET_METRICS)))
          .addMethod(
            getGetAllModelsMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.zoo.serving.grpc.service.generated.Empty,
                com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply>(
                  this, METHODID_GET_ALL_MODELS)))
          .addMethod(
            getGetModelsWithNameMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.zoo.serving.grpc.service.generated.GetModelsWithNameReq,
                com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply>(
                  this, METHODID_GET_MODELS_WITH_NAME)))
          .addMethod(
            getGetModelsWithNameAndVersionMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.zoo.serving.grpc.service.generated.GetModelsWithNameAndVersionReq,
                com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply>(
                  this, METHODID_GET_MODELS_WITH_NAME_AND_VERSION)))
          .addMethod(
            getPredictMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.zoo.serving.grpc.service.generated.PredictReq,
                com.intel.analytics.zoo.serving.grpc.service.generated.PredictReply>(
                  this, METHODID_PREDICT)))
          .build();
    }
  }

  /**
   */
  public static final class FrontEndGRPCServiceStub extends io.grpc.stub.AbstractAsyncStub<FrontEndGRPCServiceStub> {
    private FrontEndGRPCServiceStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected FrontEndGRPCServiceStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new FrontEndGRPCServiceStub(channel, callOptions);
    }

    /**
     * <pre>
     *ping port
     * </pre>
     */
    public void ping(com.intel.analytics.zoo.serving.grpc.service.generated.Empty request,
        io.grpc.stub.StreamObserver<com.intel.analytics.zoo.serving.grpc.service.generated.StringReply> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getPingMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     *metrics port
     * </pre>
     */
    public void getMetrics(com.intel.analytics.zoo.serving.grpc.service.generated.Empty request,
        io.grpc.stub.StreamObserver<com.intel.analytics.zoo.serving.grpc.service.generated.MetricsReply> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGetMetricsMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     *get models port
     * </pre>
     */
    public void getAllModels(com.intel.analytics.zoo.serving.grpc.service.generated.Empty request,
        io.grpc.stub.StreamObserver<com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGetAllModelsMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     *get models with model name port
     * </pre>
     */
    public void getModelsWithName(com.intel.analytics.zoo.serving.grpc.service.generated.GetModelsWithNameReq request,
        io.grpc.stub.StreamObserver<com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGetModelsWithNameMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     *get models with model name port
     * </pre>
     */
    public void getModelsWithNameAndVersion(com.intel.analytics.zoo.serving.grpc.service.generated.GetModelsWithNameAndVersionReq request,
        io.grpc.stub.StreamObserver<com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGetModelsWithNameAndVersionMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     *predict
     * </pre>
     */
    public void predict(com.intel.analytics.zoo.serving.grpc.service.generated.PredictReq request,
        io.grpc.stub.StreamObserver<com.intel.analytics.zoo.serving.grpc.service.generated.PredictReply> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getPredictMethod(), getCallOptions()), request, responseObserver);
    }
  }

  /**
   */
  public static final class FrontEndGRPCServiceBlockingStub extends io.grpc.stub.AbstractBlockingStub<FrontEndGRPCServiceBlockingStub> {
    private FrontEndGRPCServiceBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected FrontEndGRPCServiceBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new FrontEndGRPCServiceBlockingStub(channel, callOptions);
    }

    /**
     * <pre>
     *ping port
     * </pre>
     */
    public com.intel.analytics.zoo.serving.grpc.service.generated.StringReply ping(com.intel.analytics.zoo.serving.grpc.service.generated.Empty request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getPingMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     *metrics port
     * </pre>
     */
    public com.intel.analytics.zoo.serving.grpc.service.generated.MetricsReply getMetrics(com.intel.analytics.zoo.serving.grpc.service.generated.Empty request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetMetricsMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     *get models port
     * </pre>
     */
    public com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply getAllModels(com.intel.analytics.zoo.serving.grpc.service.generated.Empty request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetAllModelsMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     *get models with model name port
     * </pre>
     */
    public com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply getModelsWithName(com.intel.analytics.zoo.serving.grpc.service.generated.GetModelsWithNameReq request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetModelsWithNameMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     *get models with model name port
     * </pre>
     */
    public com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply getModelsWithNameAndVersion(com.intel.analytics.zoo.serving.grpc.service.generated.GetModelsWithNameAndVersionReq request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetModelsWithNameAndVersionMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     *predict
     * </pre>
     */
    public com.intel.analytics.zoo.serving.grpc.service.generated.PredictReply predict(com.intel.analytics.zoo.serving.grpc.service.generated.PredictReq request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getPredictMethod(), getCallOptions(), request);
    }
  }

  /**
   */
  public static final class FrontEndGRPCServiceFutureStub extends io.grpc.stub.AbstractFutureStub<FrontEndGRPCServiceFutureStub> {
    private FrontEndGRPCServiceFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected FrontEndGRPCServiceFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new FrontEndGRPCServiceFutureStub(channel, callOptions);
    }

    /**
     * <pre>
     *ping port
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.zoo.serving.grpc.service.generated.StringReply> ping(
        com.intel.analytics.zoo.serving.grpc.service.generated.Empty request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getPingMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     *metrics port
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.zoo.serving.grpc.service.generated.MetricsReply> getMetrics(
        com.intel.analytics.zoo.serving.grpc.service.generated.Empty request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGetMetricsMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     *get models port
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply> getAllModels(
        com.intel.analytics.zoo.serving.grpc.service.generated.Empty request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGetAllModelsMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     *get models with model name port
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply> getModelsWithName(
        com.intel.analytics.zoo.serving.grpc.service.generated.GetModelsWithNameReq request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGetModelsWithNameMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     *get models with model name port
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply> getModelsWithNameAndVersion(
        com.intel.analytics.zoo.serving.grpc.service.generated.GetModelsWithNameAndVersionReq request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGetModelsWithNameAndVersionMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     *predict
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.zoo.serving.grpc.service.generated.PredictReply> predict(
        com.intel.analytics.zoo.serving.grpc.service.generated.PredictReq request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getPredictMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_PING = 0;
  private static final int METHODID_GET_METRICS = 1;
  private static final int METHODID_GET_ALL_MODELS = 2;
  private static final int METHODID_GET_MODELS_WITH_NAME = 3;
  private static final int METHODID_GET_MODELS_WITH_NAME_AND_VERSION = 4;
  private static final int METHODID_PREDICT = 5;

  private static final class MethodHandlers<Req, Resp> implements
      io.grpc.stub.ServerCalls.UnaryMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ServerStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ClientStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.BidiStreamingMethod<Req, Resp> {
    private final FrontEndGRPCServiceImplBase serviceImpl;
    private final int methodId;

    MethodHandlers(FrontEndGRPCServiceImplBase serviceImpl, int methodId) {
      this.serviceImpl = serviceImpl;
      this.methodId = methodId;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void invoke(Req request, io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_PING:
          serviceImpl.ping((com.intel.analytics.zoo.serving.grpc.service.generated.Empty) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.zoo.serving.grpc.service.generated.StringReply>) responseObserver);
          break;
        case METHODID_GET_METRICS:
          serviceImpl.getMetrics((com.intel.analytics.zoo.serving.grpc.service.generated.Empty) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.zoo.serving.grpc.service.generated.MetricsReply>) responseObserver);
          break;
        case METHODID_GET_ALL_MODELS:
          serviceImpl.getAllModels((com.intel.analytics.zoo.serving.grpc.service.generated.Empty) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply>) responseObserver);
          break;
        case METHODID_GET_MODELS_WITH_NAME:
          serviceImpl.getModelsWithName((com.intel.analytics.zoo.serving.grpc.service.generated.GetModelsWithNameReq) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply>) responseObserver);
          break;
        case METHODID_GET_MODELS_WITH_NAME_AND_VERSION:
          serviceImpl.getModelsWithNameAndVersion((com.intel.analytics.zoo.serving.grpc.service.generated.GetModelsWithNameAndVersionReq) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.zoo.serving.grpc.service.generated.ModelsReply>) responseObserver);
          break;
        case METHODID_PREDICT:
          serviceImpl.predict((com.intel.analytics.zoo.serving.grpc.service.generated.PredictReq) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.zoo.serving.grpc.service.generated.PredictReply>) responseObserver);
          break;
        default:
          throw new AssertionError();
      }
    }

    @Override
    @SuppressWarnings("unchecked")
    public io.grpc.stub.StreamObserver<Req> invoke(
        io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        default:
          throw new AssertionError();
      }
    }
  }

  private static abstract class FrontEndGRPCServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    FrontEndGRPCServiceBaseDescriptorSupplier() {}

    @Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return com.intel.analytics.zoo.serving.grpc.service.generated.GrpcFrontEndProto.getDescriptor();
    }

    @Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("FrontEndGRPCService");
    }
  }

  private static final class FrontEndGRPCServiceFileDescriptorSupplier
      extends FrontEndGRPCServiceBaseDescriptorSupplier {
    FrontEndGRPCServiceFileDescriptorSupplier() {}
  }

  private static final class FrontEndGRPCServiceMethodDescriptorSupplier
      extends FrontEndGRPCServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final String methodName;

    FrontEndGRPCServiceMethodDescriptorSupplier(String methodName) {
      this.methodName = methodName;
    }

    @Override
    public com.google.protobuf.Descriptors.MethodDescriptor getMethodDescriptor() {
      return getServiceDescriptor().findMethodByName(methodName);
    }
  }

  private static volatile io.grpc.ServiceDescriptor serviceDescriptor;

  public static io.grpc.ServiceDescriptor getServiceDescriptor() {
    io.grpc.ServiceDescriptor result = serviceDescriptor;
    if (result == null) {
      synchronized (FrontEndGRPCServiceGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new FrontEndGRPCServiceFileDescriptorSupplier())
              .addMethod(getPingMethod())
              .addMethod(getGetMetricsMethod())
              .addMethod(getGetAllModelsMethod())
              .addMethod(getGetModelsWithNameMethod())
              .addMethod(getGetModelsWithNameAndVersionMethod())
              .addMethod(getPredictMethod())
              .build();
        }
      }
    }
    return result;
  }
}
