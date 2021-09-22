package com.intel.analytics.zoo.grpc.generated.recommend;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 * <pre>
 * Interface exported by the server.
 * </pre>
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.37.0)",
    comments = "Source: recommend.proto")
public final class RecommendServiceGrpc {

  private RecommendServiceGrpc() {}

  public static final String SERVICE_NAME = "recommend.RecommendService";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.RecommendRequest,
      com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.RecommendIDProbs> getGetRecommendIDsMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getRecommendIDs",
      requestType = com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.RecommendRequest.class,
      responseType = com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.RecommendIDProbs.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.RecommendRequest,
      com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.RecommendIDProbs> getGetRecommendIDsMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.RecommendRequest, com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.RecommendIDProbs> getGetRecommendIDsMethod;
    if ((getGetRecommendIDsMethod = RecommendServiceGrpc.getGetRecommendIDsMethod) == null) {
      synchronized (RecommendServiceGrpc.class) {
        if ((getGetRecommendIDsMethod = RecommendServiceGrpc.getGetRecommendIDsMethod) == null) {
          RecommendServiceGrpc.getGetRecommendIDsMethod = getGetRecommendIDsMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.RecommendRequest, com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.RecommendIDProbs>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "getRecommendIDs"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.RecommendRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.RecommendIDProbs.getDefaultInstance()))
              .setSchemaDescriptor(new RecommendServiceMethodDescriptorSupplier("getRecommendIDs"))
              .build();
        }
      }
    }
    return getGetRecommendIDsMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.google.protobuf.Empty,
      com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.ServerMessage> getGetMetricsMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getMetrics",
      requestType = com.google.protobuf.Empty.class,
      responseType = com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.ServerMessage.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.google.protobuf.Empty,
      com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.ServerMessage> getGetMetricsMethod() {
    io.grpc.MethodDescriptor<com.google.protobuf.Empty, com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.ServerMessage> getGetMetricsMethod;
    if ((getGetMetricsMethod = RecommendServiceGrpc.getGetMetricsMethod) == null) {
      synchronized (RecommendServiceGrpc.class) {
        if ((getGetMetricsMethod = RecommendServiceGrpc.getGetMetricsMethod) == null) {
          RecommendServiceGrpc.getGetMetricsMethod = getGetMetricsMethod =
              io.grpc.MethodDescriptor.<com.google.protobuf.Empty, com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.ServerMessage>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "getMetrics"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.ServerMessage.getDefaultInstance()))
              .setSchemaDescriptor(new RecommendServiceMethodDescriptorSupplier("getMetrics"))
              .build();
        }
      }
    }
    return getGetMetricsMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.google.protobuf.Empty,
      com.google.protobuf.Empty> getResetMetricsMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "resetMetrics",
      requestType = com.google.protobuf.Empty.class,
      responseType = com.google.protobuf.Empty.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.google.protobuf.Empty,
      com.google.protobuf.Empty> getResetMetricsMethod() {
    io.grpc.MethodDescriptor<com.google.protobuf.Empty, com.google.protobuf.Empty> getResetMetricsMethod;
    if ((getResetMetricsMethod = RecommendServiceGrpc.getResetMetricsMethod) == null) {
      synchronized (RecommendServiceGrpc.class) {
        if ((getResetMetricsMethod = RecommendServiceGrpc.getResetMetricsMethod) == null) {
          RecommendServiceGrpc.getResetMetricsMethod = getResetMetricsMethod =
              io.grpc.MethodDescriptor.<com.google.protobuf.Empty, com.google.protobuf.Empty>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "resetMetrics"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
              .setSchemaDescriptor(new RecommendServiceMethodDescriptorSupplier("resetMetrics"))
              .build();
        }
      }
    }
    return getResetMetricsMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.google.protobuf.Empty,
      com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.ServerMessage> getGetClientMetricsMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getClientMetrics",
      requestType = com.google.protobuf.Empty.class,
      responseType = com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.ServerMessage.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.google.protobuf.Empty,
      com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.ServerMessage> getGetClientMetricsMethod() {
    io.grpc.MethodDescriptor<com.google.protobuf.Empty, com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.ServerMessage> getGetClientMetricsMethod;
    if ((getGetClientMetricsMethod = RecommendServiceGrpc.getGetClientMetricsMethod) == null) {
      synchronized (RecommendServiceGrpc.class) {
        if ((getGetClientMetricsMethod = RecommendServiceGrpc.getGetClientMetricsMethod) == null) {
          RecommendServiceGrpc.getGetClientMetricsMethod = getGetClientMetricsMethod =
              io.grpc.MethodDescriptor.<com.google.protobuf.Empty, com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.ServerMessage>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "getClientMetrics"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.ServerMessage.getDefaultInstance()))
              .setSchemaDescriptor(new RecommendServiceMethodDescriptorSupplier("getClientMetrics"))
              .build();
        }
      }
    }
    return getGetClientMetricsMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static RecommendServiceStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<RecommendServiceStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<RecommendServiceStub>() {
        @Override
        public RecommendServiceStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new RecommendServiceStub(channel, callOptions);
        }
      };
    return RecommendServiceStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static RecommendServiceBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<RecommendServiceBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<RecommendServiceBlockingStub>() {
        @Override
        public RecommendServiceBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new RecommendServiceBlockingStub(channel, callOptions);
        }
      };
    return RecommendServiceBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static RecommendServiceFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<RecommendServiceFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<RecommendServiceFutureStub>() {
        @Override
        public RecommendServiceFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new RecommendServiceFutureStub(channel, callOptions);
        }
      };
    return RecommendServiceFutureStub.newStub(factory, channel);
  }

  /**
   * <pre>
   * Interface exported by the server.
   * </pre>
   */
  public static abstract class RecommendServiceImplBase implements io.grpc.BindableService {

    /**
     */
    public void getRecommendIDs(com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.RecommendRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.RecommendIDProbs> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetRecommendIDsMethod(), responseObserver);
    }

    /**
     */
    public void getMetrics(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.ServerMessage> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetMetricsMethod(), responseObserver);
    }

    /**
     */
    public void resetMetrics(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<com.google.protobuf.Empty> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getResetMetricsMethod(), responseObserver);
    }

    /**
     */
    public void getClientMetrics(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.ServerMessage> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetClientMetricsMethod(), responseObserver);
    }

    @Override public final io.grpc.ServerServiceDefinition bindService() {
      return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
          .addMethod(
            getGetRecommendIDsMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.RecommendRequest,
                com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.RecommendIDProbs>(
                  this, METHODID_GET_RECOMMEND_IDS)))
          .addMethod(
            getGetMetricsMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.google.protobuf.Empty,
                com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.ServerMessage>(
                  this, METHODID_GET_METRICS)))
          .addMethod(
            getResetMetricsMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.google.protobuf.Empty,
                com.google.protobuf.Empty>(
                  this, METHODID_RESET_METRICS)))
          .addMethod(
            getGetClientMetricsMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.google.protobuf.Empty,
                com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.ServerMessage>(
                  this, METHODID_GET_CLIENT_METRICS)))
          .build();
    }
  }

  /**
   * <pre>
   * Interface exported by the server.
   * </pre>
   */
  public static final class RecommendServiceStub extends io.grpc.stub.AbstractAsyncStub<RecommendServiceStub> {
    private RecommendServiceStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected RecommendServiceStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new RecommendServiceStub(channel, callOptions);
    }

    /**
     */
    public void getRecommendIDs(com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.RecommendRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.RecommendIDProbs> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGetRecommendIDsMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void getMetrics(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.ServerMessage> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGetMetricsMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void resetMetrics(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<com.google.protobuf.Empty> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getResetMetricsMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void getClientMetrics(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.ServerMessage> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGetClientMetricsMethod(), getCallOptions()), request, responseObserver);
    }
  }

  /**
   * <pre>
   * Interface exported by the server.
   * </pre>
   */
  public static final class RecommendServiceBlockingStub extends io.grpc.stub.AbstractBlockingStub<RecommendServiceBlockingStub> {
    private RecommendServiceBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected RecommendServiceBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new RecommendServiceBlockingStub(channel, callOptions);
    }

    /**
     */
    public com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.RecommendIDProbs getRecommendIDs(com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.RecommendRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetRecommendIDsMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.ServerMessage getMetrics(com.google.protobuf.Empty request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetMetricsMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.google.protobuf.Empty resetMetrics(com.google.protobuf.Empty request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getResetMetricsMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.ServerMessage getClientMetrics(com.google.protobuf.Empty request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetClientMetricsMethod(), getCallOptions(), request);
    }
  }

  /**
   * <pre>
   * Interface exported by the server.
   * </pre>
   */
  public static final class RecommendServiceFutureStub extends io.grpc.stub.AbstractFutureStub<RecommendServiceFutureStub> {
    private RecommendServiceFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected RecommendServiceFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new RecommendServiceFutureStub(channel, callOptions);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.RecommendIDProbs> getRecommendIDs(
        com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.RecommendRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGetRecommendIDsMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.ServerMessage> getMetrics(
        com.google.protobuf.Empty request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGetMetricsMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.google.protobuf.Empty> resetMetrics(
        com.google.protobuf.Empty request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getResetMetricsMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.ServerMessage> getClientMetrics(
        com.google.protobuf.Empty request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGetClientMetricsMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_GET_RECOMMEND_IDS = 0;
  private static final int METHODID_GET_METRICS = 1;
  private static final int METHODID_RESET_METRICS = 2;
  private static final int METHODID_GET_CLIENT_METRICS = 3;

  private static final class MethodHandlers<Req, Resp> implements
      io.grpc.stub.ServerCalls.UnaryMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ServerStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ClientStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.BidiStreamingMethod<Req, Resp> {
    private final RecommendServiceImplBase serviceImpl;
    private final int methodId;

    MethodHandlers(RecommendServiceImplBase serviceImpl, int methodId) {
      this.serviceImpl = serviceImpl;
      this.methodId = methodId;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void invoke(Req request, io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_GET_RECOMMEND_IDS:
          serviceImpl.getRecommendIDs((com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.RecommendRequest) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.RecommendIDProbs>) responseObserver);
          break;
        case METHODID_GET_METRICS:
          serviceImpl.getMetrics((com.google.protobuf.Empty) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.ServerMessage>) responseObserver);
          break;
        case METHODID_RESET_METRICS:
          serviceImpl.resetMetrics((com.google.protobuf.Empty) request,
              (io.grpc.stub.StreamObserver<com.google.protobuf.Empty>) responseObserver);
          break;
        case METHODID_GET_CLIENT_METRICS:
          serviceImpl.getClientMetrics((com.google.protobuf.Empty) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.ServerMessage>) responseObserver);
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

  private static abstract class RecommendServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    RecommendServiceBaseDescriptorSupplier() {}

    @Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.getDescriptor();
    }

    @Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("RecommendService");
    }
  }

  private static final class RecommendServiceFileDescriptorSupplier
      extends RecommendServiceBaseDescriptorSupplier {
    RecommendServiceFileDescriptorSupplier() {}
  }

  private static final class RecommendServiceMethodDescriptorSupplier
      extends RecommendServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final String methodName;

    RecommendServiceMethodDescriptorSupplier(String methodName) {
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
      synchronized (RecommendServiceGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new RecommendServiceFileDescriptorSupplier())
              .addMethod(getGetRecommendIDsMethod())
              .addMethod(getGetMetricsMethod())
              .addMethod(getResetMetricsMethod())
              .addMethod(getGetClientMetricsMethod())
              .build();
        }
      }
    }
    return result;
  }
}
