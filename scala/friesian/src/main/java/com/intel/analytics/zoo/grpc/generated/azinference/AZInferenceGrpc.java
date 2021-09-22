package com.intel.analytics.zoo.grpc.generated.azinference;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 * <pre>
 * Interface exported by the server.
 * </pre>
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.37.0)",
    comments = "Source: az_inference.proto")
public final class AZInferenceGrpc {

  private AZInferenceGrpc() {}

  public static final String SERVICE_NAME = "azinference.AZInference";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.Content,
      com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.Prediction> getDoPredictMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "doPredict",
      requestType = com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.Content.class,
      responseType = com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.Prediction.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.Content,
      com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.Prediction> getDoPredictMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.Content, com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.Prediction> getDoPredictMethod;
    if ((getDoPredictMethod = AZInferenceGrpc.getDoPredictMethod) == null) {
      synchronized (AZInferenceGrpc.class) {
        if ((getDoPredictMethod = AZInferenceGrpc.getDoPredictMethod) == null) {
          AZInferenceGrpc.getDoPredictMethod = getDoPredictMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.Content, com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.Prediction>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "doPredict"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.Content.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.Prediction.getDefaultInstance()))
              .setSchemaDescriptor(new AZInferenceMethodDescriptorSupplier("doPredict"))
              .build();
        }
      }
    }
    return getDoPredictMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.google.protobuf.Empty,
      com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.ServerMessage> getGetMetricsMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getMetrics",
      requestType = com.google.protobuf.Empty.class,
      responseType = com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.ServerMessage.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.google.protobuf.Empty,
      com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.ServerMessage> getGetMetricsMethod() {
    io.grpc.MethodDescriptor<com.google.protobuf.Empty, com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.ServerMessage> getGetMetricsMethod;
    if ((getGetMetricsMethod = AZInferenceGrpc.getGetMetricsMethod) == null) {
      synchronized (AZInferenceGrpc.class) {
        if ((getGetMetricsMethod = AZInferenceGrpc.getGetMetricsMethod) == null) {
          AZInferenceGrpc.getGetMetricsMethod = getGetMetricsMethod =
              io.grpc.MethodDescriptor.<com.google.protobuf.Empty, com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.ServerMessage>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "getMetrics"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.ServerMessage.getDefaultInstance()))
              .setSchemaDescriptor(new AZInferenceMethodDescriptorSupplier("getMetrics"))
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
    if ((getResetMetricsMethod = AZInferenceGrpc.getResetMetricsMethod) == null) {
      synchronized (AZInferenceGrpc.class) {
        if ((getResetMetricsMethod = AZInferenceGrpc.getResetMetricsMethod) == null) {
          AZInferenceGrpc.getResetMetricsMethod = getResetMetricsMethod =
              io.grpc.MethodDescriptor.<com.google.protobuf.Empty, com.google.protobuf.Empty>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "resetMetrics"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
              .setSchemaDescriptor(new AZInferenceMethodDescriptorSupplier("resetMetrics"))
              .build();
        }
      }
    }
    return getResetMetricsMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static AZInferenceStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<AZInferenceStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<AZInferenceStub>() {
        @Override
        public AZInferenceStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new AZInferenceStub(channel, callOptions);
        }
      };
    return AZInferenceStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static AZInferenceBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<AZInferenceBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<AZInferenceBlockingStub>() {
        @Override
        public AZInferenceBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new AZInferenceBlockingStub(channel, callOptions);
        }
      };
    return AZInferenceBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static AZInferenceFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<AZInferenceFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<AZInferenceFutureStub>() {
        @Override
        public AZInferenceFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new AZInferenceFutureStub(channel, callOptions);
        }
      };
    return AZInferenceFutureStub.newStub(factory, channel);
  }

  /**
   * <pre>
   * Interface exported by the server.
   * </pre>
   */
  public static abstract class AZInferenceImplBase implements io.grpc.BindableService {

    /**
     */
    public void doPredict(com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.Content request,
        io.grpc.stub.StreamObserver<com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.Prediction> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getDoPredictMethod(), responseObserver);
    }

    /**
     */
    public void getMetrics(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.ServerMessage> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetMetricsMethod(), responseObserver);
    }

    /**
     */
    public void resetMetrics(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<com.google.protobuf.Empty> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getResetMetricsMethod(), responseObserver);
    }

    @Override public final io.grpc.ServerServiceDefinition bindService() {
      return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
          .addMethod(
            getDoPredictMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.Content,
                com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.Prediction>(
                  this, METHODID_DO_PREDICT)))
          .addMethod(
            getGetMetricsMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.google.protobuf.Empty,
                com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.ServerMessage>(
                  this, METHODID_GET_METRICS)))
          .addMethod(
            getResetMetricsMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.google.protobuf.Empty,
                com.google.protobuf.Empty>(
                  this, METHODID_RESET_METRICS)))
          .build();
    }
  }

  /**
   * <pre>
   * Interface exported by the server.
   * </pre>
   */
  public static final class AZInferenceStub extends io.grpc.stub.AbstractAsyncStub<AZInferenceStub> {
    private AZInferenceStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected AZInferenceStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new AZInferenceStub(channel, callOptions);
    }

    /**
     */
    public void doPredict(com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.Content request,
        io.grpc.stub.StreamObserver<com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.Prediction> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getDoPredictMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void getMetrics(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.ServerMessage> responseObserver) {
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
  }

  /**
   * <pre>
   * Interface exported by the server.
   * </pre>
   */
  public static final class AZInferenceBlockingStub extends io.grpc.stub.AbstractBlockingStub<AZInferenceBlockingStub> {
    private AZInferenceBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected AZInferenceBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new AZInferenceBlockingStub(channel, callOptions);
    }

    /**
     */
    public com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.Prediction doPredict(com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.Content request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getDoPredictMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.ServerMessage getMetrics(com.google.protobuf.Empty request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetMetricsMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.google.protobuf.Empty resetMetrics(com.google.protobuf.Empty request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getResetMetricsMethod(), getCallOptions(), request);
    }
  }

  /**
   * <pre>
   * Interface exported by the server.
   * </pre>
   */
  public static final class AZInferenceFutureStub extends io.grpc.stub.AbstractFutureStub<AZInferenceFutureStub> {
    private AZInferenceFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected AZInferenceFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new AZInferenceFutureStub(channel, callOptions);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.Prediction> doPredict(
        com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.Content request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getDoPredictMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.ServerMessage> getMetrics(
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
  }

  private static final int METHODID_DO_PREDICT = 0;
  private static final int METHODID_GET_METRICS = 1;
  private static final int METHODID_RESET_METRICS = 2;

  private static final class MethodHandlers<Req, Resp> implements
      io.grpc.stub.ServerCalls.UnaryMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ServerStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ClientStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.BidiStreamingMethod<Req, Resp> {
    private final AZInferenceImplBase serviceImpl;
    private final int methodId;

    MethodHandlers(AZInferenceImplBase serviceImpl, int methodId) {
      this.serviceImpl = serviceImpl;
      this.methodId = methodId;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void invoke(Req request, io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_DO_PREDICT:
          serviceImpl.doPredict((com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.Content) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.Prediction>) responseObserver);
          break;
        case METHODID_GET_METRICS:
          serviceImpl.getMetrics((com.google.protobuf.Empty) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.ServerMessage>) responseObserver);
          break;
        case METHODID_RESET_METRICS:
          serviceImpl.resetMetrics((com.google.protobuf.Empty) request,
              (io.grpc.stub.StreamObserver<com.google.protobuf.Empty>) responseObserver);
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

  private static abstract class AZInferenceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    AZInferenceBaseDescriptorSupplier() {}

    @Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.getDescriptor();
    }

    @Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("AZInference");
    }
  }

  private static final class AZInferenceFileDescriptorSupplier
      extends AZInferenceBaseDescriptorSupplier {
    AZInferenceFileDescriptorSupplier() {}
  }

  private static final class AZInferenceMethodDescriptorSupplier
      extends AZInferenceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final String methodName;

    AZInferenceMethodDescriptorSupplier(String methodName) {
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
      synchronized (AZInferenceGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new AZInferenceFileDescriptorSupplier())
              .addMethod(getDoPredictMethod())
              .addMethod(getGetMetricsMethod())
              .addMethod(getResetMetricsMethod())
              .build();
        }
      }
    }
    return result;
  }
}
