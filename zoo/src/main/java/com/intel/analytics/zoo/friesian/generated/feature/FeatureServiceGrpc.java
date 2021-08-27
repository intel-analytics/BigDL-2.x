package com.intel.analytics.zoo.friesian.generated.feature;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 * <pre>
 * Interface exported by the server.
 * </pre>
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.37.0)",
    comments = "Source: feature.proto")
public final class FeatureServiceGrpc {

  private FeatureServiceGrpc() {}

  public static final String SERVICE_NAME = "feature.FeatureService";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<IDs,
          Features> getGetUserFeaturesMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getUserFeatures",
      requestType = IDs.class,
      responseType = Features.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<IDs,
          Features> getGetUserFeaturesMethod() {
    io.grpc.MethodDescriptor<IDs, Features> getGetUserFeaturesMethod;
    if ((getGetUserFeaturesMethod = FeatureServiceGrpc.getGetUserFeaturesMethod) == null) {
      synchronized (FeatureServiceGrpc.class) {
        if ((getGetUserFeaturesMethod = FeatureServiceGrpc.getGetUserFeaturesMethod) == null) {
          FeatureServiceGrpc.getGetUserFeaturesMethod = getGetUserFeaturesMethod =
              io.grpc.MethodDescriptor.<IDs, Features>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "getUserFeatures"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  IDs.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  Features.getDefaultInstance()))
              .setSchemaDescriptor(new FeatureServiceMethodDescriptorSupplier("getUserFeatures"))
              .build();
        }
      }
    }
    return getGetUserFeaturesMethod;
  }

  private static volatile io.grpc.MethodDescriptor<IDs,
          Features> getGetItemFeaturesMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getItemFeatures",
      requestType = IDs.class,
      responseType = Features.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<IDs,
          Features> getGetItemFeaturesMethod() {
    io.grpc.MethodDescriptor<IDs, Features> getGetItemFeaturesMethod;
    if ((getGetItemFeaturesMethod = FeatureServiceGrpc.getGetItemFeaturesMethod) == null) {
      synchronized (FeatureServiceGrpc.class) {
        if ((getGetItemFeaturesMethod = FeatureServiceGrpc.getGetItemFeaturesMethod) == null) {
          FeatureServiceGrpc.getGetItemFeaturesMethod = getGetItemFeaturesMethod =
              io.grpc.MethodDescriptor.<IDs, Features>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "getItemFeatures"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  IDs.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  Features.getDefaultInstance()))
              .setSchemaDescriptor(new FeatureServiceMethodDescriptorSupplier("getItemFeatures"))
              .build();
        }
      }
    }
    return getGetItemFeaturesMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.google.protobuf.Empty,
          ServerMessage> getGetMetricsMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getMetrics",
      requestType = com.google.protobuf.Empty.class,
      responseType = ServerMessage.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.google.protobuf.Empty,
          ServerMessage> getGetMetricsMethod() {
    io.grpc.MethodDescriptor<com.google.protobuf.Empty, ServerMessage> getGetMetricsMethod;
    if ((getGetMetricsMethod = FeatureServiceGrpc.getGetMetricsMethod) == null) {
      synchronized (FeatureServiceGrpc.class) {
        if ((getGetMetricsMethod = FeatureServiceGrpc.getGetMetricsMethod) == null) {
          FeatureServiceGrpc.getGetMetricsMethod = getGetMetricsMethod =
              io.grpc.MethodDescriptor.<com.google.protobuf.Empty, ServerMessage>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "getMetrics"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  ServerMessage.getDefaultInstance()))
              .setSchemaDescriptor(new FeatureServiceMethodDescriptorSupplier("getMetrics"))
              .build();
        }
      }
    }
    return getGetMetricsMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static FeatureServiceStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<FeatureServiceStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<FeatureServiceStub>() {
        @java.lang.Override
        public FeatureServiceStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new FeatureServiceStub(channel, callOptions);
        }
      };
    return FeatureServiceStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static FeatureServiceBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<FeatureServiceBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<FeatureServiceBlockingStub>() {
        @java.lang.Override
        public FeatureServiceBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new FeatureServiceBlockingStub(channel, callOptions);
        }
      };
    return FeatureServiceBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static FeatureServiceFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<FeatureServiceFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<FeatureServiceFutureStub>() {
        @java.lang.Override
        public FeatureServiceFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new FeatureServiceFutureStub(channel, callOptions);
        }
      };
    return FeatureServiceFutureStub.newStub(factory, channel);
  }

  /**
   * <pre>
   * Interface exported by the server.
   * </pre>
   */
  public static abstract class FeatureServiceImplBase implements io.grpc.BindableService {

    /**
     */
    public void getUserFeatures(IDs request,
        io.grpc.stub.StreamObserver<Features> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetUserFeaturesMethod(), responseObserver);
    }

    /**
     */
    public void getItemFeatures(IDs request,
        io.grpc.stub.StreamObserver<Features> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetItemFeaturesMethod(), responseObserver);
    }

    /**
     */
    public void getMetrics(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<ServerMessage> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetMetricsMethod(), responseObserver);
    }

    @java.lang.Override public final io.grpc.ServerServiceDefinition bindService() {
      return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
          .addMethod(
            getGetUserFeaturesMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                      IDs,
                      Features>(
                  this, METHODID_GET_USER_FEATURES)))
          .addMethod(
            getGetItemFeaturesMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                      IDs,
                      Features>(
                  this, METHODID_GET_ITEM_FEATURES)))
          .addMethod(
            getGetMetricsMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                com.google.protobuf.Empty,
                      ServerMessage>(
                  this, METHODID_GET_METRICS)))
          .build();
    }
  }

  /**
   * <pre>
   * Interface exported by the server.
   * </pre>
   */
  public static final class FeatureServiceStub extends io.grpc.stub.AbstractAsyncStub<FeatureServiceStub> {
    private FeatureServiceStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected FeatureServiceStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new FeatureServiceStub(channel, callOptions);
    }

    /**
     */
    public void getUserFeatures(IDs request,
        io.grpc.stub.StreamObserver<Features> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGetUserFeaturesMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void getItemFeatures(IDs request,
        io.grpc.stub.StreamObserver<Features> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGetItemFeaturesMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void getMetrics(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<ServerMessage> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGetMetricsMethod(), getCallOptions()), request, responseObserver);
    }
  }

  /**
   * <pre>
   * Interface exported by the server.
   * </pre>
   */
  public static final class FeatureServiceBlockingStub extends io.grpc.stub.AbstractBlockingStub<FeatureServiceBlockingStub> {
    private FeatureServiceBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected FeatureServiceBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new FeatureServiceBlockingStub(channel, callOptions);
    }

    /**
     */
    public Features getUserFeatures(IDs request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetUserFeaturesMethod(), getCallOptions(), request);
    }

    /**
     */
    public Features getItemFeatures(IDs request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetItemFeaturesMethod(), getCallOptions(), request);
    }

    /**
     */
    public ServerMessage getMetrics(com.google.protobuf.Empty request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetMetricsMethod(), getCallOptions(), request);
    }
  }

  /**
   * <pre>
   * Interface exported by the server.
   * </pre>
   */
  public static final class FeatureServiceFutureStub extends io.grpc.stub.AbstractFutureStub<FeatureServiceFutureStub> {
    private FeatureServiceFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected FeatureServiceFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new FeatureServiceFutureStub(channel, callOptions);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<Features> getUserFeatures(
        IDs request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGetUserFeaturesMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<Features> getItemFeatures(
        IDs request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGetItemFeaturesMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<ServerMessage> getMetrics(
        com.google.protobuf.Empty request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGetMetricsMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_GET_USER_FEATURES = 0;
  private static final int METHODID_GET_ITEM_FEATURES = 1;
  private static final int METHODID_GET_METRICS = 2;

  private static final class MethodHandlers<Req, Resp> implements
      io.grpc.stub.ServerCalls.UnaryMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ServerStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ClientStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.BidiStreamingMethod<Req, Resp> {
    private final FeatureServiceImplBase serviceImpl;
    private final int methodId;

    MethodHandlers(FeatureServiceImplBase serviceImpl, int methodId) {
      this.serviceImpl = serviceImpl;
      this.methodId = methodId;
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("unchecked")
    public void invoke(Req request, io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_GET_USER_FEATURES:
          serviceImpl.getUserFeatures((IDs) request,
              (io.grpc.stub.StreamObserver<Features>) responseObserver);
          break;
        case METHODID_GET_ITEM_FEATURES:
          serviceImpl.getItemFeatures((IDs) request,
              (io.grpc.stub.StreamObserver<Features>) responseObserver);
          break;
        case METHODID_GET_METRICS:
          serviceImpl.getMetrics((com.google.protobuf.Empty) request,
              (io.grpc.stub.StreamObserver<ServerMessage>) responseObserver);
          break;
        default:
          throw new AssertionError();
      }
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("unchecked")
    public io.grpc.stub.StreamObserver<Req> invoke(
        io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        default:
          throw new AssertionError();
      }
    }
  }

  private static abstract class FeatureServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    FeatureServiceBaseDescriptorSupplier() {}

    @java.lang.Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return FeatureProto.getDescriptor();
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("FeatureService");
    }
  }

  private static final class FeatureServiceFileDescriptorSupplier
      extends FeatureServiceBaseDescriptorSupplier {
    FeatureServiceFileDescriptorSupplier() {}
  }

  private static final class FeatureServiceMethodDescriptorSupplier
      extends FeatureServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final String methodName;

    FeatureServiceMethodDescriptorSupplier(String methodName) {
      this.methodName = methodName;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.MethodDescriptor getMethodDescriptor() {
      return getServiceDescriptor().findMethodByName(methodName);
    }
  }

  private static volatile io.grpc.ServiceDescriptor serviceDescriptor;

  public static io.grpc.ServiceDescriptor getServiceDescriptor() {
    io.grpc.ServiceDescriptor result = serviceDescriptor;
    if (result == null) {
      synchronized (FeatureServiceGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new FeatureServiceFileDescriptorSupplier())
              .addMethod(getGetUserFeaturesMethod())
              .addMethod(getGetItemFeaturesMethod())
              .addMethod(getGetMetricsMethod())
              .build();
        }
      }
    }
    return result;
  }
}
