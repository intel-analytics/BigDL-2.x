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
  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.zoo.grpc.generated.azinference.Content,
          com.intel.analytics.zoo.grpc.generated.azinference.Prediction> getDoPredictMethod;

  @io.grpc.stub.annotations.RpcMethod(
          fullMethodName = SERVICE_NAME + '/' + "doPredict",
          requestType = com.intel.analytics.zoo.grpc.generated.azinference.Content.class,
          responseType = com.intel.analytics.zoo.grpc.generated.azinference.Prediction.class,
          methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.zoo.grpc.generated.azinference.Content,
          com.intel.analytics.zoo.grpc.generated.azinference.Prediction> getDoPredictMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.zoo.grpc.generated.azinference.Content, com.intel.analytics.zoo.grpc.generated.azinference.Prediction> getDoPredictMethod;
    if ((getDoPredictMethod = AZInferenceGrpc.getDoPredictMethod) == null) {
      synchronized (AZInferenceGrpc.class) {
        if ((getDoPredictMethod = AZInferenceGrpc.getDoPredictMethod) == null) {
          AZInferenceGrpc.getDoPredictMethod = getDoPredictMethod =
                  io.grpc.MethodDescriptor.<com.intel.analytics.zoo.grpc.generated.azinference.Content, com.intel.analytics.zoo.grpc.generated.azinference.Prediction>newBuilder()
                          .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
                          .setFullMethodName(generateFullMethodName(SERVICE_NAME, "doPredict"))
                          .setSampledToLocalTracing(true)
                          .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                                  com.intel.analytics.zoo.grpc.generated.azinference.Content.getDefaultInstance()))
                          .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                                  com.intel.analytics.zoo.grpc.generated.azinference.Prediction.getDefaultInstance()))
                          .setSchemaDescriptor(new AZInferenceMethodDescriptorSupplier("doPredict"))
                          .build();
        }
      }
    }
    return getDoPredictMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static AZInferenceStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<AZInferenceStub> factory =
            new io.grpc.stub.AbstractStub.StubFactory<AZInferenceStub>() {
              @java.lang.Override
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
              @java.lang.Override
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
              @java.lang.Override
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
    public void doPredict(com.intel.analytics.zoo.grpc.generated.azinference.Content request,
                          io.grpc.stub.StreamObserver<com.intel.analytics.zoo.grpc.generated.azinference.Prediction> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getDoPredictMethod(), responseObserver);
    }

    @java.lang.Override public final io.grpc.ServerServiceDefinition bindService() {
      return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
              .addMethod(
                      getDoPredictMethod(),
                      io.grpc.stub.ServerCalls.asyncUnaryCall(
                              new MethodHandlers<
                                      com.intel.analytics.zoo.grpc.generated.azinference.Content,
                                      com.intel.analytics.zoo.grpc.generated.azinference.Prediction>(
                                      this, METHODID_DO_PREDICT)))
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

    @java.lang.Override
    protected AZInferenceStub build(
            io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new AZInferenceStub(channel, callOptions);
    }

    /**
     */
    public void doPredict(com.intel.analytics.zoo.grpc.generated.azinference.Content request,
                          io.grpc.stub.StreamObserver<com.intel.analytics.zoo.grpc.generated.azinference.Prediction> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
              getChannel().newCall(getDoPredictMethod(), getCallOptions()), request, responseObserver);
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

    @java.lang.Override
    protected AZInferenceBlockingStub build(
            io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new AZInferenceBlockingStub(channel, callOptions);
    }

    /**
     */
    public com.intel.analytics.zoo.grpc.generated.azinference.Prediction doPredict(com.intel.analytics.zoo.grpc.generated.azinference.Content request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
              getChannel(), getDoPredictMethod(), getCallOptions(), request);
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

    @java.lang.Override
    protected AZInferenceFutureStub build(
            io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new AZInferenceFutureStub(channel, callOptions);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.zoo.grpc.generated.azinference.Prediction> doPredict(
            com.intel.analytics.zoo.grpc.generated.azinference.Content request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
              getChannel().newCall(getDoPredictMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_DO_PREDICT = 0;

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

    @java.lang.Override
    @java.lang.SuppressWarnings("unchecked")
    public void invoke(Req request, io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_DO_PREDICT:
          serviceImpl.doPredict((com.intel.analytics.zoo.grpc.generated.azinference.Content) request,
                  (io.grpc.stub.StreamObserver<com.intel.analytics.zoo.grpc.generated.azinference.Prediction>) responseObserver);
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

  private static abstract class AZInferenceBaseDescriptorSupplier
          implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    AZInferenceBaseDescriptorSupplier() {}

    @java.lang.Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.getDescriptor();
    }

    @java.lang.Override
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

    @java.lang.Override
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
                  .build();
        }
      }
    }
    return result;
  }
}
