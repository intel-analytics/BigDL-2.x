package com.intel.analytics.zoo.ppml.generated;

import io.grpc.stub.ClientCalls;

import static io.grpc.MethodDescriptor.generateFullMethodName;
import static io.grpc.stub.ClientCalls.blockingUnaryCall;
import static io.grpc.stub.ClientCalls.futureUnaryCall;
import static io.grpc.stub.ServerCalls.asyncUnaryCall;
import static io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall;

/**
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.33.0)",
    comments = "Source: PSI.proto")
public final class PSIServiceGrpc {

  private PSIServiceGrpc() {}

  public static final String SERVICE_NAME = "PSIService";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.zoo.ppml.psi.generated.SaltRequest,
      com.intel.analytics.zoo.ppml.psi.generated.SaltReply> getSaltMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "salt",
      requestType = com.intel.analytics.zoo.ppml.psi.generated.SaltRequest.class,
      responseType = com.intel.analytics.zoo.ppml.psi.generated.SaltReply.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.zoo.ppml.psi.generated.SaltRequest,
      com.intel.analytics.zoo.ppml.psi.generated.SaltReply> getSaltMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.zoo.ppml.psi.generated.SaltRequest, com.intel.analytics.zoo.ppml.psi.generated.SaltReply> getSaltMethod;
    if ((getSaltMethod = com.intel.analytics.zoo.ppml.psi.generated.PSIServiceGrpc.getSaltMethod) == null) {
      synchronized (com.intel.analytics.zoo.ppml.psi.generated.PSIServiceGrpc.class) {
        if ((getSaltMethod = com.intel.analytics.zoo.ppml.psi.generated.PSIServiceGrpc.getSaltMethod) == null) {
          com.intel.analytics.zoo.ppml.psi.generated.PSIServiceGrpc.getSaltMethod = getSaltMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.zoo.ppml.psi.generated.SaltRequest, com.intel.analytics.zoo.ppml.psi.generated.SaltReply>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "salt"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.zoo.ppml.psi.generated.SaltRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.zoo.ppml.psi.generated.SaltReply.getDefaultInstance()))
              .setSchemaDescriptor(new PSIServiceMethodDescriptorSupplier("salt"))
              .build();
        }
      }
    }
    return getSaltMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.zoo.ppml.psi.generated.UploadRequest,
      com.intel.analytics.zoo.ppml.psi.generated.UploadResponse> getUploadSetMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "uploadSet",
      requestType = com.intel.analytics.zoo.ppml.psi.generated.UploadRequest.class,
      responseType = com.intel.analytics.zoo.ppml.psi.generated.UploadResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.zoo.ppml.psi.generated.UploadRequest,
      com.intel.analytics.zoo.ppml.psi.generated.UploadResponse> getUploadSetMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.zoo.ppml.psi.generated.UploadRequest, com.intel.analytics.zoo.ppml.psi.generated.UploadResponse> getUploadSetMethod;
    if ((getUploadSetMethod = com.intel.analytics.zoo.ppml.psi.generated.PSIServiceGrpc.getUploadSetMethod) == null) {
      synchronized (com.intel.analytics.zoo.ppml.psi.generated.PSIServiceGrpc.class) {
        if ((getUploadSetMethod = com.intel.analytics.zoo.ppml.psi.generated.PSIServiceGrpc.getUploadSetMethod) == null) {
          com.intel.analytics.zoo.ppml.psi.generated.PSIServiceGrpc.getUploadSetMethod = getUploadSetMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.zoo.ppml.psi.generated.UploadRequest, com.intel.analytics.zoo.ppml.psi.generated.UploadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "uploadSet"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.zoo.ppml.psi.generated.UploadRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.zoo.ppml.psi.generated.UploadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new PSIServiceMethodDescriptorSupplier("uploadSet"))
              .build();
        }
      }
    }
    return getUploadSetMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.intel.analytics.zoo.ppml.psi.generated.DownloadRequest,
      com.intel.analytics.zoo.ppml.psi.generated.DownloadResponse> getDownloadIntersectionMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "downloadIntersection",
      requestType = com.intel.analytics.zoo.ppml.psi.generated.DownloadRequest.class,
      responseType = com.intel.analytics.zoo.ppml.psi.generated.DownloadResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.intel.analytics.zoo.ppml.psi.generated.DownloadRequest,
      com.intel.analytics.zoo.ppml.psi.generated.DownloadResponse> getDownloadIntersectionMethod() {
    io.grpc.MethodDescriptor<com.intel.analytics.zoo.ppml.psi.generated.DownloadRequest, com.intel.analytics.zoo.ppml.psi.generated.DownloadResponse> getDownloadIntersectionMethod;
    if ((getDownloadIntersectionMethod = com.intel.analytics.zoo.ppml.psi.generated.PSIServiceGrpc.getDownloadIntersectionMethod) == null) {
      synchronized (com.intel.analytics.zoo.ppml.psi.generated.PSIServiceGrpc.class) {
        if ((getDownloadIntersectionMethod = com.intel.analytics.zoo.ppml.psi.generated.PSIServiceGrpc.getDownloadIntersectionMethod) == null) {
          com.intel.analytics.zoo.ppml.psi.generated.PSIServiceGrpc.getDownloadIntersectionMethod = getDownloadIntersectionMethod =
              io.grpc.MethodDescriptor.<com.intel.analytics.zoo.ppml.psi.generated.DownloadRequest, com.intel.analytics.zoo.ppml.psi.generated.DownloadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "downloadIntersection"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.zoo.ppml.psi.generated.DownloadRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.intel.analytics.zoo.ppml.psi.generated.DownloadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new PSIServiceMethodDescriptorSupplier("downloadIntersection"))
              .build();
        }
      }
    }
    return getDownloadIntersectionMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static PSIServiceStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<PSIServiceStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<PSIServiceStub>() {
        @Override
        public PSIServiceStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new PSIServiceStub(channel, callOptions);
        }
      };
    return PSIServiceStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static PSIServiceBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<PSIServiceBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<PSIServiceBlockingStub>() {
        @Override
        public PSIServiceBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new PSIServiceBlockingStub(channel, callOptions);
        }
      };
    return PSIServiceBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static PSIServiceFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<PSIServiceFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<PSIServiceFutureStub>() {
        @Override
        public PSIServiceFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new PSIServiceFutureStub(channel, callOptions);
        }
      };
    return PSIServiceFutureStub.newStub(factory, channel);
  }

  /**
   */
  public static abstract class PSIServiceImplBase implements io.grpc.BindableService {

    /**
     * <pre>
     * Gives SHA256 Hash salt
     * </pre>
     */
    public void salt(com.intel.analytics.zoo.ppml.psi.generated.SaltRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.zoo.ppml.psi.generated.SaltReply> responseObserver) {
      asyncUnimplementedUnaryCall(getSaltMethod(), responseObserver);
    }

    /**
     */
    public void uploadSet(com.intel.analytics.zoo.ppml.psi.generated.UploadRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.zoo.ppml.psi.generated.UploadResponse> responseObserver) {
      asyncUnimplementedUnaryCall(getUploadSetMethod(), responseObserver);
    }

    /**
     */
    public void downloadIntersection(com.intel.analytics.zoo.ppml.psi.generated.DownloadRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.zoo.ppml.psi.generated.DownloadResponse> responseObserver) {
      asyncUnimplementedUnaryCall(getDownloadIntersectionMethod(), responseObserver);
    }

    @Override public final io.grpc.ServerServiceDefinition bindService() {
      return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
          .addMethod(
            getSaltMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.zoo.ppml.psi.generated.SaltRequest,
                com.intel.analytics.zoo.ppml.psi.generated.SaltReply>(
                  this, METHODID_SALT)))
          .addMethod(
            getUploadSetMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.zoo.ppml.psi.generated.UploadRequest,
                com.intel.analytics.zoo.ppml.psi.generated.UploadResponse>(
                  this, METHODID_UPLOAD_SET)))
          .addMethod(
            getDownloadIntersectionMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                com.intel.analytics.zoo.ppml.psi.generated.DownloadRequest,
                com.intel.analytics.zoo.ppml.psi.generated.DownloadResponse>(
                  this, METHODID_DOWNLOAD_INTERSECTION)))
          .build();
    }
  }

  /**
   */
  public static final class PSIServiceStub extends io.grpc.stub.AbstractAsyncStub<PSIServiceStub> {
    private PSIServiceStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected PSIServiceStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new PSIServiceStub(channel, callOptions);
    }

    /**
     * <pre>
     * Gives SHA256 Hash salt
     * </pre>
     */
    public void salt(com.intel.analytics.zoo.ppml.psi.generated.SaltRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.zoo.ppml.psi.generated.SaltReply> responseObserver) {
      ClientCalls.asyncUnaryCall(
          getChannel().newCall(getSaltMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void uploadSet(com.intel.analytics.zoo.ppml.psi.generated.UploadRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.zoo.ppml.psi.generated.UploadResponse> responseObserver) {
      ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUploadSetMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void downloadIntersection(com.intel.analytics.zoo.ppml.psi.generated.DownloadRequest request,
        io.grpc.stub.StreamObserver<com.intel.analytics.zoo.ppml.psi.generated.DownloadResponse> responseObserver) {
      ClientCalls.asyncUnaryCall(
          getChannel().newCall(getDownloadIntersectionMethod(), getCallOptions()), request, responseObserver);
    }
  }

  /**
   */
  public static final class PSIServiceBlockingStub extends io.grpc.stub.AbstractBlockingStub<PSIServiceBlockingStub> {
    private PSIServiceBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected PSIServiceBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new PSIServiceBlockingStub(channel, callOptions);
    }

    /**
     * <pre>
     * Gives SHA256 Hash salt
     * </pre>
     */
    public com.intel.analytics.zoo.ppml.psi.generated.SaltReply salt(com.intel.analytics.zoo.ppml.psi.generated.SaltRequest request) {
      return blockingUnaryCall(
          getChannel(), getSaltMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.intel.analytics.zoo.ppml.psi.generated.UploadResponse uploadSet(com.intel.analytics.zoo.ppml.psi.generated.UploadRequest request) {
      return blockingUnaryCall(
          getChannel(), getUploadSetMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.intel.analytics.zoo.ppml.psi.generated.DownloadResponse downloadIntersection(com.intel.analytics.zoo.ppml.psi.generated.DownloadRequest request) {
      return blockingUnaryCall(
          getChannel(), getDownloadIntersectionMethod(), getCallOptions(), request);
    }
  }

  /**
   */
  public static final class PSIServiceFutureStub extends io.grpc.stub.AbstractFutureStub<PSIServiceFutureStub> {
    private PSIServiceFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected PSIServiceFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new PSIServiceFutureStub(channel, callOptions);
    }

    /**
     * <pre>
     * Gives SHA256 Hash salt
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.zoo.ppml.psi.generated.SaltReply> salt(
        com.intel.analytics.zoo.ppml.psi.generated.SaltRequest request) {
      return futureUnaryCall(
          getChannel().newCall(getSaltMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.zoo.ppml.psi.generated.UploadResponse> uploadSet(
        com.intel.analytics.zoo.ppml.psi.generated.UploadRequest request) {
      return futureUnaryCall(
          getChannel().newCall(getUploadSetMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.intel.analytics.zoo.ppml.psi.generated.DownloadResponse> downloadIntersection(
        com.intel.analytics.zoo.ppml.psi.generated.DownloadRequest request) {
      return futureUnaryCall(
          getChannel().newCall(getDownloadIntersectionMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_SALT = 0;
  private static final int METHODID_UPLOAD_SET = 1;
  private static final int METHODID_DOWNLOAD_INTERSECTION = 2;

  private static final class MethodHandlers<Req, Resp> implements
      io.grpc.stub.ServerCalls.UnaryMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ServerStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ClientStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.BidiStreamingMethod<Req, Resp> {
    private final PSIServiceImplBase serviceImpl;
    private final int methodId;

    MethodHandlers(PSIServiceImplBase serviceImpl, int methodId) {
      this.serviceImpl = serviceImpl;
      this.methodId = methodId;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void invoke(Req request, io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_SALT:
          serviceImpl.salt((com.intel.analytics.zoo.ppml.psi.generated.SaltRequest) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.zoo.ppml.psi.generated.SaltReply>) responseObserver);
          break;
        case METHODID_UPLOAD_SET:
          serviceImpl.uploadSet((com.intel.analytics.zoo.ppml.psi.generated.UploadRequest) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.zoo.ppml.psi.generated.UploadResponse>) responseObserver);
          break;
        case METHODID_DOWNLOAD_INTERSECTION:
          serviceImpl.downloadIntersection((com.intel.analytics.zoo.ppml.psi.generated.DownloadRequest) request,
              (io.grpc.stub.StreamObserver<com.intel.analytics.zoo.ppml.psi.generated.DownloadResponse>) responseObserver);
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

  private static abstract class PSIServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    PSIServiceBaseDescriptorSupplier() {}

    @Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return com.intel.analytics.zoo.ppml.psi.generated.PSIProto.getDescriptor();
    }

    @Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("PSIService");
    }
  }

  private static final class PSIServiceFileDescriptorSupplier
      extends PSIServiceBaseDescriptorSupplier {
    PSIServiceFileDescriptorSupplier() {}
  }

  private static final class PSIServiceMethodDescriptorSupplier
      extends PSIServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final String methodName;

    PSIServiceMethodDescriptorSupplier(String methodName) {
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
      synchronized (com.intel.analytics.zoo.ppml.psi.generated.PSIServiceGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new PSIServiceFileDescriptorSupplier())
              .addMethod(getSaltMethod())
              .addMethod(getUploadSetMethod())
              .addMethod(getDownloadIntersectionMethod())
              .build();
        }
      }
    }
    return result;
  }
}
