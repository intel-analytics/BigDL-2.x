package com.intel.analytics.zoo.ppml.psi.generated;

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
  private static volatile io.grpc.MethodDescriptor<SaltRequest,
      SaltReply> getSaltMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "salt",
      requestType = SaltRequest.class,
      responseType = SaltReply.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<SaltRequest,
      SaltReply> getSaltMethod() {
    io.grpc.MethodDescriptor<SaltRequest, SaltReply> getSaltMethod;
    if ((getSaltMethod = PSIServiceGrpc.getSaltMethod) == null) {
      synchronized (PSIServiceGrpc.class) {
        if ((getSaltMethod = PSIServiceGrpc.getSaltMethod) == null) {
          PSIServiceGrpc.getSaltMethod = getSaltMethod =
              io.grpc.MethodDescriptor.<SaltRequest, SaltReply>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "salt"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  SaltRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  SaltReply.getDefaultInstance()))
              .setSchemaDescriptor(new PSIServiceMethodDescriptorSupplier("salt"))
              .build();
        }
      }
    }
    return getSaltMethod;
  }

  private static volatile io.grpc.MethodDescriptor<UploadRequest,
      UploadResponse> getUploadSetMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "uploadSet",
      requestType = UploadRequest.class,
      responseType = UploadResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<UploadRequest,
      UploadResponse> getUploadSetMethod() {
    io.grpc.MethodDescriptor<UploadRequest, UploadResponse> getUploadSetMethod;
    if ((getUploadSetMethod = PSIServiceGrpc.getUploadSetMethod) == null) {
      synchronized (PSIServiceGrpc.class) {
        if ((getUploadSetMethod = PSIServiceGrpc.getUploadSetMethod) == null) {
          PSIServiceGrpc.getUploadSetMethod = getUploadSetMethod =
              io.grpc.MethodDescriptor.<UploadRequest, UploadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "uploadSet"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  UploadRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  UploadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new PSIServiceMethodDescriptorSupplier("uploadSet"))
              .build();
        }
      }
    }
    return getUploadSetMethod;
  }

  private static volatile io.grpc.MethodDescriptor<DownloadRequest,
      DownloadResponse> getDownloadIntersectionMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "downloadIntersection",
      requestType = DownloadRequest.class,
      responseType = DownloadResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<DownloadRequest,
      DownloadResponse> getDownloadIntersectionMethod() {
    io.grpc.MethodDescriptor<DownloadRequest, DownloadResponse> getDownloadIntersectionMethod;
    if ((getDownloadIntersectionMethod = PSIServiceGrpc.getDownloadIntersectionMethod) == null) {
      synchronized (PSIServiceGrpc.class) {
        if ((getDownloadIntersectionMethod = PSIServiceGrpc.getDownloadIntersectionMethod) == null) {
          PSIServiceGrpc.getDownloadIntersectionMethod = getDownloadIntersectionMethod =
              io.grpc.MethodDescriptor.<DownloadRequest, DownloadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "downloadIntersection"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  DownloadRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  DownloadResponse.getDefaultInstance()))
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
    public void salt(SaltRequest request,
                     io.grpc.stub.StreamObserver<SaltReply> responseObserver) {
      asyncUnimplementedUnaryCall(getSaltMethod(), responseObserver);
    }

    /**
     */
    public void uploadSet(UploadRequest request,
                          io.grpc.stub.StreamObserver<UploadResponse> responseObserver) {
      asyncUnimplementedUnaryCall(getUploadSetMethod(), responseObserver);
    }

    /**
     */
    public void downloadIntersection(DownloadRequest request,
                                     io.grpc.stub.StreamObserver<DownloadResponse> responseObserver) {
      asyncUnimplementedUnaryCall(getDownloadIntersectionMethod(), responseObserver);
    }

    @Override public final io.grpc.ServerServiceDefinition bindService() {
      return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
          .addMethod(
            getSaltMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                SaltRequest,
                SaltReply>(
                  this, METHODID_SALT)))
          .addMethod(
            getUploadSetMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                UploadRequest,
                UploadResponse>(
                  this, METHODID_UPLOAD_SET)))
          .addMethod(
            getDownloadIntersectionMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                DownloadRequest,
                DownloadResponse>(
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
    public void salt(SaltRequest request,
                     io.grpc.stub.StreamObserver<SaltReply> responseObserver) {
      ClientCalls.asyncUnaryCall(
          getChannel().newCall(getSaltMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void uploadSet(UploadRequest request,
                          io.grpc.stub.StreamObserver<UploadResponse> responseObserver) {
      ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUploadSetMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void downloadIntersection(DownloadRequest request,
                                     io.grpc.stub.StreamObserver<DownloadResponse> responseObserver) {
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
    public SaltReply salt(SaltRequest request) {
      return blockingUnaryCall(
          getChannel(), getSaltMethod(), getCallOptions(), request);
    }

    /**
     */
    public UploadResponse uploadSet(UploadRequest request) {
      return blockingUnaryCall(
          getChannel(), getUploadSetMethod(), getCallOptions(), request);
    }

    /**
     */
    public DownloadResponse downloadIntersection(DownloadRequest request) {
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
    public com.google.common.util.concurrent.ListenableFuture<SaltReply> salt(
        SaltRequest request) {
      return futureUnaryCall(
          getChannel().newCall(getSaltMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<UploadResponse> uploadSet(
        UploadRequest request) {
      return futureUnaryCall(
          getChannel().newCall(getUploadSetMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<DownloadResponse> downloadIntersection(
        DownloadRequest request) {
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
          serviceImpl.salt((SaltRequest) request,
              (io.grpc.stub.StreamObserver<SaltReply>) responseObserver);
          break;
        case METHODID_UPLOAD_SET:
          serviceImpl.uploadSet((UploadRequest) request,
              (io.grpc.stub.StreamObserver<UploadResponse>) responseObserver);
          break;
        case METHODID_DOWNLOAD_INTERSECTION:
          serviceImpl.downloadIntersection((DownloadRequest) request,
              (io.grpc.stub.StreamObserver<DownloadResponse>) responseObserver);
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
      return PSIProto.getDescriptor();
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
      synchronized (PSIServiceGrpc.class) {
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
