package com.intel.analytics.zoo.ppml.generated;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.37.0)",
    comments = "Source: FLProto.proto")
public final class GBServiceGrpc {

  private GBServiceGrpc() {}

  public static final String SERVICE_NAME = "GBService";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<FLProto.UploadSplitRequest,
      FLProto.UploadResponse> getUploadSplitTrainMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "UploadSplitTrain",
      requestType = FLProto.UploadSplitRequest.class,
      responseType = FLProto.UploadResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<FLProto.UploadSplitRequest,
      FLProto.UploadResponse> getUploadSplitTrainMethod() {
    io.grpc.MethodDescriptor<FLProto.UploadSplitRequest, FLProto.UploadResponse> getUploadSplitTrainMethod;
    if ((getUploadSplitTrainMethod = GBServiceGrpc.getUploadSplitTrainMethod) == null) {
      synchronized (GBServiceGrpc.class) {
        if ((getUploadSplitTrainMethod = GBServiceGrpc.getUploadSplitTrainMethod) == null) {
          GBServiceGrpc.getUploadSplitTrainMethod = getUploadSplitTrainMethod =
              io.grpc.MethodDescriptor.<FLProto.UploadSplitRequest, FLProto.UploadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "UploadSplitTrain"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.UploadSplitRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.UploadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GBServiceMethodDescriptorSupplier("UploadSplitTrain"))
              .build();
        }
      }
    }
    return getUploadSplitTrainMethod;
  }

  private static volatile io.grpc.MethodDescriptor<FLProto.DownloadSplitRequest,
      FLProto.DownloadSplitResponse> getDownloadSplitTrainMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "DownloadSplitTrain",
      requestType = FLProto.DownloadSplitRequest.class,
      responseType = FLProto.DownloadSplitResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<FLProto.DownloadSplitRequest,
      FLProto.DownloadSplitResponse> getDownloadSplitTrainMethod() {
    io.grpc.MethodDescriptor<FLProto.DownloadSplitRequest, FLProto.DownloadSplitResponse> getDownloadSplitTrainMethod;
    if ((getDownloadSplitTrainMethod = GBServiceGrpc.getDownloadSplitTrainMethod) == null) {
      synchronized (GBServiceGrpc.class) {
        if ((getDownloadSplitTrainMethod = GBServiceGrpc.getDownloadSplitTrainMethod) == null) {
          GBServiceGrpc.getDownloadSplitTrainMethod = getDownloadSplitTrainMethod =
              io.grpc.MethodDescriptor.<FLProto.DownloadSplitRequest, FLProto.DownloadSplitResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "DownloadSplitTrain"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.DownloadSplitRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.DownloadSplitResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GBServiceMethodDescriptorSupplier("DownloadSplitTrain"))
              .build();
        }
      }
    }
    return getDownloadSplitTrainMethod;
  }

  private static volatile io.grpc.MethodDescriptor<FLProto.RegisterRequest,
      FLProto.RegisterResponse> getRegisterMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "Register",
      requestType = FLProto.RegisterRequest.class,
      responseType = FLProto.RegisterResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<FLProto.RegisterRequest,
      FLProto.RegisterResponse> getRegisterMethod() {
    io.grpc.MethodDescriptor<FLProto.RegisterRequest, FLProto.RegisterResponse> getRegisterMethod;
    if ((getRegisterMethod = GBServiceGrpc.getRegisterMethod) == null) {
      synchronized (GBServiceGrpc.class) {
        if ((getRegisterMethod = GBServiceGrpc.getRegisterMethod) == null) {
          GBServiceGrpc.getRegisterMethod = getRegisterMethod =
              io.grpc.MethodDescriptor.<FLProto.RegisterRequest, FLProto.RegisterResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "Register"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.RegisterRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.RegisterResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GBServiceMethodDescriptorSupplier("Register"))
              .build();
        }
      }
    }
    return getRegisterMethod;
  }

  private static volatile io.grpc.MethodDescriptor<FLProto.UploadTreeEvalRequest,
      FLProto.UploadResponse> getUploadTreeEvalMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "UploadTreeEval",
      requestType = FLProto.UploadTreeEvalRequest.class,
      responseType = FLProto.UploadResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<FLProto.UploadTreeEvalRequest,
      FLProto.UploadResponse> getUploadTreeEvalMethod() {
    io.grpc.MethodDescriptor<FLProto.UploadTreeEvalRequest, FLProto.UploadResponse> getUploadTreeEvalMethod;
    if ((getUploadTreeEvalMethod = GBServiceGrpc.getUploadTreeEvalMethod) == null) {
      synchronized (GBServiceGrpc.class) {
        if ((getUploadTreeEvalMethod = GBServiceGrpc.getUploadTreeEvalMethod) == null) {
          GBServiceGrpc.getUploadTreeEvalMethod = getUploadTreeEvalMethod =
              io.grpc.MethodDescriptor.<FLProto.UploadTreeEvalRequest, FLProto.UploadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "UploadTreeEval"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.UploadTreeEvalRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.UploadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GBServiceMethodDescriptorSupplier("UploadTreeEval"))
              .build();
        }
      }
    }
    return getUploadTreeEvalMethod;
  }

  private static volatile io.grpc.MethodDescriptor<FLProto.UploadTreeLeavesRequest,
      FLProto.UploadResponse> getUploadTreeLeavesMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "UploadTreeLeaves",
      requestType = FLProto.UploadTreeLeavesRequest.class,
      responseType = FLProto.UploadResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<FLProto.UploadTreeLeavesRequest,
      FLProto.UploadResponse> getUploadTreeLeavesMethod() {
    io.grpc.MethodDescriptor<FLProto.UploadTreeLeavesRequest, FLProto.UploadResponse> getUploadTreeLeavesMethod;
    if ((getUploadTreeLeavesMethod = GBServiceGrpc.getUploadTreeLeavesMethod) == null) {
      synchronized (GBServiceGrpc.class) {
        if ((getUploadTreeLeavesMethod = GBServiceGrpc.getUploadTreeLeavesMethod) == null) {
          GBServiceGrpc.getUploadTreeLeavesMethod = getUploadTreeLeavesMethod =
              io.grpc.MethodDescriptor.<FLProto.UploadTreeLeavesRequest, FLProto.UploadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "UploadTreeLeaves"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.UploadTreeLeavesRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.UploadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GBServiceMethodDescriptorSupplier("UploadTreeLeaves"))
              .build();
        }
      }
    }
    return getUploadTreeLeavesMethod;
  }

  private static volatile io.grpc.MethodDescriptor<FLProto.PredictTreeRequest,
      FLProto.PredictTreeResponse> getPredictTreeMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "PredictTree",
      requestType = FLProto.PredictTreeRequest.class,
      responseType = FLProto.PredictTreeResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<FLProto.PredictTreeRequest,
      FLProto.PredictTreeResponse> getPredictTreeMethod() {
    io.grpc.MethodDescriptor<FLProto.PredictTreeRequest, FLProto.PredictTreeResponse> getPredictTreeMethod;
    if ((getPredictTreeMethod = GBServiceGrpc.getPredictTreeMethod) == null) {
      synchronized (GBServiceGrpc.class) {
        if ((getPredictTreeMethod = GBServiceGrpc.getPredictTreeMethod) == null) {
          GBServiceGrpc.getPredictTreeMethod = getPredictTreeMethod =
              io.grpc.MethodDescriptor.<FLProto.PredictTreeRequest, FLProto.PredictTreeResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "PredictTree"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.PredictTreeRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  FLProto.PredictTreeResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GBServiceMethodDescriptorSupplier("PredictTree"))
              .build();
        }
      }
    }
    return getPredictTreeMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static GBServiceStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<GBServiceStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<GBServiceStub>() {
        @Override
        public GBServiceStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new GBServiceStub(channel, callOptions);
        }
      };
    return GBServiceStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static GBServiceBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<GBServiceBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<GBServiceBlockingStub>() {
        @Override
        public GBServiceBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new GBServiceBlockingStub(channel, callOptions);
        }
      };
    return GBServiceBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static GBServiceFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<GBServiceFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<GBServiceFutureStub>() {
        @Override
        public GBServiceFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new GBServiceFutureStub(channel, callOptions);
        }
      };
    return GBServiceFutureStub.newStub(factory, channel);
  }

  /**
   */
  public static abstract class GBServiceImplBase implements io.grpc.BindableService {

    /**
     */
    public void uploadSplitTrain(FLProto.UploadSplitRequest request,
                                 io.grpc.stub.StreamObserver<FLProto.UploadResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getUploadSplitTrainMethod(), responseObserver);
    }

    /**
     */
    public void downloadSplitTrain(FLProto.DownloadSplitRequest request,
                                   io.grpc.stub.StreamObserver<FLProto.DownloadSplitResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getDownloadSplitTrainMethod(), responseObserver);
    }

    /**
     */
    public void register(FLProto.RegisterRequest request,
                         io.grpc.stub.StreamObserver<FLProto.RegisterResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getRegisterMethod(), responseObserver);
    }

    /**
     */
    public void uploadTreeEval(FLProto.UploadTreeEvalRequest request,
                               io.grpc.stub.StreamObserver<FLProto.UploadResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getUploadTreeEvalMethod(), responseObserver);
    }

    /**
     */
    public void uploadTreeLeaves(FLProto.UploadTreeLeavesRequest request,
                                 io.grpc.stub.StreamObserver<FLProto.UploadResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getUploadTreeLeavesMethod(), responseObserver);
    }

    /**
     */
    public void predictTree(FLProto.PredictTreeRequest request,
                            io.grpc.stub.StreamObserver<FLProto.PredictTreeResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getPredictTreeMethod(), responseObserver);
    }

    @Override public final io.grpc.ServerServiceDefinition bindService() {
      return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
          .addMethod(
            getUploadSplitTrainMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                FLProto.UploadSplitRequest,
                FLProto.UploadResponse>(
                  this, METHODID_UPLOAD_SPLIT_TRAIN)))
          .addMethod(
            getDownloadSplitTrainMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                FLProto.DownloadSplitRequest,
                FLProto.DownloadSplitResponse>(
                  this, METHODID_DOWNLOAD_SPLIT_TRAIN)))
          .addMethod(
            getRegisterMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                FLProto.RegisterRequest,
                FLProto.RegisterResponse>(
                  this, METHODID_REGISTER)))
          .addMethod(
            getUploadTreeEvalMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                FLProto.UploadTreeEvalRequest,
                FLProto.UploadResponse>(
                  this, METHODID_UPLOAD_TREE_EVAL)))
          .addMethod(
            getUploadTreeLeavesMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                FLProto.UploadTreeLeavesRequest,
                FLProto.UploadResponse>(
                  this, METHODID_UPLOAD_TREE_LEAVES)))
          .addMethod(
            getPredictTreeMethod(),
            io.grpc.stub.ServerCalls.asyncUnaryCall(
              new MethodHandlers<
                FLProto.PredictTreeRequest,
                FLProto.PredictTreeResponse>(
                  this, METHODID_PREDICT_TREE)))
          .build();
    }
  }

  /**
   */
  public static final class GBServiceStub extends io.grpc.stub.AbstractAsyncStub<GBServiceStub> {
    private GBServiceStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected GBServiceStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new GBServiceStub(channel, callOptions);
    }

    /**
     */
    public void uploadSplitTrain(FLProto.UploadSplitRequest request,
                                 io.grpc.stub.StreamObserver<FLProto.UploadResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUploadSplitTrainMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void downloadSplitTrain(FLProto.DownloadSplitRequest request,
                                   io.grpc.stub.StreamObserver<FLProto.DownloadSplitResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getDownloadSplitTrainMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void register(FLProto.RegisterRequest request,
                         io.grpc.stub.StreamObserver<FLProto.RegisterResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getRegisterMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void uploadTreeEval(FLProto.UploadTreeEvalRequest request,
                               io.grpc.stub.StreamObserver<FLProto.UploadResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUploadTreeEvalMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void uploadTreeLeaves(FLProto.UploadTreeLeavesRequest request,
                                 io.grpc.stub.StreamObserver<FLProto.UploadResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUploadTreeLeavesMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void predictTree(FLProto.PredictTreeRequest request,
                            io.grpc.stub.StreamObserver<FLProto.PredictTreeResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getPredictTreeMethod(), getCallOptions()), request, responseObserver);
    }
  }

  /**
   */
  public static final class GBServiceBlockingStub extends io.grpc.stub.AbstractBlockingStub<GBServiceBlockingStub> {
    private GBServiceBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected GBServiceBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new GBServiceBlockingStub(channel, callOptions);
    }

    /**
     */
    public FLProto.UploadResponse uploadSplitTrain(FLProto.UploadSplitRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getUploadSplitTrainMethod(), getCallOptions(), request);
    }

    /**
     */
    public FLProto.DownloadSplitResponse downloadSplitTrain(FLProto.DownloadSplitRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getDownloadSplitTrainMethod(), getCallOptions(), request);
    }

    /**
     */
    public FLProto.RegisterResponse register(FLProto.RegisterRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getRegisterMethod(), getCallOptions(), request);
    }

    /**
     */
    public FLProto.UploadResponse uploadTreeEval(FLProto.UploadTreeEvalRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getUploadTreeEvalMethod(), getCallOptions(), request);
    }

    /**
     */
    public FLProto.UploadResponse uploadTreeLeaves(FLProto.UploadTreeLeavesRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getUploadTreeLeavesMethod(), getCallOptions(), request);
    }

    /**
     */
    public FLProto.PredictTreeResponse predictTree(FLProto.PredictTreeRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getPredictTreeMethod(), getCallOptions(), request);
    }
  }

  /**
   */
  public static final class GBServiceFutureStub extends io.grpc.stub.AbstractFutureStub<GBServiceFutureStub> {
    private GBServiceFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected GBServiceFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new GBServiceFutureStub(channel, callOptions);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<FLProto.UploadResponse> uploadSplitTrain(
        FLProto.UploadSplitRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getUploadSplitTrainMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<FLProto.DownloadSplitResponse> downloadSplitTrain(
        FLProto.DownloadSplitRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getDownloadSplitTrainMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<FLProto.RegisterResponse> register(
        FLProto.RegisterRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getRegisterMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<FLProto.UploadResponse> uploadTreeEval(
        FLProto.UploadTreeEvalRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getUploadTreeEvalMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<FLProto.UploadResponse> uploadTreeLeaves(
        FLProto.UploadTreeLeavesRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getUploadTreeLeavesMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<FLProto.PredictTreeResponse> predictTree(
        FLProto.PredictTreeRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getPredictTreeMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_UPLOAD_SPLIT_TRAIN = 0;
  private static final int METHODID_DOWNLOAD_SPLIT_TRAIN = 1;
  private static final int METHODID_REGISTER = 2;
  private static final int METHODID_UPLOAD_TREE_EVAL = 3;
  private static final int METHODID_UPLOAD_TREE_LEAVES = 4;
  private static final int METHODID_PREDICT_TREE = 5;

  private static final class MethodHandlers<Req, Resp> implements
      io.grpc.stub.ServerCalls.UnaryMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ServerStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ClientStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.BidiStreamingMethod<Req, Resp> {
    private final GBServiceImplBase serviceImpl;
    private final int methodId;

    MethodHandlers(GBServiceImplBase serviceImpl, int methodId) {
      this.serviceImpl = serviceImpl;
      this.methodId = methodId;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void invoke(Req request, io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_UPLOAD_SPLIT_TRAIN:
          serviceImpl.uploadSplitTrain((FLProto.UploadSplitRequest) request,
              (io.grpc.stub.StreamObserver<FLProto.UploadResponse>) responseObserver);
          break;
        case METHODID_DOWNLOAD_SPLIT_TRAIN:
          serviceImpl.downloadSplitTrain((FLProto.DownloadSplitRequest) request,
              (io.grpc.stub.StreamObserver<FLProto.DownloadSplitResponse>) responseObserver);
          break;
        case METHODID_REGISTER:
          serviceImpl.register((FLProto.RegisterRequest) request,
              (io.grpc.stub.StreamObserver<FLProto.RegisterResponse>) responseObserver);
          break;
        case METHODID_UPLOAD_TREE_EVAL:
          serviceImpl.uploadTreeEval((FLProto.UploadTreeEvalRequest) request,
              (io.grpc.stub.StreamObserver<FLProto.UploadResponse>) responseObserver);
          break;
        case METHODID_UPLOAD_TREE_LEAVES:
          serviceImpl.uploadTreeLeaves((FLProto.UploadTreeLeavesRequest) request,
              (io.grpc.stub.StreamObserver<FLProto.UploadResponse>) responseObserver);
          break;
        case METHODID_PREDICT_TREE:
          serviceImpl.predictTree((FLProto.PredictTreeRequest) request,
              (io.grpc.stub.StreamObserver<FLProto.PredictTreeResponse>) responseObserver);
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

  private static abstract class GBServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    GBServiceBaseDescriptorSupplier() {}

    @Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return FLProto.getDescriptor();
    }

    @Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("GBService");
    }
  }

  private static final class GBServiceFileDescriptorSupplier
      extends GBServiceBaseDescriptorSupplier {
    GBServiceFileDescriptorSupplier() {}
  }

  private static final class GBServiceMethodDescriptorSupplier
      extends GBServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final String methodName;

    GBServiceMethodDescriptorSupplier(String methodName) {
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
      synchronized (GBServiceGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new GBServiceFileDescriptorSupplier())
              .addMethod(getUploadSplitTrainMethod())
              .addMethod(getDownloadSplitTrainMethod())
              .addMethod(getRegisterMethod())
              .addMethod(getUploadTreeEvalMethod())
              .addMethod(getUploadTreeLeavesMethod())
              .addMethod(getPredictTreeMethod())
              .build();
        }
      }
    }
    return result;
  }
}
