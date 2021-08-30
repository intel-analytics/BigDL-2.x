/*
 * Copyright 2021 The Analytic Zoo Authors
 *
 * Licensed under the Apache License,  Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,  software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,  either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


package com.intel.analytics.zoo.ppml.ps;

import com.intel.analytics.zoo.ppml.generated.FLProto.*;
import com.intel.analytics.zoo.ppml.generated.ParameterServerServiceGrpc;
import io.grpc.stub.StreamObserver;
import io.netty.handler.ssl.ClientAuth;
import io.netty.handler.ssl.SslContextBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.List;


public class ParameterServerServiceImpl
        extends ParameterServerServiceGrpc.ParameterServerServiceImplBase {
    private static final Logger logger = LoggerFactory.getLogger(ParameterServerServiceImpl.class);
    private final Aggregator aggregator;

    public ParameterServerServiceImpl(Aggregator aggregator) {
        this.aggregator = aggregator;
    }

    @Override
    public void downloadTrain(
            DownloadRequest request, StreamObserver<DownloadResponse> responseObserver) {
        int version = request.getMetaData().getVersion();
        Table data = aggregator.getTrainResult(version);
        if (data == null) {
            String response = "Your required data doesn't exist";
            responseObserver.onNext(DownloadResponse.newBuilder().setResponse(response).setCode(0).build());
            responseObserver.onCompleted();
        } else {
            String response = "Download data successfully";
            responseObserver.onNext(
                    DownloadResponse.newBuilder().setResponse(response).setData(data).setCode(1).build());
            responseObserver.onCompleted();
        }
    }

    @Override
    public void uploadTrain(
            UploadRequest request, StreamObserver<UploadResponse> responseObserver) {
        // check data version, drop all the unmatched version

        String clientUUID = request.getClientuuid();
        Table data = request.getData();
        int version = data.getMetaData().getVersion();

        try {
            aggregator.putData(clientUUID, version, data);
            UploadResponse response = UploadResponse.newBuilder().setResponse("Data received").setCode(0).build();
            responseObserver.onNext(response);
            responseObserver.onCompleted();
        } catch (Exception e) {
            UploadResponse response = UploadResponse.newBuilder().setResponse(e.getMessage()).setCode(1).build();
            responseObserver.onNext(response);
            responseObserver.onCompleted();
        } finally {

        }
    }

    @Override
    public void uploadTreeLeaves(UploadTreeLeavesRequest request,
                                 StreamObserver<UploadResponse> responseObserver) {
        String clientUUID = request.getClientuuid();
        TreeLeaves treeLeaves = request.getTreeLeaves();

        try {
            aggregator.putTreeLeaves(clientUUID, treeLeaves);
            UploadResponse response = UploadResponse.newBuilder().setResponse("Data received").setCode(0).build();
            responseObserver.onNext(response);
            responseObserver.onCompleted();
        } catch (Exception e) {
            UploadResponse response = UploadResponse.newBuilder().setResponse(e.getMessage()).setCode(1).build();
            responseObserver.onNext(response);
            responseObserver.onCompleted();
        } finally {

        }
    }


    @Override
    public void uploadSplitTrain(UploadSplitRequest request,
                                 StreamObserver<UploadResponse> responseObserver) {
        String clientUUID = request.getClientuuid();
        DataSplit split = request.getSplit();

        try {
            aggregator.putSplit(clientUUID, split);
            UploadResponse response = UploadResponse
                    .newBuilder().setResponse("Data received").setCode(0).build();
            responseObserver.onNext(response);
            responseObserver.onCompleted();
        } catch (Exception e) {
            UploadResponse response = UploadResponse
                    .newBuilder().setResponse(e.getMessage()).setCode(1).build();
            responseObserver.onNext(response);
            responseObserver.onCompleted();
        } finally {

        }
    }

    @Override
    public void downloadSplitTrain(DownloadSplitRequest request,
                                   StreamObserver<DownloadSplitResponse> responseObserver) {
        String treeID = request.getTreeID();
        String nodeID = request.getNodeID();
        DataSplit split = aggregator.getBestSplit(treeID, nodeID);
        if (split == null) {
            String response = "Your required data doesn't exist";
            responseObserver.onNext(DownloadSplitResponse.newBuilder().setResponse(response).setCode(0).build());
            responseObserver.onCompleted();
        } else {
            String response = "Download data successfully";
            responseObserver.onNext(
                    DownloadSplitResponse.newBuilder().setResponse(response).setSplit(split).setCode(1).build());
            responseObserver.onCompleted();
        }
    }


    @Override
    public void uploadTreeEval(UploadTreeEvalRequest request,
                               StreamObserver<UploadResponse> responseObserver) {
        String clientUUID = request.getClientuuid();
        int version = request.getVersion();
        List<BoostEval> predicts = request.getTreeEvalList();

        //TODO: minibatch
        boolean last = false;
        aggregator.putEvaluateData(clientUUID, predicts, last);
//            UploadResponse response;
//            if (last) {
//                Table result = aggregator.getEvaluateResult(version);
//                // TODO
//                response = UploadResponse.newBuilder()
//                  .setCode(0)
//                  .setResponse("Evaluate successfully.")
//                  .build();
//            } else {
//                response = UploadResponse.newBuilder()
//                  .setResponse("Adding to evaluate pipeline.").build();
//            }

        UploadResponse response = UploadResponse.newBuilder()
                .setCode(0)
                .setResponse("Evaluate successfully.")
                .build();

        responseObserver.onNext(response);
        responseObserver.onCompleted();

    }


    @Override
    public void uploadEvaluate(
            EvaluateRequest request, StreamObserver<EvaluateResponse> responseObserver) {
        String clientUUID = request.getClientuuid();
        Table data = request.getData();
        int version = data.getMetaData().getVersion();
        boolean last = request.getLast();

        aggregator.putEvaluateData(clientUUID, data, last);
        EvaluateResponse response;
        if (last) {
            Table result = aggregator.getEvaluateResult(version);
            response = EvaluateResponse.newBuilder()
                    .setData(result)
                    .setResponse("Evaluate successfully.")
                    .build();
        } else {
            response = EvaluateResponse.newBuilder()
                    .setResponse("Adding to evaluate pipeline.").build();
        }

        responseObserver.onNext(response);
        responseObserver.onCompleted();
    }

    @Override
    public void predictTree(PredictTreeRequest request,
                            StreamObserver<PredictTreeResponse> responseObserver) {
        String clientUUID = request.getClientuuid();
        logger.info(clientUUID + " calling");
        List<BoostEval> predicts = request.getTreeEvalList();

        int version = aggregator.putPredict(clientUUID, predicts);
        logger.info("agg version: " + version);
        Table result = aggregator.getPredictResult(version);
        PredictTreeResponse response = PredictTreeResponse.newBuilder()
                .setResult(result)
                .build();
        responseObserver.onNext(response);
        responseObserver.onCompleted();
    }

    @Override
    public void register(RegisterRequest request,
                         StreamObserver<RegisterResponse> responseObserver) {
        boolean res = aggregator.registerClient(request.getClientuuid());
        int errCode = 0;
        String response = "The client registers successfully";
        if (!res) {
            errCode = 1; // means register failed
            response = "The client uuid has existed in aggregator";
        }
        responseObserver.onNext(
                RegisterResponse.newBuilder().setCode(errCode).setResponse(response).build());
        responseObserver.onCompleted();
    }
}

