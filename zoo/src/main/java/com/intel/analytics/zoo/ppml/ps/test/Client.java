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


package com.intel.analytics.zoo.ppml.ps.test;


import com.intel.analytics.zoo.ppml.generated.FLProto.*;
import com.intel.analytics.zoo.ppml.generated.ParameterServerServiceGrpc;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
import io.grpc.netty.shaded.io.netty.handler.ssl.SslContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.net.ssl.SSLException;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

import static com.intel.analytics.zoo.ppml.generated.ParameterServerServiceGrpc.newBlockingStub;


public class Client {
    private static final Logger logger = LoggerFactory.getLogger(Client.class);

    private final String clientUUID;
    private final ParameterServerServiceGrpc.ParameterServerServiceBlockingStub blockingStub;
    private final ParameterServerServiceGrpc.ParameterServerServiceStub asyncStub;
    private ManagedChannel channel;

    public Client(String target) {
        channel = ManagedChannelBuilder.forTarget(target)
            // Channels are secure by default (via SSL/TLS).
            //extend message size of server to 200M to avoid size conflict
            .maxInboundMessageSize(200 * 1024 * 1024)
            .usePlaintext()
            .build();
        blockingStub = newBlockingStub(channel);
        asyncStub = ParameterServerServiceGrpc.newStub(channel);
        clientUUID = UUID.randomUUID().toString();
    }

    public Client(String target, SslContext sslContext) throws SSLException {
        channel = NettyChannelBuilder.forTarget(target)
                .overrideAuthority("foo.test.google.fr") // for testing
                // Channels are secure by default (via SSL/TLS).
                //extend message size of server to 200M to avoid size conflict
                .maxInboundMessageSize(200 * 1024 * 1024)
                .sslContext(sslContext)
                .build();
        blockingStub = newBlockingStub(channel);
        asyncStub = ParameterServerServiceGrpc.newStub(channel);
        clientUUID = UUID.randomUUID().toString();
    }

    public String getClientUUID() {
        return clientUUID;
    }

    public void shutdown() {
        try {
            channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            logger.error("Shutdown Client Error" + e.getMessage());
        }
    }


    public DownloadResponse downloadData(String modelName, int flVersion) {
        logger.info("Download the following data:");
        TableMetaData metadata = TableMetaData.newBuilder()
                .setName(modelName).setVersion(flVersion + 1).build();
        DownloadRequest downloadRequest = DownloadRequest.newBuilder().setMetaData(metadata).build();
        return blockingStub.downloadTrain(downloadRequest);
    }

    public UploadResponse uploadData(Table data) {

        UploadRequest uploadRequest = UploadRequest
                .newBuilder()
                .setData(data)
                .setClientuuid(clientUUID)
                .build();

        logger.info("Upload the following data:");
        logger.info("Upload Data Name:" + data.getMetaData().getName());
        logger.info("Upload Data Version:" + data.getMetaData().getVersion());
        logger.debug("Upload Data" + data.getTableMap());
//        logger.info("Upload" + data.getTableMap().get("weights").getTensorList().subList(0, 5));

        UploadResponse uploadResponse = blockingStub.uploadTrain(uploadRequest);
        return uploadResponse;
    }

    public EvaluateResponse evaluate(Table data, boolean lastBatch) {
        EvaluateRequest eRequest = EvaluateRequest
          .newBuilder()
          .setData(data)
          .setClientuuid(clientUUID)
          .setLast(lastBatch)
          .build();

        return blockingStub.uploadEvaluate(eRequest);
    }

    public UploadResponse uploadSplit(DataSplit ds) {
        UploadSplitRequest uploadRequest = UploadSplitRequest
                .newBuilder()
                .setSplit(ds)
                .setClientuuid(clientUUID)
                .build();

        return  blockingStub.uploadSplitTrain(uploadRequest);
    }

    /***
     * XGBoost download aggregated best split
     * @param treeID
     * @return
     */
    public DownloadSplitResponse downloadSplit(
            String treeID,
            String nodeID) {
        DownloadSplitRequest downloadRequest = DownloadSplitRequest
                .newBuilder()
                .setTreeID(treeID)
                .setNodeID(nodeID)
                .setClientuuid(clientUUID)
                .build();
        return blockingStub.downloadSplitTrain(downloadRequest);
    }

    public UploadResponse uploadTreeEval(
            List<BoostEval> boostEval) {
        UploadTreeEvalRequest uploadTreeEvalRequest = UploadTreeEvalRequest
          .newBuilder()
          .setClientuuid(clientUUID)
          .addAllTreeEval(boostEval)
          .build();

        return blockingStub.uploadTreeEval(uploadTreeEvalRequest);
    }

    public PredictTreeResponse uploadTreePred(
      List<BoostEval> boostEval) {
        PredictTreeRequest request = PredictTreeRequest
          .newBuilder()
          .setClientuuid(clientUUID)
          .addAllTreeEval(boostEval)
          .build();

        return blockingStub.predictTree(request);
    }


    public UploadResponse uploadTreeLeaves(
            String treeID,
            List<Integer> treeIndexes,
            List<Float> treeOutput
    ) {
        TreeLeaves treeLeaves = TreeLeaves
                .newBuilder()
                .setTreeID(treeID)
                .addAllLeafIndex(treeIndexes)
                .addAllLeafOutput(treeOutput)
                .build();
        UploadTreeLeavesRequest uploadTreeLeavesRequest = UploadTreeLeavesRequest
                .newBuilder()
                .setClientuuid(clientUUID)
                .setTreeLeaves(treeLeaves)
                .build();
        return  blockingStub.uploadTreeLeaves(uploadTreeLeavesRequest);
    }


    public static void main(String[] args) {
        Client client = new Client("localhost:15551");
        try {
            client.uploadData(TestUtils.genRandomData(0));
            client.downloadData("first_model", 1);
        } finally {
            client.shutdown();
        }
    }
}
