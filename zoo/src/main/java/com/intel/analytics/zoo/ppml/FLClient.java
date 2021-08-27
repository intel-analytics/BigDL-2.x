package com.intel.analytics.zoo.ppml;

import com.intel.analytics.zoo.grpc.ZooGrpcClient;
import com.intel.analytics.zoo.ppml.generated.FLProto.*;
import com.intel.analytics.zoo.ppml.generated.PSIServiceGrpc;
import com.intel.analytics.zoo.ppml.generated.ParameterServerServiceGrpc;
import com.intel.analytics.zoo.ppml.psi.Utils;
import io.grpc.StatusRuntimeException;
import org.apache.commons.cli.Option;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class FLClient extends ZooGrpcClient {

    protected String taskID;
    protected String clientID;
    protected String salt;
    protected int splitSize = 1000000;
    private static PSIServiceGrpc.PSIServiceBlockingStub blockingStubPSI;
    private static ParameterServerServiceGrpc.ParameterServerServiceBlockingStub blockingStubPS;
    public FLClient(String[] args) {
        super(args);

        options.addOption(new Option(
                "tid", "taskId", true, "."));
    }

    public void loadService() {

        assert services.length > 0;
        for (String service : services) {
            if (service == "psi") {
                blockingStubPSI = PSIServiceGrpc.newBlockingStub(channel);
            } else if (service == "ps") {
                blockingStubPS = ParameterServerServiceGrpc.newBlockingStub(channel);
            } else {
                logger.warn("Type is not supported, skipped. Type: " + service);
            }
        }

    }
    public String getSalt() {
        if (this.taskID.isEmpty()) {
            this.taskID = Utils.getRandomUUID();
        }
        return getSalt(this.taskID, 2, "Test");
    }

    public String getSalt(String name, int clientNum, String secureCode) {
        logger.info("Processing task with taskID: " + name + " ...");
        SaltRequest request = SaltRequest.newBuilder()
                .setTaskId(name)
                .setClientNum(clientNum)
                .setSecureCode(secureCode).build();
        SaltReply response;
        try {
            response = blockingStubPSI.getSalt(request);
        } catch (StatusRuntimeException e) {
            throw new RuntimeException("RPC failed: " + e.getMessage());
        }
        if (!response.getSaltReply().isEmpty()) {
            salt = response.getSaltReply();
        }
        return response.getSaltReply();
    }

    public void uploadSet(List<String> hashedIdArray) {
        int numSplit = Utils.getTotalSplitNum(hashedIdArray, splitSize);
        int split = 0;
        while (split < numSplit) {
            List<String> splitArray = Utils.getSplit(hashedIdArray, split, numSplit, splitSize);
            UploadSetRequest request = UploadSetRequest.newBuilder()
                    .setTaskId(taskID)
                    .setSplit(split)
                    .setNumSplit(numSplit)
                    .setSplitLength(splitSize)
                    .setTotalLength(hashedIdArray.size())
                    .setClientId(clientID)
                    .addAllHashedID(splitArray)
                    .build();
            try {
                blockingStubPSI.uploadSet(request);
            } catch (StatusRuntimeException e) {
                throw new RuntimeException("RPC failed: " + e.getMessage());
            }
            split ++;
        }
    }


    public List<String> downloadIntersection() {
        List<String> result = new ArrayList<String>();
        try {
            logger.info("Downloading 0th intersection");
            DownloadIntersectionRequest request = DownloadIntersectionRequest.newBuilder()
                    .setTaskId(taskID)
                    .setSplit(0)
                    .build();
            DownloadIntersectionResponse response = blockingStubPSI.downloadIntersection(request);
            logger.info("Downloaded 0th intersection");
            result.addAll(response.getIntersectionList());
            for (int i = 1; i < response.getNumSplit(); i++) {
                request = DownloadIntersectionRequest.newBuilder()
                        .setTaskId(taskID)
                        .setSplit(i)
                        .build();
                logger.info("Downloading " + i + "th intersection");
                response = blockingStubPSI.downloadIntersection(request);
                logger.info("Downloaded " + i + "th intersection");
                result.addAll(response.getIntersectionList());
            }
            assert(result.size() == response.getTotalLength());
        } catch (StatusRuntimeException e) {
            throw new RuntimeException("RPC failed: " + e.getMessage());
        }
        return result;
    }

    public void shutdown() {
        try {
            channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            logger.error("Shutdown Client Error" + e.getMessage());
        }
    }


    public DownloadResponse downloadTrain(String modelName, int flVersion) {
        logger.info("Download the following data:");
        TableMetaData metadata = TableMetaData.newBuilder()
                .setName(modelName).setVersion(flVersion + 1).build();
        DownloadRequest downloadRequest = DownloadRequest.newBuilder().setMetaData(metadata).build();
        return blockingStubPS.downloadTrain(downloadRequest);
    }

    public UploadResponse uploadTrain(Table data) {

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

        UploadResponse uploadResponse = blockingStubPS.uploadTrain(uploadRequest);
        return uploadResponse;
    }

    public EvaluateResponse evaluate(Table data, boolean lastBatch) {
        EvaluateRequest eRequest = EvaluateRequest
                .newBuilder()
                .setData(data)
                .setClientuuid(clientUUID)
                .setLast(lastBatch)
                .build();

        return blockingStubPS.uploadEvaluate(eRequest);
    }

    public UploadResponse uploadSplit(DataSplit ds) {
        UploadSplitRequest uploadRequest = UploadSplitRequest
                .newBuilder()
                .setSplit(ds)
                .setClientuuid(clientUUID)
                .build();

        return  blockingStubPS.uploadSplitTrain(uploadRequest);
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
        return blockingStubPS.downloadSplitTrain(downloadRequest);
    }

    public UploadResponse uploadTreeEval(
            List<BoostEval> boostEval) {
        UploadTreeEvalRequest uploadTreeEvalRequest = UploadTreeEvalRequest
                .newBuilder()
                .setClientuuid(clientUUID)
                .addAllTreeEval(boostEval)
                .build();

        return blockingStubPS.uploadTreeEval(uploadTreeEvalRequest);
    }

    public PredictTreeResponse uploadTreePred(
            List<BoostEval> boostEval) {
        PredictTreeRequest request = PredictTreeRequest
                .newBuilder()
                .setClientuuid(clientUUID)
                .addAllTreeEval(boostEval)
                .build();

        return blockingStubPS.predictTree(request);
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
        return  blockingStubPS.uploadTreeLeaves(uploadTreeLeavesRequest);
    }
}
