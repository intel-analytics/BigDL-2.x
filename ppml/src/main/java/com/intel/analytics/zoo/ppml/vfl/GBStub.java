/*
 * Copyright 2021 Analytics Zoo Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.zoo.ppml.vfl;

import com.intel.analytics.zoo.ppml.generated.FLProto;
import com.intel.analytics.zoo.ppml.generated.ParameterServerServiceGrpc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class GBStub {
    private static final Logger logger = LoggerFactory.getLogger(GBStub.class);
    GBStub(String clientID) {
        this.clientID = clientID;
    }
    protected final String clientID;
    private static ParameterServerServiceGrpc.ParameterServerServiceBlockingStub stub;
    public FLProto.UploadResponse uploadSplit(FLProto.DataSplit ds) {
        FLProto.UploadSplitRequest uploadRequest = FLProto.UploadSplitRequest
                .newBuilder()
                .setSplit(ds)
                .setClientuuid(clientID)
                .build();

        return stub.uploadSplitTrain(uploadRequest);
    }

    /***
     * XGBoost download aggregated best split
     * @param treeID
     * @return
     */
    public FLProto.DownloadSplitResponse downloadSplit(
            String treeID,
            String nodeID) {
        FLProto.DownloadSplitRequest downloadRequest = FLProto.DownloadSplitRequest
                .newBuilder()
                .setTreeID(treeID)
                .setNodeID(nodeID)
                .setClientuuid(clientID)
                .build();
        return stub.downloadSplitTrain(downloadRequest);
    }

    public FLProto.UploadResponse uploadTreeEval(
            List<FLProto.BoostEval> boostEval) {
        FLProto.UploadTreeEvalRequest uploadTreeEvalRequest = FLProto.UploadTreeEvalRequest
                .newBuilder()
                .setClientuuid(clientID)
                .addAllTreeEval(boostEval)
                .build();

        return stub.uploadTreeEval(uploadTreeEvalRequest);
    }

    public FLProto.PredictTreeResponse uploadTreePred(
            List<FLProto.BoostEval> boostEval) {
        FLProto.PredictTreeRequest request = FLProto.PredictTreeRequest
                .newBuilder()
                .setClientuuid(clientID)
                .addAllTreeEval(boostEval)
                .build();

        return stub.predictTree(request);
    }


    public FLProto.UploadResponse uploadTreeLeaves(
            String treeID,
            List<Integer> treeIndexes,
            List<Float> treeOutput
    ) {
        FLProto.TreeLeaves treeLeaves = FLProto.TreeLeaves
                .newBuilder()
                .setTreeID(treeID)
                .addAllLeafIndex(treeIndexes)
                .addAllLeafOutput(treeOutput)
                .build();
        FLProto.UploadTreeLeavesRequest uploadTreeLeavesRequest = FLProto.UploadTreeLeavesRequest
                .newBuilder()
                .setClientuuid(clientID)
                .setTreeLeaves(treeLeaves)
                .build();
        return  stub.uploadTreeLeaves(uploadTreeLeavesRequest);
    }
}
