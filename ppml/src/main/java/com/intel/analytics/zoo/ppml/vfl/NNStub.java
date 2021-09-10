package com.intel.analytics.zoo.ppml.vfl;

import com.intel.analytics.zoo.ppml.FLClient;
import com.intel.analytics.zoo.ppml.generated.FLProto;
import com.intel.analytics.zoo.ppml.generated.ParameterServerServiceGrpc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class NNStub {
    private static final Logger logger = LoggerFactory.getLogger(NNStub.class);
    private static ParameterServerServiceGrpc.ParameterServerServiceBlockingStub stub;
    String clientID;
    NNStub(String clientID) {
        this.clientID = clientID;
    }
    public FLProto.DownloadResponse downloadTrain(String modelName, int flVersion) {
        logger.info("Download the following data:");
        FLProto.TableMetaData metadata = FLProto.TableMetaData.newBuilder()
                .setName(modelName).setVersion(flVersion + 1).build();
        FLProto.DownloadRequest downloadRequest = FLProto.DownloadRequest.newBuilder().setMetaData(metadata).build();
        return stub.downloadTrain(downloadRequest);
    }

    public FLProto.UploadResponse uploadTrain(FLProto.Table data) {

        FLProto.UploadRequest uploadRequest = FLProto.UploadRequest
                .newBuilder()
                .setData(data)
                .setClientuuid(clientID)
                .build();

        logger.info("Upload the following data:");
        logger.info("Upload Data Name:" + data.getMetaData().getName());
        logger.info("Upload Data Version:" + data.getMetaData().getVersion());
        logger.debug("Upload Data" + data.getTableMap());
//        logger.info("Upload" + data.getTableMap().get("weights").getTensorList().subList(0, 5));

        FLProto.UploadResponse uploadResponse = stub.uploadTrain(uploadRequest);
        return uploadResponse;
    }

    public FLProto.EvaluateResponse evaluate(FLProto.Table data, boolean lastBatch) {
        FLProto.EvaluateRequest eRequest = FLProto.EvaluateRequest
                .newBuilder()
                .setData(data)
                .setClientuuid(clientID)
                .setLast(lastBatch)
                .build();

        return stub.uploadEvaluate(eRequest);
    }
}
