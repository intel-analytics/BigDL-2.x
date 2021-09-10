package com.intel.analytics.zoo.ppml.vfl;

import com.intel.analytics.zoo.ppml.FLClient;
import com.intel.analytics.zoo.ppml.generated.FLProto;
import com.intel.analytics.zoo.ppml.generated.PSIServiceGrpc;
import com.intel.analytics.zoo.ppml.psi.Utils;
import io.grpc.StatusRuntimeException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

public class PSIStub {
    private static final Logger logger = LoggerFactory.getLogger(PSIStub.class);
    public static PSIServiceGrpc.PSIServiceBlockingStub stub;

    protected String clientID = UUID.randomUUID().toString();
    protected String salt;
    protected int splitSize = 1000000;
    protected String taskID;

    public String getSalt() {
        if (this.taskID.isEmpty()) {
            this.taskID = Utils.getRandomUUID();
        }
        return getSalt(this.taskID, 2, "Test");
    }

    /**
     * For PSI usage only
     * To get salt from FL Server, will get a new one if its salt does not exist on server
     * @param name String, taskID
     * @param clientNum int, client number
     * @param secureCode String, secure code
     * @return String, the salt get from server
     */
    public String getSalt(String name, int clientNum, String secureCode) {
        logger.info("Processing task with taskID: " + name + " ...");
        FLProto.SaltRequest request = FLProto.SaltRequest.newBuilder()
                .setTaskId(name)
                .setClientNum(clientNum)
                .setSecureCode(secureCode).build();
        FLProto.SaltReply response;
        try {
            response = stub.getSalt(request);
        } catch (StatusRuntimeException e) {
            throw new RuntimeException("RPC failed: " + e.getMessage());
        }
        if (!response.getSaltReply().isEmpty()) {
            salt = response.getSaltReply();
        }
        return response.getSaltReply();
    }

    /**
     * For PSI usage only
     * Upload local set to FL Server in VFL
     * @param hashedIdArray List of String, the set trained at local
     */
    public void uploadSet(List<String> hashedIdArray) {
        int numSplit = Utils.getTotalSplitNum(hashedIdArray, splitSize);
        int split = 0;
        while (split < numSplit) {
            List<String> splitArray = Utils.getSplit(hashedIdArray, split, numSplit, splitSize);
            FLProto.UploadSetRequest request = FLProto.UploadSetRequest.newBuilder()
                    .setTaskId(taskID)
                    .setSplit(split)
                    .setNumSplit(numSplit)
                    .setSplitLength(splitSize)
                    .setTotalLength(hashedIdArray.size())
                    .setClientId(clientID)
                    .addAllHashedID(splitArray)
                    .build();
            try {
                stub.uploadSet(request);
            } catch (StatusRuntimeException e) {
                throw new RuntimeException("RPC failed: " + e.getMessage());
            }
            split ++;
        }
    }

    /**
     * For PSI usage only
     * Download intersection from FL Server in VFL
     * @return List of String, the intersection downloaded
     */
    public List<String> downloadIntersection() {
        List<String> result = new ArrayList<String>();
        try {
            logger.info("Downloading 0th intersection");
            FLProto.DownloadIntersectionRequest request = FLProto.DownloadIntersectionRequest.newBuilder()
                    .setTaskId(taskID)
                    .setSplit(0)
                    .build();
            FLProto.DownloadIntersectionResponse response = stub.downloadIntersection(request);
            logger.info("Downloaded 0th intersection");
            result.addAll(response.getIntersectionList());
            for (int i = 1; i < response.getNumSplit(); i++) {
                request = FLProto.DownloadIntersectionRequest.newBuilder()
                        .setTaskId(taskID)
                        .setSplit(i)
                        .build();
                logger.info("Downloading " + i + "th intersection");
                response = stub.downloadIntersection(request);
                logger.info("Downloaded " + i + "th intersection");
                result.addAll(response.getIntersectionList());
            }
            assert(result.size() == response.getTotalLength());
        } catch (StatusRuntimeException e) {
            throw new RuntimeException("RPC failed: " + e.getMessage());
        }
        return result;
    }
}
