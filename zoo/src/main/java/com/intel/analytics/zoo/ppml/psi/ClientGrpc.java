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

package com.intel.analytics.zoo.ppml.psi;

import com.intel.analytics.zoo.ppml.psi.generated.*;
import io.grpc.Channel;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.TimeUnit;


public class ClientGrpc {
    private static final Logger logger = LoggerFactory.getLogger(ClientGrpc.class);

    private static PSIServiceGrpc.PSIServiceBlockingStub blockingStub;

    protected String salt;
    protected String taskID;
    protected String clientID;
    protected int splitSize = 1000000;

    /**
     * Construct client for accessing server using the existing channel.
     */
    public ClientGrpc(Channel channel) {
        // 'channel' here is a Channel,  not a ManagedChannel,  so it is not this code's responsibility to
        // shut it down.

        // Passing Channels to code makes code easier to test and makes it easier to reuse Channels.
        this.clientID = SecurityUtils.getRandomUUID();
        blockingStub = PSIServiceGrpc.newBlockingStub(channel);
    }

    public ClientGrpc(Channel channel, String taskID) {
        this.taskID = taskID;
        this.clientID = SecurityUtils.getRandomUUID();
        blockingStub = PSIServiceGrpc.newBlockingStub(channel);
    }

    public String getSalt() {
        if (this.taskID.isEmpty()) {
            this.taskID = SecurityUtils.getRandomUUID();
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
            response = blockingStub.salt(request);
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
            UploadRequest request = UploadRequest.newBuilder()
                    .setTaskId(taskID)
                    .setSplit(split)
                    .setNumSplit(numSplit)
                    .setSplitLength(splitSize)
                    .setTotalLength(hashedIdArray.size())
                    .setClientId(clientID)
                    .addAllHashedID(splitArray)
                    .build();
            try {
                blockingStub.uploadSet(request);
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
            DownloadRequest request = DownloadRequest.newBuilder()
                    .setTaskId(taskID)
                    .setSplit(0)
                    .build();
            DownloadResponse response = blockingStub.downloadIntersection(request);
            logger.info("Downloaded 0th intersection");
            result.addAll(response.getIntersectionList());
            for (int i = 1; i < response.getNumSplit(); i++) {
                request = DownloadRequest.newBuilder()
                        .setTaskId(taskID)
                        .setSplit(i)
                        .build();
                logger.info("Downloading " + i + "th intersection");
                response = blockingStub.downloadIntersection(request);
                logger.info("Downloaded " + i + "th intersection");
                result.addAll(response.getIntersectionList());
            }
            assert(result.size() == response.getTotalLength());
        } catch (StatusRuntimeException e) {
            throw new RuntimeException("RPC failed: " + e.getMessage());
        }
        return result;
    }


    public static void main(String[] args) throws Exception {
        String taskID;
        String target;
        // Number of arguments to be passed.
        int argNum = 3;
        if (args.length == 0) {
            logger.info("No argument passed, using default parameters.");
            taskID = "taskID";
            target = "localhost:50051";
        } else if (args.length < argNum || args.length > argNum + 1) {
            logger.info("Error: detecting " + Integer.toString(args.length) + " arguments. Expecting " + Integer.toString(argNum) + ".");
            logger.info("Usage: ClientGrpc taskID ServerIP ServerPort");
            taskID = "";
            target = "";
            System.exit(0);
        } else {
            taskID = args[0];
            target = args[1] + ":" + args[2];
        }
        logger.info("TaskID is: " + taskID);
        logger.info("Accessing service at: " + target);

        int max_wait = 20;
        // Example code for client
        int idSize = 11;
        // Quick lookup for the plaintext of hashed ids
        HashMap<String, String> data = Utils.genRandomHashSet(idSize);
        HashMap<String, String> hashedIds = new HashMap<>();
        List<String> hashedIdArray;
        String salt;
        List<String> ids = new ArrayList<>(data.keySet());

        // Create a communication channel to the server,  known as a Channel. Channels are thread-safe
        // and reusable. It is common to create channels at the beginning of your application and reuse
        // them until the application shuts down.
        ManagedChannel channel = ManagedChannelBuilder.forTarget(target)
                // Channels are secure by default (via SSL/TLS).
                .usePlaintext()
                .build();
        try {
            ClientGrpc client = new ClientGrpc(channel, taskID);
            // Get salt from Server
            salt = client.getSalt();
            logger.debug("Client get Slat=" + salt);
            // Hash(IDs, salt) into hashed IDs
            hashedIdArray = SecurityUtils.parallelToSHAHexString(ids, salt);
            for (int i = 0; i < ids.size(); i++) {
                hashedIds.put(hashedIdArray.get(i), ids.get(i));
            }
            logger.debug("HashedIDs Size = " + hashedIds.size());
            client.uploadSet(hashedIdArray);
            List<String> intersection;

            while (max_wait > 0) {
                intersection = client.downloadIntersection();
                if (intersection == null) {
                    logger.info("Wait 1000ms");
                    Thread.sleep(1000);
                } else {
                    logger.info("Intersection successful. Intersection's size is " + intersection.size() + ".");
                    break;
                }
                max_wait--;
            }

        } finally {
            // ManagedChannels use resources like threads and TCP connections. To prevent leaking these
            // resources the channel should be shut down when it will no longer be used. If it may be used
            // again leave it running.
            channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }
}


