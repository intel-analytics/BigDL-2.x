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

package com.intel.analytics.zoo.ppml.psi.example;

import com.intel.analytics.zoo.ppml.psi.ClientGrpc;
import com.intel.analytics.zoo.ppml.psi.SecurityUtils;
import com.intel.analytics.zoo.ppml.psi.generated.PSIServiceGrpc;
import io.grpc.Channel;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class BenchmarkClient extends ClientGrpc {
    private static final Logger logger = LoggerFactory.getLogger(BenchmarkClient.class);

    private static PSIServiceGrpc.PSIServiceBlockingStub blockingStub;

    /**
     * Construct client for accessing server using the existing channel.
     */
    public BenchmarkClient(Channel channel) {
        // 'channel' here is a Channel,  not a ManagedChannel,  so it is not this code's responsibility to
        // shut it down.
        // Passing Channels to code makes code easier to test and makes it easier to reuse Channels.
        super(channel);
	blockingStub = PSIServiceGrpc.newBlockingStub(channel);
    }

    public BenchmarkClient(Channel channel, String taskID) {
	super(channel, taskID);
        blockingStub = PSIServiceGrpc.newBlockingStub(channel);
    }

    public static void main(String[] args) throws Exception {
	String taskID;
	String target;
	int idSize;
	int startNum;
	// Number of arguments to be passed.
        int argNum = 5;
        if (args.length == 0) {
            logger.info("No argument passed, using default parameters.");
            taskID = "taskID";
            target = "localhost:50051";
            idSize = 10000;
            startNum = 0;
        } else if (args.length < argNum || args.length > argNum + 1) {
            logger.info("Error: detecting " + Integer.toString(args.length) + " arguments. Expecting " + Integer.toString(argNum) + ".");
            logger.info("Usage: BenchmarkClient taskID ServerIP ServerPort");
            taskID = "";
            target = "";
            idSize = 0;
            startNum = 0;
            System.exit(0);
        } else {
            taskID = args[0];
            target = args[1] + ":" + args[2];
            idSize = Integer.parseInt(args[3]);
            startNum = Integer.parseInt(args[4]);
        }
        logger.info("TaskID is: " + taskID);
        logger.info("Accessing service at: " + target);

        // Example code for client
        // Quick lookup for the plaintext of hashed ids
        List<String> ids = new ArrayList<String>(idSize);
        long stproduce = System.currentTimeMillis();
        for (int i = startNum; i < idSize; i++) {
            ids.add(i-startNum, String.valueOf(i));
        }
        long etproduce = System.currentTimeMillis();
        logger.info("### Time of producing data: " + (etproduce - stproduce) + " ms ###");
        HashMap<String, String> hashedIds = new HashMap<>();
        List<String> hashedIdArray;
        String salt;

        // Create a communication channel to the server,  known as a Channel. Channels are thread-safe
        // and reusable. It is common to create channels at the beginning of your application and reuse
        // them until the application shuts down.
        ManagedChannel channel = ManagedChannelBuilder.forTarget(target)
                // Channels are secure by default (via SSL/TLS).
                //extend message size of server to 200M to avoid size conflict
		.maxInboundMessageSize(Integer.MAX_VALUE)
		.usePlaintext()
                .build();
        try {
            BenchmarkClient client = new BenchmarkClient(channel, taskID);
            // Get salt from Server
            salt = client.getSalt();
            logger.info("Client get Slat=" + salt);
            // Hash(IDs, salt) into hashed IDs
            long shash = System.currentTimeMillis();
            hashedIdArray = SecurityUtils.parallelToSHAHexString(ids, salt);
            for (int i = 0; i < ids.size(); i++) {
                logger.debug(hashedIdArray.get(i));
                hashedIds.put(hashedIdArray.get(i), ids.get(i));
            }
            long ehash = System.currentTimeMillis();
            logger.info("### Time of hash data: " + (ehash - shash) + " ms ###");
            logger.info("HashedIDs Size = " + hashedIdArray.size());
            long supload = System.currentTimeMillis();
            client.uploadSet(hashedIdArray);
            long eupload = System.currentTimeMillis();
            logger.info("### Time of upload data: " + (eupload - supload) + " ms ###");
            logger.info("upload hashed id successfully");
            List<String> intersection;
            
            long sdownload = System.currentTimeMillis();
            intersection = client.downloadIntersection();
            long edownload = System.currentTimeMillis();
            logger.info("### Time of download data: " + (edownload - sdownload) + " ms ###");
            logger.info("Intersection successful. Total id(s) in intersection is " + intersection.size());

        } finally {
            // ManagedChannels use resources like threads and TCP connections. To prevent leaking these
            // resources the channel should be shut down when it will no longer be used. If it may be used
            // again leave it running.
            channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }
}


