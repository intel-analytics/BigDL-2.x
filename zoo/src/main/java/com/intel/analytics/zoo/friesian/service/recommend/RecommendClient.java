package com.intel.analytics.zoo.friesian.service.recommend;

import com.google.common.annotations.VisibleForTesting;
import com.google.protobuf.Empty;
import com.google.protobuf.Message;
import com.intel.analytics.zoo.friesian.generated.recommend.RecommendIDProbs;
import com.intel.analytics.zoo.friesian.generated.recommend.RecommendRequest;
import com.intel.analytics.zoo.friesian.generated.recommend.RecommendServiceGrpc;
import com.intel.analytics.zoo.friesian.generated.recommend.ServerMessage;
import io.grpc.Channel;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import org.apache.commons.cli.*;
import com.intel.analytics.zoo.friesian.utils.feature.FeatureUtils;

import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

import org.apache.log4j.Logger;

public class RecommendClient {
    private static final Logger logger = Logger.getLogger(RecommendClient.class.getName());

    private final RecommendServiceGrpc.RecommendServiceBlockingStub blockingStub;
    private RecommendClient.TestHelper testHelper;

    /** Construct client for accessing RecommendService server using the existing channel. */
    public RecommendClient(Channel channel) {
        blockingStub = RecommendServiceGrpc.newBlockingStub(channel);
    }

    public RecommendIDProbs getUserRecommends(int[] userIds, int candidateNum, int recommendNum) {
//        info("*** Get input: " + jsonStr);

        RecommendRequest.Builder request = RecommendRequest.newBuilder();
        for (int id: userIds) {
            request.addID(id);
        }
        request.setCandidateNum(candidateNum);
        request.setRecommendNum(recommendNum);

        RecommendIDProbs recommendIDProbs = null;
        try {
            recommendIDProbs = blockingStub.getRecommendIDs(request.build());
//            logger.info(recommendIDProbs.getIDProbListList());
            if (testHelper != null) {
                testHelper.onMessage(recommendIDProbs);
            }

        } catch (StatusRuntimeException e) {
            logger.warn("RPC failed: " + e.getStatus().toString());
            if (testHelper != null) {
                testHelper.onRpcError(e);
            }
        }
//        info("Got predResult: " + predResult.getPredictStr());
        return recommendIDProbs;
    }

    public void getMetrics() {
        Empty request = Empty.newBuilder().build();

        ServerMessage msg;
        try {
            msg = blockingStub.getMetrics(request);
        } catch (StatusRuntimeException e) {
            logger.warn("RPC failed: " + e.getStatus().toString());
            if (testHelper != null) {
                testHelper.onRpcError(e);
            }
            return;
        }

        logger.info("Got metrics: " + msg.getStr());
    }

    public void resetMetrics() {
        Empty request = Empty.newBuilder().build();
        try {
            blockingStub.resetMetrics(request);
        } catch (StatusRuntimeException e) {
            logger.warn("RPC failed: " + e.getStatus().toString());
            if (testHelper != null) {
                testHelper.onRpcError(e);
            }
        }
    }

    private boolean checkEqual(Object[] o1, Object[] o2) {
        return true;
    }

    /** Issues several different requests and then exits. */
    public static void main(String[] args) throws InterruptedException{
        Options options = new Options();
        Option target = new Option("t", "target", true, "The server to connect to.");
        options.addOption(target);
        Option dataDir = new Option("dataDir", true, "The data file.");
        options.addOption(dataDir);
        Option threadNum = new Option("c", "clientNum", true, "Concurrent client number.");
        options.addOption(threadNum);
        Option testNum = new Option("testNum", true, "Test case run number.");
        options.addOption(testNum);

        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();
        CommandLine cmd = null;

        try {
            cmd = parser.parse(options, args);
        } catch (ParseException e) {
            System.out.println(e.getMessage());
            formatter.printHelp("utility-name", options);

            System.exit(1);
        }
        assert cmd != null;
        String targetURL = cmd.getOptionValue("target", "localhost:8980");
        String dir = cmd.getOptionValue("dataDir", "/home/yina/Documents/data/recsys/preprocess_output/wnd_user.parquet");
//        String dir = cmd.getOptionValue("dataDir", "/home/yina/Documents/data/dien/dien_user.parquet");
        int concurrentNum = Integer.parseInt(cmd.getOptionValue("clientNum", "1"));
        int testRepeatNum = Integer.parseInt(cmd.getOptionValue("testNum", "1"));

        ManagedChannel channel = ManagedChannelBuilder.forTarget(targetURL).usePlaintext().build();
        int[] userList = FeatureUtils.loadUserData(dir, "enaging_user_id");
        RecommendClient client = new RecommendClient(channel);
        try {
            for (int r = 0; r < testRepeatNum; r ++) {
                logger.info("Test round: " + (r + 1));
                ArrayList<RecommendThread> tList = new ArrayList<>();
                for (int i = 0; i < concurrentNum; i ++) {
                    RecommendThread t = new RecommendThread(channel, userList);
                    tList.add(t);
                    t.start();
                }
                for (RecommendThread t: tList) {
                    t.join();
                }
                client.getMetrics();
                client.resetMetrics();
                Thread.sleep(10000);
            }
        } finally {
            channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    /**
     * Only used for helping unit test.
     */
    @VisibleForTesting
    interface TestHelper {
        /**
         * Used for verify/inspect message received from server.
         */
        void onMessage(Message message);

        /**
         * Used for verify/inspect error received from server.
         */
        void onRpcError(Throwable exception);
    }

    @VisibleForTesting
    void setTestHelper(RecommendClient.TestHelper testHelper) {
        this.testHelper = testHelper;
    }
}

class RecommendThread extends Thread {
    private ManagedChannel channel;
    private int[] userList;
    private int dataNum;

    RecommendThread(ManagedChannel channel, int[] userList) {
        this.channel = channel;
        this.userList = userList;
        this.dataNum = userList.length;
    }

    @Override
    public void run() {
        RecommendClient client = new RecommendClient(channel);
        long start = System.nanoTime();
        for (int userId: userList){
            RecommendIDProbs result = client.getUserRecommends(new int[]{userId}, 50, 10);
        }
        long end = System.nanoTime();
        long time = (end - start)/dataNum;
        System.out.println("Total user number: " + dataNum);
        System.out.println("Average search time: " + time);
    }
}
