package com.intel.analytics.zoo.grpc.recommend;

import com.google.protobuf.Empty;
import com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto;
import com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.RecommendIDProbs;
import com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.RecommendRequest;
import com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.ServerMessage;
import com.intel.analytics.zoo.grpc.generated.recommend.RecommendServiceGrpc;
import com.intel.analytics.zoo.grpc.utils.CMDParser;
import io.grpc.Channel;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import utils.Utils;

import java.util.concurrent.TimeUnit;

public class RecommendClient {
    private static final Logger logger = Logger.getLogger(RecommendClient.class.getName());
    private final RecommendServiceGrpc.RecommendServiceBlockingStub blockingStub;

    public RecommendClient(Channel channel) {
        blockingStub = RecommendServiceGrpc.newBlockingStub(channel);
    }

    public RecommendIDProbs getUserRecommends(int[] userIds, int candidateNum, int recommendNum) {
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
        } catch (StatusRuntimeException e) {
            logger.warn("RPC failed: " + e.getStatus().toString());
        }
        return recommendIDProbs;
    }

    public void getMetrics() {
        Empty request = Empty.newBuilder().build();

        ServerMessage msg;
        try {
            msg = blockingStub.getMetrics(request);
        } catch (StatusRuntimeException e) {
            logger.warn("RPC failed: " + e.getStatus().toString());
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
        }
    }

    public void getClientMetrics() {
        Empty request = Empty.newBuilder().build();
        ServerMessage msg;
        try {
            msg = blockingStub.getClientMetrics(request);
        } catch (StatusRuntimeException e) {
            logger.warn("RPC failed: " + e.getStatus().toString());
            return;
        }
        logger.info("Got client metrics: " + msg.getStr());
    }

    /** Issues several different requests and then exits. */
    public static void main(String[] args) throws InterruptedException {
        Logger.getLogger("org").setLevel(Level.ERROR);

        CMDParser cmdParser = new CMDParser();
        cmdParser.addOption("target", "The server to connect to.", "localhost:8980");
        cmdParser.addOption("dataDir", "The data file.", "wnd_user.parquet");
        cmdParser.addOption("k", "The candidate num, default: 50.", "50");

        cmdParser.parseOptions(args);
        String target = cmdParser.getOptionValue("target");
        String dir = cmdParser.getOptionValue("dataDir");
        int candidateK = cmdParser.getIntOptionValue("k");

        ManagedChannel channel = ManagedChannelBuilder.forTarget(target).usePlaintext().build();
        RecommendClient client = new RecommendClient(channel);

        int dataNum = 1000;
        int[] userList = Utils.loadUserData(dir, "enaging_user_id", dataNum);

        long start = System.nanoTime();
        for (int userId: userList) {
            RecommendProto.RecommendIDProbs result = client.getUserRecommends(new int[]{userId},
                    candidateK, 10);
        }
        long end = System.nanoTime();
        long time = (end - start)/dataNum;
        System.out.println("Total user number: " + dataNum);
        System.out.println("Average search time: " + time);
        client.getMetrics();
        client.resetMetrics();
        client.getClientMetrics();
        channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
    }
}
