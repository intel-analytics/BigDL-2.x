package com.intel.analytics.zoo.grpc.recommend;

import com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto;
import com.intel.analytics.zoo.grpc.utils.CMDParser;
import io.grpc.Channel;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import utils.Utils;

import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

public class RecommendMultiThreadClient {
    private static final Logger logger = Logger.getLogger(RecommendClient.class.getName());

    /** Issues several different requests and then exits. */
    public static void main(String[] args) throws InterruptedException {
        Logger.getLogger("org").setLevel(Level.ERROR);

        CMDParser cmdParser = new CMDParser();
        cmdParser.addOption("target", "The server to connect to.", "localhost:8980");
        cmdParser.addOption("dataDir", "The data file.", "wnd_user.parquet");
        cmdParser.addOption("k", "The candidate num, default: 50.", "50");
        cmdParser.addOption("clientNum", "Concurrent client number.", "1");
        cmdParser.addOption("testNum", "Test case run number.", "1");

        cmdParser.parseOptions(args);
        String target = cmdParser.getOptionValue("target");
        String dir = cmdParser.getOptionValue("dataDir");
        int candidateK = cmdParser.getIntOptionValue("k");
        int concurrentNum = cmdParser.getIntOptionValue("clientNum");
        int testRepeatNum = cmdParser.getIntOptionValue("testNum");

        ManagedChannel channel = ManagedChannelBuilder.forTarget(target).usePlaintext().build();
        RecommendClient client = new RecommendClient(channel);
        int dataNum = 1000;
        int[] userList = Utils.loadUserData(dir, "enaging_user_id", dataNum);

        for (int r = 0; r < testRepeatNum; r ++) {
            logger.info("Test round: " + (r + 1));
            ArrayList<RecommendThread> tList = new ArrayList<>();
            for (int i = 0; i < concurrentNum; i ++) {
                RecommendThread t = new RecommendThread(userList, candidateK, channel);
                tList.add(t);
                t.start();
            }
            for (RecommendThread t: tList) {
                t.join();
            }
            client.getMetrics();
            client.resetMetrics();
            client.getClientMetrics();
            Thread.sleep(10000);
        }
        channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
    }
}

class RecommendThread extends Thread {
    private int[] userList;
    private int dataNum;
    private int candidateNum;
    private Channel channel;

    RecommendThread(int[] userList, int candidateNum, Channel channel) {
        this.userList = userList;
        this.dataNum = userList.length;
        this.candidateNum = candidateNum;
        this.channel = channel;
    }

    @Override
    public void run() {
        RecommendClient client = new RecommendClient(this.channel);
        long start = System.nanoTime();
        for (int userId: userList){
            RecommendProto.RecommendIDProbs result = client.getUserRecommends(new int[]{userId}, candidateNum, 10);
        }
        long end = System.nanoTime();
        long time = (end - start)/dataNum;
        System.out.println("Total user number: " + dataNum);
        System.out.println("Average search time: " + time);
    }
}

