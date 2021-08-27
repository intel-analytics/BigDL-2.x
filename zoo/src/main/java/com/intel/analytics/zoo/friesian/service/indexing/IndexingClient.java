package com.intel.analytics.zoo.friesian.service.indexing;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import org.apache.log4j.Logger;

import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import com.google.protobuf.Empty;
import com.intel.analytics.zoo.friesian.generated.indexing.Candidates;
import com.intel.analytics.zoo.friesian.generated.indexing.IndexingGrpc;
import com.intel.analytics.zoo.friesian.generated.indexing.Query;
import com.intel.analytics.zoo.friesian.generated.indexing.ServerMessage;
import com.intel.analytics.zoo.serving.http.JacksonJsonSerializer;
import io.grpc.Channel;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

import io.grpc.StatusRuntimeException;
import org.apache.commons.cli.*;
import com.google.common.annotations.VisibleForTesting;
import com.google.protobuf.Message;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import com.intel.analytics.zoo.friesian.utils.TimerMetrics;
import com.intel.analytics.zoo.friesian.utils.TimerMetrics$;
import com.intel.analytics.zoo.friesian.utils.feature.FeatureUtils;


public class IndexingClient {
    private static final Logger logger = Logger.getLogger(IndexingClient.class.getName());

    private final IndexingGrpc.IndexingBlockingStub blockingStub;
    private final IndexingGrpc.IndexingStub asyncStub;
    private TestHelper testHelper;

    private static MetricRegistry metrics = new MetricRegistry();
    private static Timer preProcessTimer = metrics.timer("indexing.preProcess");
    private static Timer searchTimer = metrics.timer("indexing.search");

    /** Construct client for accessing Indexing server using the existing channel. */
    public IndexingClient(Channel channel) {
        blockingStub = IndexingGrpc.newBlockingStub(channel);
        asyncStub = IndexingGrpc.newStub(channel);
    }

    public void searchTop10(int userId) {
        search(userId, 10);
    }

    public void search(int userId, int k) {
//        info("*** Get input: " + jsonStr);

        Query request = Query.newBuilder().setUserID(userId).setK(k).build();

        Candidates candidates;
        try {
            candidates = blockingStub.searchCandidates(request);
            printCandidates(userId, candidates);
            if (testHelper != null) {
                testHelper.onMessage(candidates);
            }
        } catch (StatusRuntimeException e) {
            logger.warn("RPC failed: " + e.getStatus().toString());
            if (testHelper != null) {
                testHelper.onRpcError(e);
            }
            return;
        }
//        info("Got predResult: " + predResult.getPredictStr());
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

    public void getClientMetrics() {
        JacksonJsonSerializer jacksonJsonSerializer = new JacksonJsonSerializer();
        Set<String> keys = metrics.getTimers().keySet();
        List<TimerMetrics> timerMetrics = keys.stream()
                .map(key ->
                        TimerMetrics$.MODULE$.apply(key, metrics.getTimers().get(key)))
                .collect(Collectors.toList());
        String jsonStr = jacksonJsonSerializer.serialize(timerMetrics);
        logger.info("Client metrics: " + jsonStr);
    }

    public void resetClientMetrics() {
        metrics = new MetricRegistry();
        preProcessTimer = metrics.timer("indexing.preProcess");
        searchTimer = metrics.timer("indexing.search");
    }

    public void printCandidates(int userId, Candidates candidates) {
        List<Integer> candidateList = candidates.getCandidateList();
        System.out.printf("UserID %d: Candidates: ", userId);
        StringBuilder sb = new StringBuilder();
        for (Integer candidate: candidateList) {
            sb.append(candidate).append('\t');
        }
        sb.append('\n');
        System.out.print(sb.toString());
    }

    /** Issues several different requests and then exits. */
    public static void main(String[] args) throws InterruptedException{
        org.apache.log4j.Logger.getLogger("org.apache").setLevel(org.apache.log4j.Level.ERROR);
        Options options = new Options();
        Option target = new Option("t", "target", true, "The server to connect to.");
        options.addOption(target);
        Option dataDir = new Option("dataDir", true, "The data file.");
        options.addOption(dataDir);
        Option threadNum = new Option("c", "clientNum", true, "Concurrent client number.");
        options.addOption(threadNum);
        Option testNum = new Option("testNum", true, "Test case run number.");
        options.addOption(testNum);
        Option preProcessNum = new Option("p", "preProcessNum", true, "PreProcess mock function run number.");
        options.addOption(preProcessNum);
        Option sleepTime = new Option("s", "sleepTime", true, "Sleep time before search.");
        options.addOption(sleepTime);

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
        String targetURL = cmd.getOptionValue("target", "localhost:8084");
        String dir = cmd.getOptionValue("dataDir", "/home/yina/Documents/data/recsys/preprocess_output/vec_feature_user.parquet");
        int testRepeatNum = Integer.parseInt(cmd.getOptionValue("testNum", "1"));
        int concurrentNum = Integer.parseInt(cmd.getOptionValue("clientNum", "1"));
        int preProcess = Integer.parseInt(cmd.getOptionValue("preProcessNum", "1"));
        long sleep = Integer.parseInt(cmd.getOptionValue("sleepTime", "10"));

        ManagedChannel channel = ManagedChannelBuilder.forTarget(targetURL).usePlaintext().build();
        int[] userList = FeatureUtils.loadUserData(dir, "enaging_user_id");
        IndexingClient client = new IndexingClient(channel);
        try {
            for (int r = 0; r < testRepeatNum; r ++) {
                logger.info("Test round: " + (r + 1));
                ArrayList<IndexingThread> tList = new ArrayList<>();
                for (int i = 0; i < concurrentNum; i ++) {
                    IndexingThread t = new IndexingThread(
                            channel, userList, sleep, preProcess, preProcessTimer, searchTimer);
                    tList.add(t);
                    t.start();
                }
                for (IndexingThread t: tList) {
                    t.join();
                }
                client.getClientMetrics();
                client.resetClientMetrics();
                client.getMetrics();
                client.resetMetrics();
                Thread.sleep(10000);
            }
        } finally {
            channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }

//    private void info(String msg, Object... params) {
//        logger.log(Level.INFO, msg, params);
//    }

//    private void warning(String msg, Object... params) {
//        logger.log(Level.WARNING, msg, params);
//    }
//
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
    void setTestHelper(TestHelper testHelper) {
        this.testHelper = testHelper;
    }
}

class IndexingThread extends Thread {
    private ManagedChannel channel;
    private int[] userList;
    private int dataNum;
    private long sleepTime;
    private int preProcessNum;
    private Timer preProcessTimer;
    private Timer searchTimer;

    IndexingThread(ManagedChannel channel, int[] userList, long sleepTime, int preProcessNum,
                   Timer preProcessTimer, Timer searchTimer) {
        this.channel = channel;
        this.userList = userList;
        this.dataNum = userList.length;
        this.sleepTime = sleepTime;
        this.preProcessNum = preProcessNum;
        this.preProcessTimer = preProcessTimer;
        this.searchTimer = searchTimer;
    }

    // mock preprocess function, takes around 0.5ms
    private void mockPreProcess() {
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < 1000; i++) list.add((int)((Math.random() * 88)));
        Collections.shuffle(list);
        Collections.sort(list);
    }

    @Override
    public void run() {
        IndexingClient client = new IndexingClient(channel);
        long start = System.nanoTime();
        for (int userId: userList){
            try{
                Timer.Context preProcessContext = preProcessTimer.time();
                for (int i = 0; i < preProcessNum; i ++) {
                    mockPreProcess();
                }
                preProcessContext.stop();
                Timer.Context searchContext = searchTimer.time();
                client.search(userId, 50);
                searchContext.stop();
                sleep(sleepTime);
            } catch(InterruptedException e) {
                System.out.println("Got interrupted!");
            }
        }
        long end = System.nanoTime();
        long time = (end - start)/dataNum;
        System.out.println("Total user number: " + dataNum);
        System.out.println("Average search time: " + time);
    }
}