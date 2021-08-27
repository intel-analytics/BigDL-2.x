package com.intel.analytics.zoo.friesian.service.recommend;

import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import com.google.protobuf.Empty;
import com.intel.analytics.bigdl.utils.Table;
import com.intel.analytics.zoo.friesian.generated.recommend.*;
import com.intel.analytics.zoo.serving.http.JacksonJsonSerializer;
import com.intel.analytics.zoo.friesian.generated.azinference.AZInferenceGrpc;
import com.intel.analytics.zoo.friesian.generated.feature.FeatureServiceGrpc;
import com.intel.analytics.zoo.friesian.generated.feature.Features;
import com.intel.analytics.zoo.friesian.generated.feature.IDs;
import com.intel.analytics.zoo.friesian.generated.indexing.Candidates;
import com.intel.analytics.zoo.friesian.generated.indexing.IndexingGrpc;
import com.intel.analytics.zoo.friesian.generated.indexing.Query;
import io.grpc.*;
import io.grpc.stub.StreamObserver;
import org.apache.commons.cli.*;
import scala.Tuple2;
import com.intel.analytics.zoo.friesian.utils.ConfigParser;
import com.intel.analytics.zoo.friesian.utils.recommend.RecommendUtils;
import com.intel.analytics.zoo.friesian.utils.TimerMetrics;
import com.intel.analytics.zoo.friesian.utils.TimerMetrics$;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;
import org.apache.log4j.Logger;
import java.util.stream.Collectors;

public class RecommendServer {
    private static final Logger logger = Logger.getLogger(RecommendServer.class.getName());

    private final int port;
    private final Server server;

    /** Create a Recommend server listening on {@code port} */
    public RecommendServer(int port, String configPath) {
        this(ServerBuilder.forPort(port), port, configPath);
    }

    public RecommendServer(ServerBuilder<?> serverBuilder, int port, String configPath) {
        this.port = port;
        server = serverBuilder.addService(new RecommendServer.RecommendService(configPath))
                .build();
    }

    /** Start serving requests. */
    public void start() throws IOException {
        server.start();
        logger.info("Server started, listening on " + port);
        Runtime.getRuntime().addShutdownHook(new Thread() {
            @Override
            public void run() {
                // Use stderr here since the logger may have been reset by its JVM shutdown hook.
                System.err.println("*** shutting down gRPC server since JVM is shutting down");
                try {
                    RecommendServer.this.stop();
                } catch (InterruptedException e) {
                    e.printStackTrace(System.err);
                }
                System.err.println("*** server shut down");
            }
        });
    }

    /** Stop serving requests and shutdown resources. */
    public void stop() throws InterruptedException {
        if (server != null) {
            server.shutdown().awaitTermination(30, TimeUnit.SECONDS);
        }
    }

    /**
     * Await termination on the main thread since the grpc library uses daemon threads.
     */
    private void blockUntilShutdown() throws InterruptedException {
        if (server != null) {
            server.awaitTermination();
        }
    }

    /**
     * Main method.  This comment makes the linter happy.
     */
    public static void main(String[] args) throws Exception {
        Options options = new Options();
        Option portArg = new Option("p", "port", true, "The port to listen.");
        options.addOption(portArg);
        Option configPathArg = new Option("c", "config", true, "The path to config.yaml file");
        options.addOption(configPathArg);
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
        int port = Integer.parseInt(cmd.getOptionValue("port", "8980"));
        String configPath = cmd.getOptionValue("config", "config.yaml");
        RecommendServer server = new RecommendServer(port, configPath);
        server.start();
        server.blockUntilShutdown();
    }

    private static class RecommendService extends RecommendServiceGrpc.RecommendServiceImplBase {
        private MetricRegistry metrics = new MetricRegistry();
        private IndexingGrpc.IndexingBlockingStub vectorSearchStub;
        private FeatureServiceGrpc.FeatureServiceBlockingStub featureServiceStub;
        private AZInferenceGrpc.AZInferenceBlockingStub rankingStub;
        Timer overallTimer = metrics.timer("recommend.overall");
        Timer vectorSearchTimer = metrics.timer("recommend.vector.search");
        Timer itemFeatureTimer = metrics.timer("recommend.feature.item");
        Timer userFeatureTimer = metrics.timer("recommend.feature.user");
        Timer preprocessTimer = metrics.timer("recommend.preprocess");
        Timer rankingInferenceTimer = metrics.timer("recommend.rankingInference");
        Timer topKTimer = metrics.timer("recommend.topK");

        RecommendService(String configPath) {
            ConfigParser parser = new ConfigParser(configPath);
            RecommendUtils.helper_$eq(parser.loadConfig());
            ManagedChannel vectorSearchChannel =
                    ManagedChannelBuilder.forTarget(RecommendUtils.helper().getVectorSearchURL())
                            .usePlaintext().build();
            ManagedChannel featureServiceChannel =
                    ManagedChannelBuilder.forTarget(RecommendUtils.helper().getFeatureServiceURL())
                            .usePlaintext().build();
            ManagedChannel rankingServiceChannel =
                    ManagedChannelBuilder.forTarget(RecommendUtils.helper().getRankingServiceURL())
                            .usePlaintext().build();
            vectorSearchStub = IndexingGrpc.newBlockingStub(vectorSearchChannel);
            featureServiceStub = FeatureServiceGrpc.newBlockingStub(featureServiceChannel);
            rankingStub = AZInferenceGrpc.newBlockingStub(rankingServiceChannel);
        }

        @Override
        public void getRecommendIDs(RecommendRequest request,
                                    StreamObserver<RecommendIDProbs> responseObserver) {
            RecommendIDProbs.Builder resultBuilder = RecommendIDProbs.newBuilder();
            Timer.Context overallContext = overallTimer.time();
            List<Integer> ids = request.getIDList();
            int canK = request.getCandidateNum();
            int k = request.getRecommendNum();
            if (canK < k) {
                responseObserver.onError(Status.FAILED_PRECONDITION.withDescription("CandidateNum" +
                        " should be larger than recommendNum.").asRuntimeException());
                return;
            }
            for (Integer id: ids) {
                Timer.Context vectorSearchContext = vectorSearchTimer.time();
                Query query = Query.newBuilder().setUserID(id).setK(canK).build();
                Candidates candidates;
                try {
                    candidates = vectorSearchStub.searchCandidates(query);
                } catch (StatusRuntimeException e) {
                    e.printStackTrace();
                    logger.warn("Vector search unavailable: "+ e.getMessage());
                    responseObserver.onError(Status.UNAVAILABLE.withDescription("vector search " +
                            "service unavailable: " + e.getMessage()).asRuntimeException());
                    return;
                }
                vectorSearchContext.stop();
                Timer.Context userFeatureContext = userFeatureTimer.time();
                IDs userIds = IDs.newBuilder().addID(id).build();
                Features userFeature;
                try {
                    userFeature = featureServiceStub.getUserFeatures(userIds);
                } catch (StatusRuntimeException e) {
                    e.printStackTrace();
                    logger.warn("FeatureService unavailable: "+ e.getMessage());
                    responseObserver.onError(Status.UNAVAILABLE.withDescription("feature " +
                            "service unavailable: " + e.getMessage()).asRuntimeException());
                    return;
                }
                userFeatureContext.stop();
                Timer.Context itemFeatureContext = itemFeatureTimer.time();
                IDs.Builder itemIDsBuilder = IDs.newBuilder();
                for (Integer itemId: candidates.getCandidateList()) {
                    itemIDsBuilder.addID(itemId);
                }
                IDs itemIDs = itemIDsBuilder.build();
                Features itemFeature;
                try {
                    itemFeature = featureServiceStub.getItemFeatures(itemIDs);
                } catch (StatusRuntimeException e) {
                    e.printStackTrace();
                    logger.warn("FeatureService unavailable: "+ e.getMessage());
                    responseObserver.onError(Status.UNAVAILABLE.withDescription("feature " +
                            "service unavailable: " + e.getMessage()).asRuntimeException());
                    return;
                }
                itemFeatureContext.stop();
                Timer.Context preprocessContext = preprocessTimer.time();
                Tuple2<int[], Table[]> itemInputTuple;
                try {
                     itemInputTuple = RecommendUtils.featuresToRankingInputSet(userFeature,
                             itemFeature, 0);
                } catch (Exception e) {
                    e.printStackTrace();
                    logger.warn("FeaturesToRankingInputSet: "+ e.getMessage());
                    responseObserver.onError(Status.FAILED_PRECONDITION
                            .withDescription(e.getMessage()).asRuntimeException());
                    return;
                }
                int[] itemIDArr = itemInputTuple._1;
                Table[] input = itemInputTuple._2;
                preprocessContext.stop();
                Timer.Context rankingContext = rankingInferenceTimer.time();
                String[] result;
                try {
                    result = RecommendUtils.doPredictParallel(input, rankingStub);
                } catch (StatusRuntimeException e) {
                    e.printStackTrace();
                    logger.warn("Inference service unavailable: "+ e.getMessage());
                    responseObserver.onError(Status.UNAVAILABLE.withDescription("inference " +
                            "service unavailable: " + e.getMessage()).asRuntimeException());
                    return;
                }
                rankingContext.stop();
                Timer.Context topKContext = topKTimer.time();
                Tuple2<int[], float[]> topKIDProbsTuple = RecommendUtils.getTopK(result,
                        itemIDArr, k);
                int[] topKIDs = topKIDProbsTuple._1;
                float[] topKProbs = topKIDProbsTuple._2;
                IDProbs.Builder idProbBuilder = IDProbs.newBuilder();
                for (int i = 0; i < topKIDs.length; i ++) {
                    idProbBuilder.addID(topKIDs[i]);
                    idProbBuilder.addProb(topKProbs[i]);
                }
                topKContext.stop();
                resultBuilder.addIDProbList(idProbBuilder.build());
            }
            overallContext.stop();responseObserver.onNext(resultBuilder.build());
            responseObserver.onCompleted();
        }

        @Override
        public void getMetrics(Empty request,
                               StreamObserver<ServerMessage> responseObserver) {
            responseObserver.onNext(getMetrics());
            responseObserver.onCompleted();
        }

        private ServerMessage getMetrics() {
            JacksonJsonSerializer jacksonJsonSerializer = new JacksonJsonSerializer();
            Set<String> keys = metrics.getTimers().keySet();
            List<TimerMetrics> timerMetrics = keys.stream()
                    .map(key ->
                            TimerMetrics$.MODULE$.apply(key, metrics.getTimers().get(key)))
                    .collect(Collectors.toList());
            String jsonStr = jacksonJsonSerializer.serialize(timerMetrics);
            return ServerMessage.newBuilder().setStr(jsonStr).build();
        }

        @Override
        public void resetMetrics(Empty request, StreamObserver<Empty> responseObserver) {
            metrics = new MetricRegistry();
            overallTimer = metrics.timer("recommend.overall");
            vectorSearchTimer = metrics.timer("recommend.vector.search");
            itemFeatureTimer = metrics.timer("recommend.feature.item");
            userFeatureTimer = metrics.timer("recommend.feature.user");
            preprocessTimer = metrics.timer("recommend.preprocess");
            rankingInferenceTimer = metrics.timer("recommend.rankingInference");
            topKTimer = metrics.timer("recommend.topK");
            responseObserver.onNext(Empty.newBuilder().build());
            responseObserver.onCompleted();
        }
    }
}
