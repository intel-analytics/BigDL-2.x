package com.intel.analytics.zoo.grpc.recommend;

import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import com.google.protobuf.Empty;
import com.intel.analytics.bigdl.dllib.grpc.ZooGrpcServer;
import com.intel.analytics.bigdl.dllib.utils.Table;
import com.intel.analytics.bigdl.dllib.utils.grpc.JacksonJsonSerializer;
import com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceGrpc;
import com.intel.analytics.zoo.grpc.generated.feature.FeatureProto.Features;
import com.intel.analytics.zoo.grpc.generated.feature.FeatureProto.IDs;
import com.intel.analytics.zoo.grpc.generated.feature.FeatureServiceGrpc;
import com.intel.analytics.zoo.grpc.generated.indexing.IndexingGrpc;
import com.intel.analytics.zoo.grpc.generated.indexing.IndexingProto.Candidates;
import com.intel.analytics.zoo.grpc.generated.indexing.IndexingProto.Query;
import com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto;
import com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.IDProbs;
import com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.RecommendIDProbs;
import com.intel.analytics.zoo.grpc.generated.recommend.RecommendProto.RecommendRequest;
import com.intel.analytics.zoo.grpc.generated.recommend.RecommendServiceGrpc;
import io.grpc.*;
import io.grpc.stub.StreamObserver;
import io.prometheus.client.exporter.HTTPServer;
import me.dinowernli.grpc.prometheus.Configuration;
import me.dinowernli.grpc.prometheus.MonitoringServerInterceptor;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import scala.Tuple2;
import utils.TimerMetrics;
import utils.TimerMetrics$;
import utils.Utils;
import utils.gRPCHelper;
import utils.recommend.RecommendUtils;

import java.io.IOException;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class RecommendServer extends ZooGrpcServer {
    private static final Logger logger = Logger.getLogger(RecommendServer.class.getName());

    public RecommendServer(String[] args) {
        super(args);
        configPath = "config_recommend.yaml";
        Logger.getLogger("org").setLevel(Level.ERROR);
    }

    @Override
    public void parseConfig() throws IOException {
        Utils.helper_$eq(getConfigFromYaml(gRPCHelper.class, configPath));
        Utils.helper().parseConfigStrings();
        if (Utils.helper() != null) {
            port = Utils.helper().getServicePort();
        }

        if (Utils.runMonitor()) {
            logger.info("Starting monitoringInterceptor....");
            MonitoringServerInterceptor monitoringInterceptor =
                    MonitoringServerInterceptor.create(Configuration.allMetrics()
                            .withLatencyBuckets(Utils.getPromBuckets()));
            serverDefinitionServices.add(ServerInterceptors
                    .intercept(new RecommendService().bindService(), monitoringInterceptor));
        } else {
            serverServices.add(new RecommendService());
        }
    }

    /**
     * Main method.  This comment makes the linter happy.
     */
    public static void main(String[] args) throws Exception {
        RecommendServer recommendServer = new RecommendServer(args);
        recommendServer.build();
        if (Utils.runMonitor()) {
            new HTTPServer.Builder()
                    .withPort(Utils.helper().monitorPort()).build();
        }
        recommendServer.start();
        recommendServer.blockUntilShutdown();
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

        RecommendService() {
            ManagedChannel vectorSearchChannel =
                    ManagedChannelBuilder.forTarget(Utils.helper().getVectorSearchURL())
                            .usePlaintext().build();
            ManagedChannel featureServiceChannel =
                    ManagedChannelBuilder.forTarget(Utils.helper().getFeatureServiceURL())
                            .usePlaintext().build();
            ManagedChannel rankingServiceChannel =
                    ManagedChannelBuilder.forTarget(Utils.helper().getRankingServiceURL())
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
                               StreamObserver<RecommendProto.ServerMessage> responseObserver) {
            responseObserver.onNext(getMetrics());
            responseObserver.onCompleted();
        }

        private RecommendProto.ServerMessage getMetrics() {
            JacksonJsonSerializer jacksonJsonSerializer = new JacksonJsonSerializer();
            Set<String> keys = metrics.getTimers().keySet();
            List<TimerMetrics> timerMetrics = keys.stream()
                    .map(key ->
                            TimerMetrics$.MODULE$.apply(key, metrics.getTimers().get(key)))
                    .collect(Collectors.toList());
            String jsonStr = jacksonJsonSerializer.serialize(timerMetrics);
            return RecommendProto.ServerMessage.newBuilder().setStr(jsonStr).build();
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

        @Override
        public void getClientMetrics(Empty request,
                                     StreamObserver<RecommendProto.ServerMessage> responseObserver) {
            StringBuilder sb = new StringBuilder();
            String vecMetrics = vectorSearchStub.getMetrics(request).getStr();
            vectorSearchStub.resetMetrics(request);
            sb.append("Vector Search Service backend metrics:\n");
            sb.append(vecMetrics).append("\n\n");
            String feaMetrics = featureServiceStub.getMetrics(request).getStr();
            featureServiceStub.resetMetrics(request);
            sb.append("Feature Service backend metrics:\n");
            sb.append(feaMetrics).append("\n\n");
            String infMetrics = rankingStub.getMetrics(request).getStr();
            rankingStub.resetMetrics(request);
            sb.append("Inference Service backend metrics:\n");
            sb.append(infMetrics).append("\n\n");
            responseObserver.onNext(RecommendProto.ServerMessage.newBuilder().setStr(sb.toString()).build());
            responseObserver.onCompleted();
        }
    }
}
