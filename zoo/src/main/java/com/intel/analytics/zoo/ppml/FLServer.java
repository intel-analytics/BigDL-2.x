package com.intel.analytics.zoo.ppml;

import com.intel.analytics.zoo.grpc.ZooGrpcServer;
import com.intel.analytics.zoo.ppml.ps.ParameterServerServiceImpl;
import com.intel.analytics.zoo.ppml.psi.PSIServiceImpl;
import io.grpc.BindableService;
import io.grpc.ServerBuilder;
import io.grpc.netty.GrpcSslContexts;
import io.grpc.netty.NettyServerBuilder;
import io.netty.handler.ssl.ClientAuth;
import io.netty.handler.ssl.SslContext;
import io.netty.handler.ssl.SslContextBuilder;
import org.apache.commons.cli.Option;

import javax.net.ssl.SSLException;
import java.io.File;

/**
 * FLServer is Analytics Zoo PPML gRPC server used for FL based on ZooGrpcServer
 * It would parse ZooGrpcServer args and add its specific args after it.
 * User could also call main method and parse server type to start gRPC service
 * Supported types: PSI, VFL,
 */
public class FLServer extends ZooGrpcServer {
    String certChainFilePath;
    String privateKeyFilePath;
    String trustCertCollectionFilePath;
    FLServer(String[] args, BindableService service) {
        super(args, service);
        options.addOption(new Option(
                "cc", "certChainFilePath", true, "certChainFilePath"));
        options.addOption(new Option(
                "pk", "privateKeyFilePath", true, "privateKeyFilePath"));
        options.addOption(new Option(
                "tcc", "trustCertCollectionFilePath", true, "trustCertCollectionFilePath"));
    }
    void startWithTls() throws SSLException {
        parseArgs();
        certChainFilePath = cmd.getOptionValue("cc", null);
        privateKeyFilePath = cmd.getOptionValue("pk", null);
        trustCertCollectionFilePath = cmd.getOptionValue("tcc", null);
        NettyServerBuilder serverBuilder = NettyServerBuilder.forPort(port)
                .addService(service);
        if (certChainFilePath != null && privateKeyFilePath != null) {
            serverBuilder.sslContext(getSslContext());
        }
    }

    SslContext getSslContext() throws SSLException {
        SslContextBuilder sslClientContextBuilder = SslContextBuilder.forServer(new File(certChainFilePath),
                new File(privateKeyFilePath));
        if (trustCertCollectionFilePath != null) {
            sslClientContextBuilder.trustManager(new File(trustCertCollectionFilePath));
            sslClientContextBuilder.clientAuth(ClientAuth.REQUIRE);
        }
        return GrpcSslContexts.configure(sslClientContextBuilder).build();
    }
    @Override
    public void build() {
        parseArgs();
        ServerBuilder builder = ServerBuilder.forPort(port);
        if (service != null) {
            builder.addService(service);
        }
        for (String service : services) {
            if (service == "psi") {
                builder.addService(new PSIServiceImpl());
            } else if (service == "ps") {
                builder.addService(new ParameterServerServiceImpl(aggregator));
            } else {
                logger.warn("Type is not supported, skipped. Type: " + service);
            }
        }
        server = builder.maxInboundMessageSize(Integer.MAX_VALUE).build();
    }
}
