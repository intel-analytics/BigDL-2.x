package com.intel.analytics.zoo.ppml;

import com.intel.analytics.zoo.grpc.ZooGrpcServer;
import com.intel.analytics.zoo.ppml.psi.PSIServiceImpl;
import io.grpc.BindableService;

import io.grpc.netty.shaded.io.grpc.netty.GrpcSslContexts;
import io.grpc.netty.shaded.io.grpc.netty.NettyServerBuilder;
import io.grpc.netty.shaded.io.netty.handler.ssl.ClientAuth;
import io.grpc.netty.shaded.io.netty.handler.ssl.SslContext;
import io.grpc.netty.shaded.io.netty.handler.ssl.SslContextBuilder;


import org.apache.commons.cli.*;
import org.apache.hadoop.yarn.webapp.hamlet.Hamlet;
import org.omg.PortableInterceptor.SYSTEM_EXCEPTION;

import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLException;
import java.io.File;
import java.io.IOException;

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

    public static void main(String[] args) throws IOException, InterruptedException {
        Options options = new Options();
        options.addOption(new Option(
                "s", "serverType", true, "Server type."));
        CommandLine cmd = null;
        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();
        try {
            cmd = parser.parse(options, args);
        } catch (ParseException e) {
            System.out.println(e.getMessage());
            formatter.printHelp("utility-name", options);
            System.exit(1);
        }
        assert cmd != null;
        String serverType = cmd.getOptionValue("s", null);
        if (serverType == null) {
            System.out.println("No serverType chosen, exiting..");
            System.exit(1);
        }
        ZooGrpcServer server;
        if (serverType == "vflpsi") {
            server = new ZooGrpcServer(args, new PSIServiceImpl());
        } else if (serverType == "vflps") {
            //TODO: add other supported types
        } else {
            System.out.println("Type not supported: " + serverType);
            System.exit(0);
        }

        server.start();
        server.blockUntilShutdown();
    }
}
