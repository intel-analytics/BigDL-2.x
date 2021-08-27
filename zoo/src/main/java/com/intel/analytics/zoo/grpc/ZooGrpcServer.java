/*
 * Copyright 2018 Analytics Zoo Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


package com.intel.analytics.zoo.grpc;


import com.intel.analytics.zoo.ppml.ps.Aggregator;
import io.grpc.BindableService;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import org.apache.commons.cli.*;
import org.apache.log4j.Logger;

import java.io.IOException;
import java.util.concurrent.TimeUnit;


/**
 * All Analytics Zoo gRPC server are based on ZooGrpcServer
 * This class could also be directly used for start a single service
 */
public class ZooGrpcServer {
    protected static final Logger logger = Logger.getLogger(ZooGrpcServer.class.getName());
    protected int port;
    protected Server server;
    protected String[] args;
    protected Options options;
    protected String configPath;
    protected BindableService service;
    protected String[] services;
    protected CommandLine cmd;

    /**
     * One Server could support multiple servives.
     * Also support a single service constructor
     * @param service
     */
    public ZooGrpcServer(BindableService service) {
        this(null, service);
    }
    public ZooGrpcServer(String[] args, BindableService service) {
        options = new Options();
        Option portArg = new Option(
                "p", "port", true, "The port to listen.");
        options.addOption(portArg);
        Option configPathArg = new Option(
                "c", "config", true, "The path to config YAML file");
        options.addOption(configPathArg);
        this.service = service;
        this.args = args;

    }
    public ZooGrpcServer(String[] args) {
        this(args, null);
    }
    protected void parseArgs() {
        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();
        cmd = null;

        try {
            cmd = parser.parse(options, args);
        } catch (ParseException e) {
            System.out.println(e.getMessage());
            formatter.printHelp("utility-name", options);
            System.exit(1);
        }
        assert cmd != null;
        port = Integer.parseInt(cmd.getOptionValue("port", "8082"));
        configPath = cmd.getOptionValue("config", "config.yaml");
        services = cmd.getOptionValue("s", "").split(",");

    }

    /** Entrypoint of ZooGrpcServer */
    public void build() {
        parseArgs();
        ServerBuilder builder = ServerBuilder.forPort(port);
        if (service != null) {
            builder.addService(service);
        }
        server = builder.maxInboundMessageSize(Integer.MAX_VALUE).build();
    }

    /** Start serving requests. */
    public void start() throws IOException {
        /* The port on which the server should run */
        server.start();
        logger.info("Server started, listening on " + port);
        Runtime.getRuntime().addShutdownHook(new Thread() {
            @Override
            public void run() {
                // Use stderr here since the logger may have been reset by its JVM shutdown hook.
                System.err.println("*** shutting down gRPC server since JVM is shutting down");
                try {
                    ZooGrpcServer.this.stop();
                } catch (InterruptedException e) {
                    e.printStackTrace(System.err);
                }
                System.err.println("*** server shut down");
            }
        });
    }
    public void stop() throws InterruptedException {
        if (server != null) {
            server.shutdown().awaitTermination(30, TimeUnit.SECONDS);
        }
    }
    /**
     * Await termination on the main thread since the grpc library uses daemon threads.
     */
    public void blockUntilShutdown() throws InterruptedException {
        if (server != null) {
            server.awaitTermination();
        }
    }
}
