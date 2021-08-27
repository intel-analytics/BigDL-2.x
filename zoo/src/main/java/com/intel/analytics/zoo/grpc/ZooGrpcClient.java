package com.intel.analytics.zoo.grpc;


import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import org.apache.commons.cli.*;
import org.apache.log4j.Logger;

import java.util.UUID;

public class ZooGrpcClient {
    protected static final Logger logger = Logger.getLogger(ZooGrpcServer.class.getName());
    protected String[] args;
    protected Options options;
    protected String target;
    protected String[] services;
    protected String configPath;
    protected final String clientUUID;
    protected CommandLine cmd;
    protected ManagedChannel channel;
    public ZooGrpcClient(String[] args) {
        clientUUID = UUID.randomUUID().toString();
        options = new Options();
        Option portArg = new Option(
                "t", "target", true, "URL.");
        options.addOption(portArg);
        Option configPathArg = new Option(
                "c", "config", true, "The path to config YAML file");
        options.addOption(new Option(
                "s", "service", true, "service to use"));
        options.addOption(configPathArg);
        this.args = args;


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
        target = cmd.getOptionValue("t", "locaohost:8980");
        configPath = cmd.getOptionValue("config", "config.yaml");
        services = cmd.getOptionValue("s", "").split(",");
    }
    public void build() {
        parseArgs();
        channel = ManagedChannelBuilder.forTarget(target)
                // Channels are secure by default (via SSL/TLS).
                .usePlaintext()
                .build();
    }

}
