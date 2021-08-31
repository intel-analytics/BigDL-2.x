package com.intel.analytics.zoo.grpc;

import com.intel.analytics.zoo.utils.ConfigParser;
import org.apache.commons.cli.*;

import java.io.IOException;

public abstract class AbstractZooGrpc {
    protected String[] args;
    protected Options options;
    protected String configPath;
    protected CommandLine cmd;
    protected String serviceList = "";
    protected String[] services;
    protected <T> T getCmd(Class<T> valueType) throws IOException {
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
        configPath = cmd.getOptionValue("config", null);
        if (configPath != null) {
            // config YAML passed, use config YAML first, command-line could overwrite
            assert valueType != null;
            return ConfigParser.loadConfigFromPath(configPath, valueType);
        }
        else {
            return null;
        }

    }
}
