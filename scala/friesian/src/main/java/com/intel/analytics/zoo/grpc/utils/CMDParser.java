package com.intel.analytics.zoo.grpc.utils;

import org.apache.commons.cli.*;
import org.apache.log4j.Logger;

import java.util.HashMap;
import java.util.Map;

public class CMDParser {
    private static final Logger logger = Logger.getLogger(CMDParser.class.getName());
    private Options options;
    private Map<String, String> defaultValueMap;
    private CommandLine cmd;

    public CMDParser() {
        options = new Options();
        defaultValueMap = new HashMap<>();
    }

    public void addOption(String optName, String description, String defaultValue) {
        options.addOption(new Option(optName, true, description));
        defaultValueMap.put(optName, defaultValue);
    }

    public void parseOptions(String[] args) {
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
    }

    public String getOptionValue(String optName) {
        if (cmd == null) {
            logger.error("You should call parseOptions before calling getOptionValues");
            return null;
        } else {
            if (defaultValueMap.containsKey(optName)) {
                String defaultValue = defaultValueMap.getOrDefault(optName, "");
                return cmd.getOptionValue(optName, defaultValue);
            } else {
                logger.error("Option " + optName + " does not exist.");
                return null;
            }
        }
    }

    public int getIntOptionValue(String optName) {
        String value = this.getOptionValue(optName);
        if (value != null) {
            return Integer.parseInt(value);
        } else {
            return -1;
        }
    }
}
