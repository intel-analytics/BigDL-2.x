package com.intel.analytics.zoo.utils;


import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class ConfigParser {
    static ObjectMapper objectMapper;
    static {
        objectMapper = new ObjectMapper(new YAMLFactory());
        objectMapper.configure(MapperFeature.ACCEPT_CASE_INSENSITIVE_PROPERTIES, true);
        objectMapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        objectMapper.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
    }

    public static <T> T loadConfigFromPath(String configPath, Class<T> valueType)
            throws IOException {
        return objectMapper.readValue(new java.io.File(configPath), valueType);
    }
    public static <T> T loadConfigFromString(String configString, Class<T> valueType)
            throws JsonProcessingException {

        return objectMapper.readValue(configString, valueType);
    }
}