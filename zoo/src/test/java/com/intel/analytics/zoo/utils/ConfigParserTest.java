package com.intel.analytics.zoo.utils;

import com.fasterxml.jackson.core.JsonProcessingException;
import org.junit.Assert;
import org.junit.Test;

public class ConfigParserTest {
    @Test
    public void testConfigParserFromString() throws JsonProcessingException {
        String testString = String.join("\n",
                "stringProp: abc",
                "intProp: 123",
                "boolProp: true");
        TestHelper testHelper = ConfigParser.loadConfigFromString(testString, TestHelper.class);
        Assert.assertEquals(testHelper.intProp, 123);
        Assert.assertEquals(testHelper.boolProp, true);
        Assert.assertEquals(testHelper.stringProp, "abc");
    }
    @Test
    public void testConfigParserFromStringWithEmptyBool() throws JsonProcessingException {
        String testString = String.join("\n",
                "stringProp: abc",
                "intProp: 123");
        TestHelper testHelper = ConfigParser.loadConfigFromString(testString, TestHelper.class);
        Assert.assertEquals(testHelper.intProp, 123);
        Assert.assertEquals(testHelper.boolProp, false);
        Assert.assertEquals(testHelper.stringProp, "abc");
    }
    @Test
    public void testConfigParserFromStringWithEmptyString() throws JsonProcessingException {
        String testString = String.join("\n",
                "boolProp: true",
                "intProp: 123");
        TestHelper testHelper = ConfigParser.loadConfigFromString(testString, TestHelper.class);
        Assert.assertEquals(testHelper.intProp, 123);
        Assert.assertEquals(testHelper.boolProp, true);
        Assert.assertEquals(testHelper.stringProp, null);
    }
    @Test
    public void testConfigParserFromStringWithExtra() throws JsonProcessingException {
        String testString = String.join("\n",
                "stringProp: abc",
                "intProp: 123",
                "invalidProp: 123");
        TestHelper testHelper = ConfigParser.loadConfigFromString(testString, TestHelper.class);
        Assert.assertEquals(testHelper.intProp, 123);
        Assert.assertEquals(testHelper.boolProp, false);
        Assert.assertEquals(testHelper.stringProp, "abc");
    }

}

