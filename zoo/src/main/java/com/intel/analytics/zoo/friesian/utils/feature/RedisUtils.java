package com.intel.analytics.zoo.friesian.utils.feature;


import redis.clients.jedis.*;
import redis.clients.jedis.exceptions.JedisConnectionException;
import com.intel.analytics.zoo.friesian.utils.feature.FeatureUtils;

import java.util.*;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.log4j.Logger;


public class RedisUtils {
    private static final Logger logger = Logger.getLogger(RedisUtils.class.getName());
    private static RedisUtils instance = null;
    private static JedisPool jedisPool = null;
    private static JedisCluster cluster = null;

    private RedisUtils() {
        JedisPoolConfig jedisPoolConfig = new JedisPoolConfig();
        jedisPoolConfig.setMaxTotal(256);


        if (FeatureUtils.helper().redisHostPort().size() == 1) {
            jedisPool = new JedisPool(jedisPoolConfig, FeatureUtils.helper().redisHostPort().get(0)._1,
                    (int)FeatureUtils.helper().redisHostPort().get(0)._2);
        } else {
            Set<HostAndPort> hps = new HashSet<HostAndPort>();
            for (int i = 0; i < FeatureUtils.helper().redisHostPort().size(); i++) {
                HostAndPort hp = new HostAndPort(FeatureUtils.helper().redisHostPort().get(i)._1,
                        (int)FeatureUtils.helper().redisHostPort().get(i)._2);
                hps.add(hp);
            }
            // default maxAttempt=5, service likely to down, increase to 20
            cluster = new JedisCluster(hps, 50000, 20, jedisPoolConfig);
        }

    }
    public JedisCluster getCluster() {
        return cluster;
    }
    public static RedisUtils getInstance() {
        if (instance == null) {
            instance = new RedisUtils();
        }
        return instance;
    }

    public Jedis getRedisClient() {
        Jedis jedis = null;
        int cnt = 10;
        while (jedis == null) {
            try {
                jedis = jedisPool.getResource();
            } catch (JedisConnectionException e) {
                e.printStackTrace();
                cnt--;
                if (cnt <= 0) {
                    throw new Error("Cannot get redis from the pool");
                }
                try {
                    Thread.sleep(100);
                } catch (InterruptedException ex) {
                    ex.printStackTrace();
                }
            }
        }
        return jedis;
    }
    public void Hset(String keyPrefix, List<String>[] dataArray) {
        if (cluster == null) {
            piplineHmset(keyPrefix, dataArray);
        } else {
            clusterHset(keyPrefix, dataArray);
        }
    }

    public void clusterHset(String keyPrefix, List<String>[] dataArray) {
        int cnt = 0;
        for(List<String> data: dataArray) {
            if(data.size() != 2) {
                logger.warn("Data size in dataArray should be 2, but got" + data.size());
            } else {
                String hKey = FeatureUtils.helper().getRedisKeyPrefix() + keyPrefix + ":" +
                        data.get(0);
                Map<String, String> hValue = new HashMap<>();
                hValue.put("value", data.get(1));
                getCluster().hset(hKey, hValue);
                cnt += 1;
            }
        }
        logger.info(cnt + " valid records written to redis.");
    }
    public void piplineHmset(String keyPrefix, List<String>[] dataArray) {
        Jedis jedis = getRedisClient();
        Pipeline ppl = jedis.pipelined();
        int cnt = 0;
        for(List<String> data: dataArray) {
            if(data.size() != 2) {
                logger.warn("Data size in dataArray should be 2, but got" + data.size());
            } else {
                String hKey = FeatureUtils.helper().getRedisKeyPrefix() + keyPrefix + ":" +
                        data.get(0);
                Map<String, String> hValue = new HashMap<>();
                hValue.put("value", data.get(1));
                ppl.hmset(hKey, hValue);
                cnt += 1;
            }
        }
        ppl.sync();
        jedis.close();
        logger.info(cnt + " valid records written to redis.");
    }

    // TODO: close
//    public void closePool() {
//        jedisPool.close();
//        jedisPool = null;
//    }
}
