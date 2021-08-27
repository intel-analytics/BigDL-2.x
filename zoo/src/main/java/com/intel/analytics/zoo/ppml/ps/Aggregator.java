/*
 * Copyright 2021 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.ppml.ps;

import com.intel.analytics.zoo.ppml.generated.FLProto.DataSplit;
import com.intel.analytics.zoo.ppml.generated.FLProto.Table;
import com.intel.analytics.zoo.ppml.generated.FLProto.TreeLeaves;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;

public abstract class Aggregator {

  protected MemoryStorage trainStorage = new MemoryStorage();
  protected MemoryStorage evalStorage = new MemoryStorage();
  protected MemoryStorage predStorage = new MemoryStorage();
  // temporary storage used in training
  protected final Map<String, Table> trainMap = new HashMap<>();
  // TODO for XGBoost
  protected final Map<String, DataSplit> bestSplit = new HashMap<>();
  protected final Map<String, TreeLeaves> leafMap = new HashMap<>();
  protected final Map<String, DataSplit> splitMap = new HashMap<>();
  protected final Map<String, List<Float>> gradMap = new HashMap<>();
  protected final Map<String, Object> predMap = new HashMap<>();

  // END of XGBoost
  // temporary storage used in evaluation
  protected Map<String, Object> evalMap = new HashMap<>();
  protected Integer clientNum;
  protected Integer currentVersion = 0;
  protected Set<String> clientUUIDSet = new HashSet<>();
  protected ExecutorService pool = Executors.newFixedThreadPool(1, new ThreadFactory() {
    @Override
    public Thread newThread(Runnable r) {
      Thread t = Executors.defaultThreadFactory().newThread(r);
      t.setDaemon(true);
      return t;
    }
  });

  protected void checkVersion(String clientUUID, int version) throws IllegalArgumentException{
    if (version != currentVersion) {
      throw new IllegalArgumentException("Version miss match.");
    }
    if (trainMap.get(clientUUID) != null) {
      throw new IllegalArgumentException("Train data is already existed: clientUUID: " +
              clientUUID + ", version: " + version + ".");
    }
  }

  /**
   * put the temporary data with the same version into current map,
   * aggregate in this operation
   *
   * @param clientUUID
   * @param data
   */
  public void putData(
          String clientUUID,
          int version,
          Table data) throws IllegalArgumentException{
    // check the version

    // put the data in the train map
    synchronized(trainMap) {
      checkVersion(clientUUID, version);
      trainMap.put(clientUUID, data);
      // Aggregate when buffer is full
      if (trainMap.size() >= clientNum) {
        pool.execute(new Runnable() {
          @Override
          public void run() {
            try {
              aggregate();
            } catch(Throwable t) {
              throw t;
            }
          }
        });
      }
    }
  }


  public void putSplit(
          String clientUUID,
          DataSplit dataSplit) {
    synchronized(splitMap) {
      splitMap.put(clientUUID, dataSplit);
      // Aggregate when buffer is full
      if (splitMap.size() >= clientNum) {
        pool.execute(new Runnable() {
          @Override
          public void run() {
            try {
              aggregate();
            } catch(Throwable t) {
              throw t;
            }
          }
        });
      }
    }

  }


  public int putPredict(
          String clientUUID,
          Object data) {
    // put the data in the eval map
    int predictVersion = predStorage.retrieveCurrentVersion();
    synchronized (predMap) {
      if (predMap.containsKey(clientUUID)) {
        try {
          predMap.wait();
        } catch (InterruptedException ie) {
          throw new RuntimeException(ie);
        }
      }
      predMap.put(clientUUID, data);

      // Aggregate when data number reach min requirement
      if (predMap.size() >= clientNum) {
        pool.execute(new Runnable() {
          @Override
          public void run() {
            try {
              aggPredict();
            } catch(Throwable t) {
              throw t;
            }
          }
        });
      }
    }
    return predictVersion + 1;
  }

  public void putTreeLeaves(
          String clientUUID,
          TreeLeaves treeLeaves) {
    synchronized (leafMap) {
      if (leafMap.containsKey(clientUUID)) {
        try {
          leafMap.wait();
        } catch (InterruptedException ie) {
          throw new RuntimeException(ie);
        }
      }
      leafMap.put(clientUUID, treeLeaves);
    }
  }

  public void putEvaluateData(
    String clientUUID,
    Object data,
    final boolean lastMiniBatch) {
    // check the version

    // put the data in the eval map
    synchronized (evalMap) {
      if (evalMap.containsKey(clientUUID)) {
        try {
          evalMap.wait();
        } catch (InterruptedException ie) {
          throw new RuntimeException(ie);
        }
      }
      evalMap.put(clientUUID, data);

      // Aggregate when data number reach min requirement
      if (evalMap.size() >= clientNum) {
        pool.execute(new Runnable() {
          @Override
          public void run() {
            try {
              aggEvaluate(lastMiniBatch);
            } catch(Throwable t) {
              throw t;
            }
          }
        });
      }
    }
  }

  public static String getTreeNodeId(String treeID, String nodeID) {
    return treeID + "_" + nodeID;
  }

  public DataSplit getBestSplit(
          String treeID,
          String nodeID) {
    synchronized(bestSplit) {
      String id = getTreeNodeId(treeID, nodeID);
      while (!bestSplit.containsKey(id)) {
        try {
          bestSplit.wait();
        } catch (InterruptedException ie) {
          throw new RuntimeException(ie);
        }
      }
      return bestSplit.get(id);
    }
  }

  /**
   * get train aggregated result by version
   *
   * @param version
   * @return
   */
  public Table getTrainResult(int version) {
    return trainStorage.retrieve(version);
  }

  public Table getEvaluateResult(int version) {
    return evalStorage.retrieve(version);
  }

  public Table getPredictResult(int version) {
    return predStorage.retrieve(version);
  }

  /**
   * register a new client.
   *
   * @param uuid
   * @return
   */
  public boolean registerClient(String uuid) {
    if (clientUUIDSet.contains(uuid)) {
      return false;
    } else {
      clientUUIDSet.add(uuid);
      return true;
    }
  }

  public abstract void initAgg();

  public abstract void aggregate();

  public abstract void aggEvaluate(boolean agg);

  public abstract void aggPredict();
}
