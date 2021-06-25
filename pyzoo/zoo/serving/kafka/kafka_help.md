# Kafka Guide
This is a guide to use python to run kafka.

You can use `kafka_example.py` to test the kafka environment and get basic usages of kafka by python.

## Install Kafka
To setup the kafka environment, install kafka from [Kafka Download](https://kafka.apache.org/downloads).

## Install Python Package
To use kafka in python, run `pip install kafka-python`.
    
## Test With Example
### Start Kafka
First go to your kafka folder `cd <pathToKafka>`.

With default settings, you can start the zookeeper by
```bash
bin/zookeeper-server-start.sh config/zookeeper.properties
```

Then start the kafka by   
```bash
bin/kafka-server-start.sh config/server.properties
```

### Run python test
Now use `kafka_example.py` to run a simple test. `kafka_example.py` includes constructors of both a **Producer** and a **Consumer**. You can first build a consumer with
```python
python kafka_examples.py consumer_demo
```
this will open up a consumer listening for messages from a kafka topic (initially no output).

Then go to a new terminal and run
```python
python kafka_examples.py producer_demo
```
to build a producer and send some sample messages to the kafka topic. You should see the output
```bash
send 0
send 1
send 2
```
Now switch back to the consumer terminal, you should see the messages sent by the producers have already been received by the consumers
```bash
receive, key: test, value: 0
receive, key: test, value: 1
receive, key: test, value: 2
```


