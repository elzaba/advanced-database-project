# Advanced Database Topics Project

This repository provides a Docker-based monitoring solution for Kafka clusters using Prometheus. The tool tracks Kafka health metrics, enabling users to visualize and monitor the health and performance of Kafka services in real-time.

## Requirements

- [Docker](https://docs.docker.com/get-docker/)

Ensure Docker is installed before running the monitoring environment.

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/elzaba/advanced-database-project.git
cd advanced-database-project
```

### 2. Build Docker images
To build all required images, run:
```bash
docker compose build
```

### 3. Start the environment
To start all services in detached mode, use the following command:
```bash
docker compose up -d
```

### 4. Access Kafka Health Metrics
Metrics will be accessible through Prometheus. To view metrics collected from Kafka and related services:
1. Open Prometheus in your browser at `http://localhost:9090`
2. Use the query interface to inspect various Kafka-related metrics.

## Purpose of Each Service

- **Zookeeper (zookeeper-1, zookeeper-2, zookeeper-3)**

  Zookeeper is a coordination service for distributed systems. It coordinates Kafka brokers and stores metadata. Running multiple instances provides reliability and failover support.

- **Kafka Brokers (kafka-1, kafka-2, kafka-3)**

  Kafka is a distributed messaging system that handles high-throughput, fault-tolerant data streams. Each broker handles part of the Kafka cluster workload.

- **Producer Application (producer)**

  This service produces messages and sends them to Kafka topics, simulating data input.

- **Consumer Application (consumer)**

  This service consumes messages from Kafka topics, simulating data consumption from Kafka.

- **Kafka Streams Application (streams)**

  Processes data in real time from the sample topic and publishes results to the count-even-odd-entries topic using Kafka Streams API.

- **Kafka Connect (connect)**

  Kafka Connect allows integration between Kafka and external systems (such as databases, Hadoop, etc.), enabling seamless data movement.

- **Prometheus (prometheus)**

  Prometheus is a monitoring system that collects and stores time-series metrics from Kafka and other services in the environment.

- **JMX Exporter (jmx-exporter)**

  The JMX Exporter is used to expose metrics from Java-based applications (like Kafka) for Prometheus.

- **Node Exporter (node-exporter)**

  Node Exporter collects system-level metrics (CPU, memory, disk usage) from the Kafka environment.

## Steps to Run the Dynamic Alert Processor

1. Install Required Python Dependencies

   The requirements.txt file contains the necessary dependencies for the Dynamic Alert Processor. Install them by running:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Dynamic Alert Processor

   Once the virtual environment is set up and dependencies are installed, you can run the alert processor by executing the following command:
   ```bash
   python dynamic_alert_processor.py
   ```

## Stopping the Environment

To stop all services, run:
```bash
docker compose down
```

## Utilizing Kafka Connect

To use Kafka Connect and gather related metrics:
### 1. Create a Kafka Topic

```bash
docker-compose exec kafka-1 bash -c 'KAFKA_OPTS="" kafka-topics --create --partitions 4 --replication-factor 3 --topic connect-topic --zookeeper zookeeper-1:2181'
```
### 2. Produce Messages to the Topic

```bash
docker-compose exec kafka-1 bash -c 'KAFKA_OPTS="" kafka-producer-demo --throughput 500 --num-records 100000000 --topic connect-topic --record-size 100 --producer-props bootstrap.servers=kafka-1:9092'
```
### 3. Consume Messages from the Topic

```bash
docker-compose exec kafka-1 bash -c 'KAFKA_OPTS="" consumer-demo --messages 100000000 --threads 1 --topic connect-topic --broker-list kafka-1:9092 --timeout 60000'
```
### 4. Create a Sink Connector
Sinks in Kafka Connect refer to sink connectors, which are components that take data from Kafka topics and send it to external systems or storage. These can include databases, file systems, or other services.
```bash
docker-compose exec connect \
     curl -X PUT \
     -H "Content-Type: application/json" \
     --data '{
            "connector.class":"org.apache.kafka.connect.file.FileStreamSinkConnector",
            "tasks.max":"4",
            "file": "/tmp/testconnect.txt",
            "topics": "connect-topic"
}' \
     http://localhost:8083/connectors/file-sink/config | jq .
```
### 5. Verify Data Output

```bash
docker-compose exec connect bash -c 'tail -10 /tmp/testconnect.txt'
```
