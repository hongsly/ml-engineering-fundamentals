# Day 17 Quick Reference: Kafka & Feature Stores

**Study Date**: 2025-11-13 (Week 3, Day 3)
**Topics**: Apache Kafka fundamentals, Feature Store architecture
**Knowledge Check Score**: 95.1% (A)

---

## Kafka Fundamentals

### Core Concepts

**Topic & Partitions**
- **Topic**: Log of events (e.g., "user-clicks", "payments")
    - Messages in a topic are immutable
    - Replay support
- **Partition**: Physical subdivision within topic for parallelism
- **Key-based routing**: Records with same key → same partition (ordering guarantee)
- **Partition count**: Determines max consumer parallelism

**Brokers & Clusters**
- **Broker**: Kafka server that stores data
- **Cluster**: Group of brokers working together
- **Partition leader**: Broker handling all reads/writes for that partition
- **Replication factor**: Number of copies per partition (typically 3)

**In-Sync Replicas (ISR)**
- **Definition**: Replicas that are active and caught up with leader
    - a write is considered commited only if synced to all ISR
- **Leader election**: When leader fails, new leader chosen from ISR
- **ISR maintenance**: Replica falls out of ISR if it lags beyond threshold

### Producers

**Acks Configuration** (durability vs latency tradeoff):
- **acks=0**: Fire-and-forget (fastest, least durable)
- **acks=1**: Wait for leader acknowledgment (balanced)
- **acks=all**: Wait for all ISR replicas (slowest, most durable)

**Partitioning Strategy**:
- With key: Hash(key) % num_partitions → ensures ordering per key
- Without key: Round-robin across partitions

### Consumers

Consumer subscribe to topics; only gets committed messages.

**Consumer Groups**
- **Purpose**: Scale consumption, load balancing, fault tolerance
- **Partition assignment**: Each partition consumed by exactly 1 consumer in group
- **Max parallelism**: # consumers ≤ # partitions

**Offset Management**
- **Offset**: Position in partition (sequential ID), tracked in Kafka internal topic
- **Auto-commit**: Periodic automatic offset commits (may lose/duplicate on crash)
- **Manual commit**: Explicit commit after processing (exactly-once semantics)

**Rebalancing**
- **Trigger**: Consumer joins/leaves/crashes, partition added
- **Impact**: Brief pause in consumption during reassignment
- **Strategies**: Range, round-robin, sticky (minimize movement)

### Metadata Management

**ZooKeeper (Legacy)**
- Stores cluster metadata (brokers, topics, configs)
- Coordinates leader election
- External dependency, operational complexity

**KRaft (Kafka Raft)**
- Built-in consensus using Raft protocol
- No ZooKeeper dependency
- Simplified operations, faster metadata updates

---

## Feature Stores

### Purpose & Benefits

**Core Problem Solved**:
- **Training-serving skew**: Different feature logic in training vs serving
- **Feature reusability**: Share features across teams/models
- **Consistency**: Same feature values for online/offline use
- **Monitoring**: Track feature drift and data quality

### Architecture

**Online Store** (Low-latency serving):
- **Purpose**: Real-time model predictions (< 10ms)
- **Storage**: Redis, DynamoDB, Cassandra (key-value stores)
- **Data freshness**: Streaming updates (Kafka, Kinesis)
- **Use case**: Production inference API

**Offline Store** (Batch training):
- **Purpose**: Model training, batch scoring, backfills
- **Storage**: S3, Snowflake, BigQuery (data warehouses)
- **Data freshness**: Hourly/daily batch jobs
- **Use case**: Training data generation

### Point-in-Time Correctness

**Definition**: Ensures training features match what was available at prediction time

**Problem Without It**:
```
Training: Use features from transaction_time (e.g., 2024-01-15 10:00)
But join fetches user_age from latest snapshot (e.g., 2024-12-01)
→ Data leakage: using future information
```

**Solution**:
- Store feature values with timestamps
- Query: "Get user features as of 2024-01-15 10:00"
- Prevents future information from leaking into training

### Feature Transformations

**Batch Transformations**:
- Spark/Airflow jobs on data warehouse
- Complex aggregations (30-day click rate)
- Scheduled (hourly/daily)

**Streaming Transformations**:
- Flink/Spark Streaming on event streams
- Simple operations (last click timestamp)
- Real-time updates

**On-Demand Transformations**:
- Computed at request time (age from birthdate)
- Low-latency requirements (< 5ms)
- Stateless operations

### Feature Registry

**Metadata Stored**:
- Feature definitions (name, type, description)
- Transformation logic (SQL, Python code)
- Owner, version, lineage
- Statistics (min/max/mean/null rate)

**Benefits**:
- Discoverability: Search existing features before creating new
- Documentation: Understand feature semantics
- Governance: Track ownership and dependencies

### Monitoring

* data drift
* train/serve skew
* operational metrics


---

## Interview Q&A

### Kafka

**Q: What happens when a consumer in a group crashes?**

A: The group coordinator detects the failure (via heartbeat timeout) and triggers a rebalancing. The crashed consumer's partitions are reassigned to remaining consumers. During rebalancing, consumption pauses briefly. If offsets were committed, new consumer resumes from last committed offset; otherwise, behavior depends on auto.offset.reset (earliest/latest).

**Q: How does Kafka guarantee ordering?**

A: Kafka guarantees ordering within a partition, not across partitions. Use keyed messages to route related events to the same partition (Hash(key) % num_partitions). Example: All events for user_id=123 go to same partition, maintaining order.

**Q: Explain acks=all vs acks=1 tradeoff**

A:
- **acks=1**: Leader writes to log and acknowledges (lower latency, risk of data loss if leader fails before replication)
- **acks=all**: Leader waits for all ISR replicas to write (higher latency, no data loss as long as 1 ISR survives)
- **Tradeoff**: Durability vs throughput/latency

**Q: When would you use multiple consumer groups?**

A: When different applications need to consume the same topic independently:
- Example: "user-clicks" topic consumed by:
  - Group 1: Real-time analytics dashboard
  - Group 2: Recommendation model retraining
  - Group 3: Fraud detection system
- Each group maintains independent offsets and processes all messages.

### Feature Stores

**Q: Why can't we just use a database for features?**

A: Databases lack ML-specific capabilities:
- No point-in-time correctness (joins use latest data)
- No train/serve consistency guarantees
- No feature versioning or lineage tracking
- No built-in monitoring for feature drift
- Manual coordination between online/offline stores

**Q: What is training-serving skew?**

A: Inconsistency between features used in training vs production:
- **Example**: Training computes "days_since_signup" in Spark (UTC), production computes in Python service (local time) → different values
- **Solution**: Feature store uses same transformation logic for both, preventing skew

**Q: How does point-in-time correctness prevent data leakage?**

A: Without it, training joins might use future information:
- Example: Training on Jan 15 transaction, but join fetches user's credit score from latest snapshot (Dec 1), which includes events after Jan 15
- Point-in-time join: "Get features as of Jan 15" → uses only historical data available at prediction time

**Q: Online vs offline store - when to use each?**

A:
- **Online**: Real-time inference (< 10ms latency) - use Redis/DynamoDB
- **Offline**: Batch training/scoring (large scans OK) - use Snowflake/S3
- **Sync**: Pipeline keeps both stores consistent with same feature values

---

## Common Pitfalls

### Kafka

1. **Consumer parallelism**: More consumers than partitions → idle consumers (wasted resources)
2. **Key design**: Poor key distribution → hot partitions (unbalanced load)
3. **Auto-commit**: Enables at-least-once delivery, may duplicate processing on crash
4. **Rebalancing storms**: Frequent consumer restarts → constant rebalancing → no progress

### Feature Stores

1. **Ignoring point-in-time**: Causes data leakage, inflated offline metrics
2. **Stale online features**: Forgot to set up streaming pipeline → serving outdated data
3. **Over-transformation**: Complex on-demand transforms → high latency (> 10ms)
4. **No monitoring**: Feature drift undetected → silent model degradation

---

## Key Formulas & Patterns

### Kafka Sizing

**Max consumer parallelism per group**:
```
max_consumers = num_partitions
```

**Replication overhead**:
```
storage_per_topic = retention_size × replication_factor
```

**Throughput calculation**:
```
total_throughput = num_partitions × per_partition_throughput
```

### Feature Store Latency

**Online serving SLA**:
```
p99_latency = feature_fetch (< 5ms) + transformation (< 5ms) + model_inference
```

**Offline training data generation**:
```
training_set = point_in_time_join(labels, features, timestamp_column)
```

---

## Resources Studied

**Kafka**:
- Official docs: [kafka.apache.org](https://kafka.apache.org/documentation/)
- Key sections: Producer configs, Consumer groups, Replication

**Feature Stores**:
- Feast documentation: [feast.dev](https://feast.dev/)
- Tecton architecture: [tecton.ai/blog](https://www.tecton.ai/blog/)

**Interview prep focus**:
- Kafka: Consumer group mechanics, ISR/leader election, acks configuration
- Feature Stores: Point-in-time correctness (most commonly asked), online/offline separation

---

## Next Steps

**Day 18 (Tomorrow)**: Airflow DAGs + Feature Store deep dive
**Day 19 (Saturday)**: Docker + Kubernetes basics
**Review**: kafka_architecture, kafka_producers_consumers due 2025-11-14
