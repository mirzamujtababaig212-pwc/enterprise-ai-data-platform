# Data Flow Diagram

## Purpose

This document describes the end-to-end movement of data through the Enterprise AI Platform, from ingestion to analytics, machine learning, retrieval-augmented generation (RAG), and business consumption.

It complements the C4 Context, C4 Container, and Physical Deployment diagrams by focusing on **data movement and processing pipelines**.

---

# High-Level Data Flow

```text
                           +--------------------------------------+
                           | Enterprise Source Systems            |
                           |--------------------------------------|
                           | SAP ERP                              |
                           | Salesforce                           |
                           | ServiceNow                           |
                           | SharePoint                           |
                           | REST APIs                            |
                           | IoT Devices                          |
                           | Files (CSV, JSON, Parquet, PDF)      |
                           +------------------+-------------------+
                                              |
                                              |
                                     Batch / CDC / Streaming
                                              |
                                              v
                          +---------------------------------------+
                          | Ingestion Layer                       |
                          |---------------------------------------|
                          | Kafka                                |
                          | Airflow                              |
                          | REST Connectors                      |
                          | CDC Connectors                       |
                          +------------------+--------------------+
                                             |
                                             |
                                             v
                          +---------------------------------------+
                          | Apache Spark                          |
                          |---------------------------------------|
                          | Batch Processing                      |
                          | Structured Streaming                  |
                          | Data Validation                       |
                          | Cleansing                             |
                          | Enrichment                            |
                          +------------------+--------------------+
                                             |
                    -------------------------------------------------------
                    |                         |                           |
                    |                         |                           |
                    v                         v                           v

          +----------------+      +---------------------+      +-------------------+
          | Delta Lake     |      | Document Pipeline   |      | Feature Store     |
          | Bronze          |      | PDFs, Docs, APIs    |      | ML Features       |
          | Silver          |      | Chunking            |      | Versioning        |
          | Gold            |      | Metadata Extraction |      |                   |
          +--------+--------+      +----------+----------+      +---------+---------+
                   |                          |                           |
                   |                          |                           |
                   |                          v                           |
                   |             +--------------------------+             |
                   |             | Embedding Service        |             |
                   |             | Vector Generation        |             |
                   |             +-----------+--------------+             |
                   |                         |                            |
                   |                         v                            |
                   |              +-------------------------+             |
                   |              | Qdrant Vector Database  |             |
                   |              +-----------+-------------+             |
                   |                          |                           |
                   --------------------------------------------------------
                                              |
                                              v
                              +-------------------------------+
                              | Retrieval Layer               |
                              | Hybrid Search                 |
                              | Context Assembly              |
                              +---------------+---------------+
                                              |
                                              v
                              +-------------------------------+
                              | LLM Gateway                   |
                              | OpenAI                        |
                              | Claude                        |
                              | Gemini                        |
                              | Bedrock                       |
                              +---------------+---------------+
                                              |
                                              v
                              +-------------------------------+
                              | Multi-Agent Platform          |
                              | Planning                      |
                              | Tool Execution                |
                              | Memory                        |
                              +---------------+---------------+
                                              |
                           -----------------------------------------------
                           |                     |                      |
                           v                     v                      v

                  +----------------+    +----------------+    +-------------------+
                  | AI Assistant   |    | Analytics      |    | Business Agents   |
                  | Enterprise Q&A |    | Copilot        |    | HR, Finance, Dev  |
                  +----------------+    +----------------+    +-------------------+

                                              |
                                              v

                              +-------------------------------+
                              | Analytics & Reporting         |
                              | Snowflake                     |
                              | dbt Models                    |
                              | Power BI                      |
                              | Tableau                       |
                              | Apache Superset               |
                              +-------------------------------+
```

---

# Data Processing Stages

## 1. Data Ingestion

### Sources

- ERP systems
- CRM systems
- Enterprise applications
- Cloud storage
- APIs
- IoT streams
- Document repositories

### Ingestion Methods

- Batch
- Change Data Capture (CDC)
- Streaming
- Event-driven APIs

---

## 2. Data Processing

Apache Spark performs:

- Validation
- Schema enforcement
- Data cleansing
- Deduplication
- Enrichment
- Transformation

---

## 3. Lakehouse Storage

### Bronze

Raw immutable data.

### Silver

Validated and standardized data.

### Gold

Business-ready datasets.

---

## 4. Semantic Modeling

dbt transforms curated datasets into:

- Star schemas
- Dimensional models
- Business metrics
- Reporting views

---

## 5. AI Knowledge Pipeline

Documents are:

- Collected
- Parsed
- Chunked
- Enriched with metadata
- Embedded
- Indexed into Qdrant

---

## 6. Retrieval-Augmented Generation (RAG)

User query flow:

1. Receive query
2. Embed query
3. Search Qdrant
4. Retrieve relevant context
5. Assemble prompt
6. Invoke selected LLM
7. Return grounded response

---

## 7. Machine Learning Pipeline

Workflow:

- Feature engineering
- Feature Store registration
- Model training
- Validation
- Deployment
- Monitoring

---

## 8. Multi-Agent Execution

Agents may:

- Query enterprise data
- Invoke APIs
- Trigger workflows
- Generate reports
- Coordinate with other agents

---

## 9. Analytics Consumption

Business users access:

- Dashboards
- KPIs
- Reports
- AI insights
- Operational metrics

---

# Cross-Cutting Controls

Applied throughout the pipeline:

- Authentication
- Authorization (RBAC)
- Encryption in transit
- Encryption at rest
- Audit logging
- Data lineage
- Metadata catalog
- Data quality checks
- Monitoring and alerting

---

# References

- Vision
- Business Context
- Logical Architecture
- Physical Architecture
- C4 Context Diagram
- C4 Container Diagram
- Physical Deployment Diagram
- Security Architecture
- Governance Architecture
