# C4 Level 2 – Container Diagram
## Purpose
This diagram describes the major containers, services, data stores, and infrastructure that make up the Enterprise AI Platform and how they interact.
---
## Container Diagram
```text
                                        +--------------------------------------+
                                        |             End Users                |
                                        |--------------------------------------|
                                        | Data Engineers                       |
                                        | Data Scientists                      |
                                        | AI Engineers                         |
                                        | Business Analysts                    |
                                        | Executives                           |
                                        +------------------+-------------------+
                                                           |
                                                           |
                                                 HTTPS / REST / GraphQL
                                                           |
                                                           v
                                          +-------------------------------+
                                          | API Gateway / Ingress         |
                                          | Rate Limiting                 |
                                          | Routing                       |
                                          | API Versioning                |
                                          +---------------+---------------+
                                                          |
                                                          |
                                          +---------------v---------------+
                                          | Identity & Access             |
                                          | Keycloak / Azure AD / OAuth2  |
                                          | RBAC                          |
                                          | JWT                           |
                                          +---------------+---------------+
                                                          |
                            ----------------------------------------------------------------
                            |              |              |              |                  |
                            |              |              |              |                  |
                            v              v              v              v                  v
                  +---------------+ +---------------+ +---------------+ +---------------+ +---------------+
                  | Data APIs     | | AI APIs       | | Agent APIs    | | Admin APIs    | | Metadata APIs |
                  +-------+-------+ +-------+-------+ +-------+-------+ +-------+-------+ +-------+-------+
                          |                 |                 |                 |                 |
                          -----------------------------------------------------------------------------------
                                                          |
                                                          v
                                        +--------------------------------------+
                                        | Enterprise Service Layer             |
                                        |--------------------------------------|
                                        | Business Services                    |
                                        | Workflow Services                    |
                                        | Notification Services                |
                                        | Integration Services                 |
                                        +------------------+-------------------+
                                                           |
              ---------------------------------------------------------------------------------------------
              |                         |                        |                     |                    |
              |                         |                        |                     |                    |
              v                         v                        v                     v                    v

    +------------------+      +------------------+      +------------------+     +----------------+    +----------------+
    | Kafka Cluster    |      | Airflow          |      | FastAPI          |     | GraphQL        |    | EventBridge    |
    | Event Streaming  |      | Orchestration    |      | Microservices    |     | Federation     |    | Event Routing  |
    +--------+---------+      +--------+---------+      +--------+---------+     +--------+-------+    +--------+-------+
             |                         |                         |                         |                     |
             |                         |                         |                         |                     |
             -----------------------------------------------------------------------------------------------
                                                     |
                                                     v

                                      +-------------------------------+
                                      | Apache Spark Cluster          |
                                      | Batch Processing              |
                                      | Structured Streaming          |
                                      | ML Processing                 |
                                      +---------------+---------------+
                                                      |
                    -----------------------------------------------------------------------
                    |                         |                          |                  |
                    |                         |                          |                  |
                    v                         v                          v                  v

            +---------------+        +---------------+          +---------------+   +---------------+
            | Delta Lake    |        | dbt           |          | ML Pipeline   |   | Feature Store |
            | Bronze         |        | Transform     |          | Training      |   | Feature Mgmt  |
            | Silver         |        | Semantic      |          | Evaluation    |   |               |
            | Gold           |        | Models        |          | Deployment    |   |               |
            +-------+--------+        +-------+-------+          +-------+-------+   +-------+-------+
                    |                         |                          |                   |
                    -------------------------------------------------------------------------
                                                  |
                                                  v

                                  +-------------------------------------+
                                  | Snowflake Data Warehouse            |
                                  | Enterprise Reporting                |
                                  | Analytics                           |
                                  +----------------+--------------------+
                                                   |
                                                   |
                                        +----------v-----------+
                                        | BI & Analytics       |
                                        | Power BI             |
                                        | Tableau              |
                                        | Apache Superset      |
                                        +----------------------+

============================================================================================

                         Artificial Intelligence Platform

============================================================================================

             +------------------------+
             | Document Ingestion     |
             | PDFs                   |
             | SharePoint             |
             | Confluence             |
             | APIs                   |
             +-----------+------------+
                         |
                         v

             +------------------------+
             | Embedding Service      |
             | Chunking               |
             | Metadata               |
             | Embeddings             |
             +-----------+------------+
                         |
                         v

             +------------------------+
             | Vector Database        |
             | Qdrant                 |
             | Collections            |
             | Similarity Search      |
             +-----------+------------+
                         |
                         v

             +------------------------+
             | Retrieval Layer        |
             | Hybrid Search          |
             | Context Assembly       |
             +-----------+------------+
                         |
                         v

             +------------------------+
             | LLM Gateway            |
             | OpenAI                 |
             | Anthropic              |
             | Gemini                 |
             | Bedrock                |
             +-----------+------------+
                         |
                         v

             +------------------------+
             | Multi-Agent Platform   |
             | LangGraph              |
             | Planning               |
             | Tool Execution         |
             +-----------+------------+
                         |
                         v

             +------------------------+
             | AI Applications        |
             | Enterprise Assistant   |
             | Analytics Copilot      |
             | HR Agent               |
             | Finance Agent          |
             | DevOps Agent           |
             +------------------------+

============================================================================================

                    Cross-Cutting Platform Services

============================================================================================

+------------------------------------------------------------------------------------------+
| Observability                                                                             |
| OpenTelemetry • Prometheus • Grafana • Loki • Jaeger • Alertmanager                      |
+------------------------------------------------------------------------------------------+

+------------------------------------------------------------------------------------------+
| Governance                                                                                |
| Data Catalog • Lineage • Metadata • Audit Logs • Policy Engine                           |
+------------------------------------------------------------------------------------------+

+------------------------------------------------------------------------------------------+
| Security                                                                                  |
| RBAC • Secrets Manager • Vault • Encryption • Network Policies • IAM                     |
+------------------------------------------------------------------------------------------+

+------------------------------------------------------------------------------------------+
| DevOps                                                                                    |
| GitHub Actions • Docker • Terraform • Kubernetes • Helm                                 |
+------------------------------------------------------------------------------------------+
```

---

## Container Responsibilities

| Container | Responsibility |
|-----------|----------------|
| API Gateway | Entry point for all external traffic |
| Identity | Authentication and authorization |
| Enterprise Service Layer | Business orchestration |
| Kafka | Event streaming backbone |
| Spark | Batch and streaming processing |
| Delta Lake | Lakehouse storage |
| dbt | Transformations and semantic models |
| Snowflake | Enterprise analytics warehouse |
| Document Ingestion | Enterprise knowledge ingestion |
| Embedding Service | Create vector embeddings |
| Qdrant | Vector similarity search |
| Retrieval Layer | Context retrieval |
| LLM Gateway | Multi-model routing |
| Multi-Agent Platform | Agent orchestration |
| Observability | Metrics, logs, traces, alerting |
| Governance | Metadata, lineage, compliance |
| Security | IAM, encryption, secrets management |
| DevOps | CI/CD and infrastructure automation |

---

## References

- Vision
- Business Context
- Logical Architecture
- Physical Architecture
- ADR-001 to ADR-005
