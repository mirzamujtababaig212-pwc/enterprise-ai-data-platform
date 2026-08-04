# Physical Deployment Diagram
## Purpose
This diagram illustrates how the Enterprise AI Platform is deployed across cloud infrastructure, Kubernetes clusters, managed services, networking, and external integrations.
It complements the C4 Container Diagram by showing deployment topology rather than logical responsibilities.
---
# Physical Deployment
```text
                                            Internet
                                                │
                                   +------------▼-------------+
                                   │       DNS / CDN          │
                                   │ CloudFront / Azure CDN   │
                                   +------------+-------------+
                                                │
                                                ▼
                                   +--------------------------+
                                   │ Web Application Firewall │
                                   │ AWS WAF / Azure WAF      │
                                   +------------+-------------+
                                                │
                                                ▼
                                   +--------------------------+
                                   │ Load Balancer            │
                                   │ ALB / NGINX Ingress      │
                                   +------------+-------------+
                                                │
                           ==========================================
                           Kubernetes Production Cluster
                           ==========================================

+--------------------------------------------------------------------------------------+
| Kubernetes Control Plane                                                             |
|--------------------------------------------------------------------------------------|
| API Server • Scheduler • Controller Manager • etcd                                  |
+--------------------------------------------------------------------------------------+

         ┌───────────────────────┬────────────────────────┬─────────────────────────┐
         │                       │                        │                         │
         ▼                       ▼                        ▼                         ▼

+-------------------+   +-------------------+   +-------------------+   +-------------------+
| Worker Node 1     |   | Worker Node 2     |   | Worker Node 3     |   | Worker Node 4     |
|-------------------|   |-------------------|   |-------------------|   |-------------------|
| API Gateway       |   | Kafka Brokers     |   | Spark Executors   |   | AI Services       |
| FastAPI           |   | EventBridge       |   | dbt Runner         |   | LangGraph         |
| GraphQL           |   | Airflow Workers   |   | Feature Store      |   | Embedding Service |
| Auth Service      |   | Integration APIs  |   | ML Services        |   | LLM Gateway       |
+-------------------+   +-------------------+   +-------------------+   +-------------------+

                           ==========================================
                                      Managed Services
                           ==========================================

+--------------------------------------------------------------------------------------+
| Snowflake                                                                            |
| Enterprise Data Warehouse                                                            |
+--------------------------------------------------------------------------------------+

+--------------------------------------------------------------------------------------+
| Object Storage                                                                        |
| Amazon S3 / Azure Data Lake / Google Cloud Storage                                   |
+--------------------------------------------------------------------------------------+

+--------------------------------------------------------------------------------------+
| Qdrant Cluster                                                                        |
| Vector Database                                                                       |
+--------------------------------------------------------------------------------------+

+--------------------------------------------------------------------------------------+
| Redis Cluster                                                                         |
| Cache • Session Store                                                                 |
+--------------------------------------------------------------------------------------+

+--------------------------------------------------------------------------------------+
| PostgreSQL                                                                            |
| Metadata • Workflow State • Configuration                                             |
+--------------------------------------------------------------------------------------+

                           ==========================================
                                 External Enterprise Systems
                           ==========================================

+---------------------------------------------------------+
| SAP ERP                                                 |
+---------------------------------------------------------+

+---------------------------------------------------------+
| Salesforce                                               |
+---------------------------------------------------------+

+---------------------------------------------------------+
| ServiceNow                                               |
+---------------------------------------------------------+

+---------------------------------------------------------+
| SharePoint                                               |
+---------------------------------------------------------+

+---------------------------------------------------------+
| Microsoft Teams                                          |
+---------------------------------------------------------+

+---------------------------------------------------------+
| Slack                                                    |
+---------------------------------------------------------+

                           ==========================================
                                    AI Model Providers
                           ==========================================

+---------------------------------------------------------+
| OpenAI                                                   |
+---------------------------------------------------------+

+---------------------------------------------------------+
| Anthropic Claude                                         |
+---------------------------------------------------------+

+---------------------------------------------------------+
| Google Gemini                                            |
+---------------------------------------------------------+

+---------------------------------------------------------+
| AWS Bedrock                                              |
+---------------------------------------------------------+

                           ==========================================
                                     Monitoring Stack
                           ==========================================

+---------------------------------------------------------+
| Prometheus                                               |
+---------------------------------------------------------+

+---------------------------------------------------------+
| Grafana                                                  |
+---------------------------------------------------------+

+---------------------------------------------------------+
| Loki                                                     |
+---------------------------------------------------------+

+---------------------------------------------------------+
| Jaeger                                                   |
+---------------------------------------------------------+

+---------------------------------------------------------+
| OpenTelemetry                                            |
+---------------------------------------------------------+
```

---

# Deployment Layers

## Edge Layer

- DNS
- CDN
- WAF
- Load Balancer

Responsibilities

- Global routing
- TLS termination
- DDoS protection
- Request filtering

---

## Kubernetes Layer

Responsibilities

- Microservices
- API services
- Authentication
- Kafka
- Airflow
- Spark
- AI services

Scaling

- Horizontal Pod Autoscaler
- Cluster Autoscaler

---

## Data Layer

Includes

- Delta Lake
- Snowflake
- PostgreSQL
- Redis
- Qdrant

---

## AI Layer

Includes

- Embedding Service
- Retrieval Service
- LLM Gateway
- Agent Runtime

---

## Enterprise Integration Layer

- SAP
- Salesforce
- SharePoint
- ServiceNow
- Microsoft Teams
- Slack

---

## Observability Layer

Provides

- Metrics
- Logs
- Distributed tracing
- Alerting
- Dashboards

---

# High Availability

The deployment supports:

- Multiple Kubernetes worker nodes
- Load-balanced API services
- Replicated Kafka brokers
- Highly available Redis
- Managed Snowflake
- Replicated PostgreSQL
- Distributed Spark executors

---

# Disaster Recovery Considerations

Recovery objectives:

- RPO: ≤15 minutes
- RTO: ≤1 hour

Strategies:

- Cross-region backups
- Infrastructure as Code
- Automated redeployment
- Database snapshots
- Object storage replication

---

# References

- C4 Context Diagram
- C4 Container Diagram
- Logical Architecture
- Physical Architecture
- Security Architecture
