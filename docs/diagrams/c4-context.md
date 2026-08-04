# C4 Level 1 – System Context
## Purpose
This diagram illustrates how external users and enterprise systems interact with the Enterprise AI Platform.
---
## System Context
```text
                               +-----------------------+
                               |   Business Users      |
                               +----------+------------+
                                          |
                                          |
                               +----------v------------+
                               |  Enterprise AI        |
                               |      Platform         |
                               +----------+------------+
                                          |
      -------------------------------------------------------------------------
      |            |              |             |              |               |
      |            |              |             |              |               |
+-----v----+ +-----v----+ +-------v------+ +----v-----+ +------v------+ +------v------+
| Snowflake| | Databricks| | Salesforce | | SAP ERP | | SharePoint | | External APIs |
+----------+ +-----------+ +------------+ +----------+ +------------+ +-------------+
                                          |
                                          |
                                   +------v------+
                                   | Power BI    |
                                   | Tableau     |
                                   | Superset    |
                                   +-------------+
```
---
## Actors
Business Users
- Data Engineers
- Data Scientists
- AI Engineers
- Analysts
- Executives
- Operations Teams
---
## External Systems
Enterprise Applications
- SAP
- Salesforce
- Oracle
- ServiceNow
Storage
- S3
- Azure Data Lake
- Google Cloud Storage
Reporting
- Power BI
- Tableau
- Apache Superset
Identity
- Azure AD
- Keycloak
- Okta
AI Providers
- OpenAI
- Anthropic
- Google Gemini
- AWS Bedrock
---
## Responsibilities
The Enterprise AI Platform provides:
- Data ingestion
- Streaming
- Lakehouse
- Data warehousing
- AI inference
- RAG
- Multi-agent workflows
- Enterprise governance
- Monitoring
- Security
---
## References
- Vision
- Business Context
- Logical Architecture
