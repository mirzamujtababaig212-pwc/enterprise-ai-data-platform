# ADR-009: Adopt PostgreSQL as the Enterprise Operational Database

**Status:** Accepted

**Date:** YYYY-MM-DD

**Decision Owners:** Enterprise Architecture Team

---

# Context

The Enterprise AI Platform consists of numerous operational services requiring transactional data storage, including:

- Authentication
- User Management
- API metadata
- Prompt management
- Model registry metadata
- Workflow metadata
- AI agent configuration
- Platform configuration
- Audit logs
- Application settings

While Delta Lake stores analytical datasets and Snowflake supports business intelligence, neither is designed to serve as the transactional database for operational microservices.

A relational database is required to provide ACID transactions, strong consistency, referential integrity, and efficient query capabilities.

---

# Problem Statement

The platform requires an operational database capable of:

- ACID transactions
- High availability
- Relational data modeling
- SQL support
- Strong consistency
- Backup and recovery
- Kubernetes deployment
- Cloud portability
- Horizontal application scaling
- Enterprise security

---

# Decision Drivers

The selected database should provide:

- Mature SQL support
- Strong transaction guarantees
- Excellent reliability
- Open-source ecosystem
- Kubernetes compatibility
- High availability
- JSON document support
- Cloud portability
- Minimal vendor lock-in
- Large enterprise adoption

---

# Options Considered

## Option 1 — PostgreSQL

Advantages

- ACID-compliant transactions
- Mature SQL implementation
- Excellent reliability
- JSONB support
- Strong indexing capabilities
- Rich extension ecosystem
- Large community
- Cloud portability

Disadvantages

- Vertical scaling limitations
- Read replicas required for very large workloads
- Operational administration

---

## Option 2 — MySQL

Advantages

- Mature ecosystem
- Broad adoption
- High performance

Disadvantages

- Less advanced JSON capabilities
- Fewer enterprise analytical extensions
- Less flexibility for complex queries

---

## Option 3 — MongoDB

Advantages

- Flexible schema
- Horizontal scaling
- Document-oriented

Disadvantages

- Not ideal for highly relational workloads
- Eventual consistency considerations
- More difficult transactional modeling

---

## Option 4 — Azure SQL Database

Advantages

- Fully managed
- Enterprise features
- High availability

Disadvantages

- Azure-specific
- Vendor dependency
- Reduced portability

---

# Decision

PostgreSQL is selected as the standard operational database for the Enterprise AI Platform.

It provides enterprise-grade transactional capabilities, relational integrity, strong SQL support, JSON document storage, and seamless integration with Kubernetes-based deployments.

---

# Architecture Impact

PostgreSQL stores:

- User accounts
- Roles
- Permissions
- OAuth metadata
- Platform configuration
- AI agent metadata
- Prompt templates
- Workflow metadata
- Application settings
- Audit records
- Service metadata

---

# Integration Points

PostgreSQL integrates with:

- FastAPI
- Keycloak
- Kubernetes
- API Gateway
- LangGraph
- MLflow
- OpenTelemetry
- Prometheus
- Airflow (metadata database where appropriate)

---

# Data Responsibilities

PostgreSQL is responsible for:

- Transactional storage
- Metadata management
- Configuration storage
- User data
- Security metadata
- Audit records
- Application state

PostgreSQL is not responsible for:

- Large-scale analytics
- Data lake storage
- Machine learning datasets
- Distributed event streaming
- Feature engineering
- BI reporting

---

# Relationship with Delta Lake and Snowflake

Each platform component has a distinct responsibility.

### PostgreSQL

- Transactional database
- Operational metadata
- User information
- Configuration
- Platform state

### Delta Lake

- System of record
- Raw, Bronze, Silver, and Gold datasets
- AI datasets
- Feature engineering
- Historical processing

### Snowflake

- Business intelligence
- Enterprise reporting
- Executive dashboards
- Dimensional models
- Analytical SQL

These technologies complement one another and should not be used interchangeably.

---

# Deployment Architecture

Application

↓

FastAPI

↓

Connection Pool

↓

PostgreSQL Primary

↓

Read Replicas (future)

↓

Automated Backup

---

# Security Considerations

PostgreSQL implements:

- TLS encryption
- Role-Based Access Control (RBAC)
- Least-privilege database roles
- Database auditing
- Encrypted backups
- Secret-based credential management
- Network isolation
- Regular patching

---

# Consequences

## Positive

- Strong transactional consistency
- Mature SQL ecosystem
- Flexible JSON support
- Reliable relational modeling
- Cloud portability
- Enterprise adoption

## Negative

- Operational management
- Backup administration
- Replica management
- Capacity planning

---

# Risks

Potential risks include:

- Database outages
- Long-running queries
- Index fragmentation
- Connection exhaustion
- Storage growth

Mitigation strategies:

- Connection pooling
- High availability deployment
- Automated backups
- Monitoring
- Query optimization
- Routine maintenance

---

# Alternatives Rejected

### MySQL

Rejected because PostgreSQL offers richer SQL capabilities, stronger JSON support, and a broader extension ecosystem aligned with enterprise requirements.

### MongoDB

Rejected because the platform primarily requires transactional consistency and relational data integrity rather than flexible document storage.

### Azure SQL Database

Rejected because the Enterprise AI Platform targets a cloud-agnostic architecture rather than provider-specific services.

---

# Future Considerations

Potential future enhancements include:

- PostgreSQL clustering
- Read replicas
- Logical replication
- Partitioning
- pgvector integration
- Multi-region failover
- Automated failover orchestration

---

# References

Related ADRs:

- ADR-006: Kubernetes
- ADR-007: API Gateway
- ADR-008: FastAPI
- ADR-010: Object Storage
- ADR-012: Keycloak

Related Architecture Documents:

- Logical Architecture
- Physical Architecture
- Security Architecture
- Data Platform Architecture
- Quality Attributes
