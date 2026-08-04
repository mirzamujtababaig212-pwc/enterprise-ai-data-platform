# ADR-030: Adopt Centralized Secrets Management for Secure Credential and Key Management

**Status:** Accepted

**Date:** YYYY-MM-DD

**Decision Owners:** Enterprise Architecture Team

---

# Context

The Enterprise AI Platform integrates with numerous internal and external
systems requiring sensitive credentials, including:

- LLM provider API keys
- PostgreSQL credentials
- Kafka authentication
- Snowflake credentials
- Object Storage access keys
- OAuth client secrets
- JWT signing keys
- TLS certificates
- Kubernetes service credentials
- Third-party API tokens
- CI/CD deployment credentials

Managing secrets through application configuration files,
container images, or source code introduces significant security
risks including credential leakage, unauthorized access,
and compliance violations.

A centralized secrets management solution is required to securely
store, distribute, rotate, and audit sensitive information.

---

# Problem Statement

The platform requires a secrets management solution capable of:

- Secure secret storage
- Encryption at rest
- Encryption in transit
- Fine-grained access control
- Secret rotation
- Audit logging
- Dynamic secret generation
- Kubernetes integration
- CI/CD integration
- Multi-cloud compatibility

---

# Decision Drivers

The selected solution should provide:

- Strong encryption
- Centralized management
- Kubernetes integration
- Cloud portability
- Identity integration
- Automated rotation
- Policy enforcement
- High availability
- Enterprise adoption
- Compliance support

---

# Options Considered

## Option 1 — HashiCorp Vault

Advantages

- Enterprise-grade secret management
- Dynamic credentials
- Automatic secret rotation
- PKI support
- Encryption as a Service
- Kubernetes integration
- Extensive audit logging
- Multi-cloud support
- Strong policy engine

Disadvantages

- Additional operational complexity
- Infrastructure maintenance
- Learning curve

---

## Option 2 — Kubernetes Secrets Only

Advantages

- Native Kubernetes support
- Easy deployment
- No additional infrastructure

Disadvantages

- Limited secret lifecycle management
- Weak rotation capabilities
- Less comprehensive auditing
- Base64 encoding is not encryption
- Limited enterprise governance

---

## Option 3 — Cloud Provider Secret Managers

Examples:

- AWS Secrets Manager
- Azure Key Vault
- Google Secret Manager

Advantages

- Fully managed
- Strong cloud integration
- Automatic scaling

Disadvantages

- Vendor lock-in
- Reduced portability
- Multi-cloud complexity

---

## Option 4 — Environment Variables

Advantages

- Simple implementation
- Minimal tooling

Disadvantages

- High security risk
- Difficult rotation
- Poor auditability
- Exposure through process inspection
- Unsuitable for enterprise environments

---

# Decision

HashiCorp Vault is selected as the enterprise secrets management platform.

Vault provides centralized secret storage, encryption, fine-grained
access control, dynamic secret generation, automated rotation,
and comprehensive auditing while supporting cloud-agnostic
deployments.

Kubernetes workloads retrieve secrets securely at runtime rather
than embedding them in container images or application code.

---

# Architecture Impact

The secrets management platform stores and manages:

- Database credentials
- Kafka credentials
- Snowflake credentials
- API keys
- LLM provider tokens
- OAuth client secrets
- JWT signing keys
- TLS certificates
- Encryption keys
- Service account credentials
- CI/CD deployment credentials

---

# Integration Points

Secrets Management integrates with:

- Kubernetes
- Helm
- Terraform
- GitHub Actions
- Keycloak
- FastAPI
- LangGraph
- Apache Spark
- Apache Airflow
- PostgreSQL
- Kafka
- Snowflake
- Open Policy Agent
- Service Mesh

---

# Responsibilities

Secrets Management is responsible for:

- Secure secret storage
- Secret distribution
- Secret rotation
- Credential lifecycle
- Encryption key management
- Certificate management
- Access auditing
- Dynamic credential generation

Secrets Management is **not** responsible for:

- User authentication
- Authorization decisions
- Identity federation
- Infrastructure provisioning
- Business logic
- Application deployment

---

# Secret Categories

The platform manages the following categories:

## Infrastructure Secrets

- Kubernetes credentials
- Cloud credentials
- Storage credentials
- DNS credentials

---

## Database Secrets

- PostgreSQL
- Snowflake
- Metadata databases
- Vector database credentials

---

## AI Platform Secrets

- OpenAI API keys
- Azure OpenAI credentials
- Anthropic API keys
- Gemini API keys
- Hugging Face tokens
- Model registry credentials

---

## Application Secrets

- JWT signing keys
- OAuth client secrets
- Session encryption keys
- API Gateway credentials

---

## CI/CD Secrets

- GitHub deployment tokens
- Terraform credentials
- Helm deployment credentials
- Container registry authentication

---

# Secret Lifecycle

Secret creation

↓

Encryption

↓

Secure storage

↓

Runtime retrieval

↓

Application usage

↓

Automatic rotation

↓

Revocation

↓

Audit logging

---

# Relationship with Kubernetes

Kubernetes manages workloads.

Secrets Management provides secrets securely to those workloads.

Applications never contain hard-coded credentials.

Secrets are injected during runtime through secure integrations.

---

# Relationship with Terraform

Terraform provisions:

- Secret infrastructure
- Vault configuration
- Access policies

Terraform does **not** store production secrets within source code.

---

# Relationship with GitHub Actions

GitHub Actions retrieves deployment credentials securely during CI/CD.

Secrets are never stored in workflow files or repositories.

Deployment pipeline:

Developer Commit

↓

GitHub Actions

↓

Authentication

↓

Secret Retrieval

↓

Terraform

↓

Helm

↓

Deployment

---

# Relationship with Keycloak

Keycloak manages:

- User identity
- Authentication
- OAuth
- OpenID Connect

Secrets Management stores:

- Client secrets
- Signing keys
- Certificates
- Encryption material

Identity management and secret storage remain separate responsibilities.

---

# Relationship with Service Mesh

The Service Mesh secures communication between services using mTLS.

Secrets Management provides:

- Certificates
- Private keys
- Certificate rotation
- Trust anchors

---

# Consequences

## Positive

- Centralized secret management
- Strong encryption
- Improved compliance
- Reduced credential leakage
- Automated rotation
- Comprehensive audit logging
- Cloud portability
- Reduced operational risk

## Negative

- Additional infrastructure
- Operational complexity
- Availability requirements
- Initial configuration effort

---

# Risks

Potential risks include:

- Vault availability
- Misconfigured access policies
- Expired certificates
- Secret sprawl
- Inadequate rotation policies

Mitigation strategies:

- High-availability deployment
- Backup and disaster recovery
- Automated certificate renewal
- Least-privilege access
- Regular secret rotation
- Continuous auditing

---

# Alternatives Rejected

### Kubernetes Secrets

Rejected because enterprise secret lifecycle management,
dynamic credentials, auditing, and rotation capabilities
are insufficient for the platform's security requirements.

### Cloud-Native Secret Managers

Rejected because the Enterprise AI Platform targets
a cloud-agnostic deployment strategy.

### Environment Variables

Rejected because they increase the risk of credential exposure
and do not provide centralized governance.

---

# Future Considerations

Future enhancements may include:

- Dynamic database credentials
- Automatic certificate issuance
- Hardware Security Module (HSM) integration
- Bring Your Own Key (BYOK)
- Secret usage analytics
- AI-assisted secret rotation
- Zero Trust identity integration

---

# References

Related ADRs:

- ADR-006: Kubernetes
- ADR-012: OpenTelemetry
- ADR-024: Open Policy Agent
- ADR-025: Keycloak
- ADR-026: GitHub Actions
- ADR-027: Terraform
- ADR-028: Helm
- ADR-029: Service Mesh

Related Architecture Documents:

- Security Architecture
- Deployment Architecture
- DevOps Architecture
- Physical Architecture
- Quality Attributes
