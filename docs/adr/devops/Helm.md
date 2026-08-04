# ADR-028: Adopt Helm as the Enterprise Kubernetes Package Management Platform

**Status:** Accepted

**Date:** YYYY-MM-DD

**Decision Owners:** Enterprise Architecture Team

---

# Context

The Enterprise AI Platform consists of numerous containerized services
deployed onto Kubernetes, including:

- FastAPI microservices
- LangGraph agents
- RAG services
- Apache Spark components
- Airflow
- dbt services
- Kafka
- Qdrant
- PostgreSQL
- Monitoring stack
- Identity services

Managing Kubernetes manifests individually across multiple environments
introduces configuration drift, deployment inconsistencies, duplicated
configuration, and operational complexity.

A standardized deployment mechanism is required for packaging,
configuration, versioning, and lifecycle management of Kubernetes
applications.

---

# Problem Statement

The platform requires a Kubernetes package management solution capable of:

- Packaging Kubernetes applications
- Environment-specific configuration
- Versioned application releases
- Rollback support
- Dependency management
- Template-based deployments
- CI/CD integration
- Kubernetes-native deployment
- Enterprise scalability
- Release lifecycle management

---

# Decision Drivers

The selected platform should provide:

- Kubernetes-native deployment
- Declarative configuration
- Reusable templates
- Environment parameterization
- Easy upgrades
- Rollback capability
- GitOps compatibility
- CI/CD integration
- Enterprise adoption
- Strong community support

---

# Options Considered

## Option 1 — Helm

Advantages

- Kubernetes-native
- Reusable templates
- Parameterized deployments
- Versioned releases
- Rollback support
- Large community
- Mature ecosystem
- Excellent CI/CD integration
- Strong Kubernetes adoption

Disadvantages

- Template syntax learning curve
- Chart maintenance
- Values file complexity

---

## Option 2 — Raw Kubernetes YAML

Advantages

- Simple
- Native Kubernetes resources
- No additional tooling

Disadvantages

- Configuration duplication
- Difficult environment management
- Limited reusability
- Increased maintenance effort

---

## Option 3 — Kustomize

Advantages

- Native Kubernetes support
- Overlay model
- Simple customization

Disadvantages

- Less suitable for reusable application packaging
- Limited dependency management
- Smaller chart ecosystem

---

## Option 4 — Operators Only

Advantages

- Automation
- Advanced lifecycle management
- Domain-specific automation

Disadvantages

- High development effort
- Complex implementation
- Not appropriate for all workloads

---

# Decision

Helm is selected as the enterprise Kubernetes package management platform.

Helm provides reusable application packaging, parameterized deployments,
version-controlled releases, and simplified lifecycle management across
development, testing, staging, and production environments.

---

# Architecture Impact

Helm deploys and manages:

- FastAPI services
- LangGraph agents
- RAG services
- Spark applications
- Airflow
- dbt services
- Kafka components
- Qdrant
- PostgreSQL
- Monitoring stack
- Keycloak
- API Gateway
- Internal platform services

---

# Integration Points

Helm integrates with:

- Kubernetes
- Terraform
- GitHub Actions
- Docker
- FastAPI
- Apache Spark
- Apache Airflow
- LangGraph
- Qdrant
- Prometheus
- Grafana
- OpenTelemetry
- Keycloak

---

# Responsibilities

Helm is responsible for:

- Kubernetes application deployment
- Release management
- Application configuration
- Environment-specific values
- Rolling updates
- Rollback management
- Kubernetes templating
- Dependency management

Helm is **not** responsible for:

- Infrastructure provisioning
- CI/CD orchestration
- Container image creation
- Source code management
- Runtime monitoring
- Identity management

---

# Deployment Workflow

Developer Commit

↓

GitHub Actions

↓

Container Image Build

↓

Container Registry

↓

Terraform Provisioning (if required)

↓

Helm Upgrade / Install

↓

Kubernetes Cluster

↓

Application Running

---

# Chart Structure

A standard Helm chart contains:

- Chart.yaml
- values.yaml
- templates/
- charts/
- NOTES.txt

Environment-specific values include:

- Development
- Testing
- Staging
- Production

---

# Release Strategy

Helm supports:

- Versioned releases
- Rolling updates
- Canary deployments
- Blue-Green deployments
- Rollbacks
- Incremental upgrades

---

# Relationship with Terraform

Terraform and Helm solve different problems.

Terraform provisions:

- Kubernetes clusters
- Networking
- Storage
- Cloud infrastructure
- IAM resources

Helm manages:

- Kubernetes workloads
- Deployments
- Services
- ConfigMaps
- Secrets references
- Ingress resources

Terraform creates the platform.

Helm deploys applications onto the platform.

---

# Relationship with GitHub Actions

GitHub Actions automates software delivery.

Typical deployment pipeline:

Developer Commit

↓

GitHub Actions

↓

Unit Tests

↓

Security Scanning

↓

Docker Build

↓

Container Registry

↓

Terraform (Infrastructure)

↓

Helm (Application Deployment)

↓

Production

---

# Consequences

## Positive

- Standardized Kubernetes deployments
- Version-controlled releases
- Simplified upgrades
- Easier rollback
- Reduced configuration duplication
- Environment consistency
- Reusable deployment templates

## Negative

- Chart maintenance
- Template complexity
- Values management across environments
- Helm version compatibility

---

# Risks

Potential risks include:

- Incorrect values files
- Configuration drift
- Failed upgrades
- Secret misconfiguration
- Chart dependency conflicts

Mitigation strategies:

- Version-controlled charts
- Automated Helm linting
- CI/CD validation
- GitOps workflows
- Standardized chart templates
- Release testing

---

# Alternatives Rejected

### Raw Kubernetes YAML

Rejected because maintaining hundreds of individual manifests would
increase duplication, reduce maintainability, and complicate
multi-environment deployments.

### Kustomize

Rejected because Helm provides stronger application packaging,
dependency management, and release lifecycle capabilities.

### Operators

Rejected because operators are appropriate for specialized workloads
rather than general application deployment.

---

# Future Considerations

Future enhancements may include:

- GitOps with Argo CD or Flux
- Helm OCI Registry
- Progressive delivery
- Automated rollback
- Policy validation with OPA
- Helm chart security scanning
- Multi-cluster deployments

---

# References

Related ADRs:

- ADR-006: Kubernetes
- ADR-024: Open Policy Agent
- ADR-026: GitHub Actions
- ADR-027: Terraform
- ADR-029: Service Mesh
- ADR-030: Secrets Management

Related Architecture Documents:

- Physical Architecture
- Deployment Architecture
- DevOps Architecture
- Security Architecture
- Quality Attributes
