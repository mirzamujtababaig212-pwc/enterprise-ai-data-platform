# ADR-024: Adopt Open Policy Agent (OPA) as the Enterprise Policy-as-Code Engine

**Status:** Accepted

**Date:** YYYY-MM-DD

**Decision Owners:** Enterprise Architecture Team

---

# Context

The Enterprise AI Platform comprises numerous distributed services,
including APIs, AI agents, data pipelines, Kubernetes workloads,
and administrative interfaces.

Authorization and governance requirements span multiple layers:

- API access control
- Kubernetes admission control
- AI model authorization
- Data access policies
- RAG document authorization
- Infrastructure governance
- Regulatory compliance

Embedding authorization logic directly within applications creates
duplication, inconsistent enforcement, and increased maintenance effort.

A centralized policy engine is required to externalize authorization
logic and enforce consistent governance across the platform.

---

# Problem Statement

The platform requires a policy engine capable of:

- Centralized authorization
- Policy-as-Code
- Fine-grained access control
- Attribute-Based Access Control (ABAC)
- Role-Based Access Control (RBAC)
- Kubernetes policy enforcement
- API authorization
- Auditability
- Cloud portability

---

# Decision Drivers

The selected policy engine should provide:

- Externalized authorization
- Declarative policy language
- Kubernetes integration
- API integration
- Vendor neutrality
- Enterprise scalability
- High performance
- CI/CD compatibility
- Strong community support

---

# Options Considered

## Option 1 — Open Policy Agent (OPA)

Advantages

- Open source
- Policy-as-Code
- Declarative Rego language
- Kubernetes native integration
- Envoy integration
- REST API authorization
- Fine-grained policy evaluation
- Cloud portable
- Large CNCF ecosystem

Disadvantages

- Learning curve for Rego
- Additional operational component
- Policy lifecycle management required

---

## Option 2 — Custom Authorization Framework

Advantages

- Fully customized
- Tailored to business requirements

Disadvantages

- High maintenance effort
- Duplicate authorization logic
- Difficult governance
- Limited reusability

---

## Option 3 — Application-Level Authorization

Advantages

- Simple implementation
- Minimal infrastructure

Disadvantages

- Authorization duplicated across services
- Inconsistent enforcement
- Difficult auditing
- Poor maintainability

---

## Option 4 — Cloud Provider IAM Policies

Advantages

- Managed service
- Native cloud integration

Disadvantages

- Vendor lock-in
- Limited multi-cloud portability
- Inconsistent cross-platform policies

---

# Decision

The Enterprise AI Platform adopts Open Policy Agent (OPA)
as the centralized Policy-as-Code engine.

OPA evaluates authorization requests independently from
application code.

Applications remain responsible for authentication,
while authorization decisions are delegated to OPA.

Policies are managed as version-controlled code and
deployed through the platform's CI/CD pipeline.

---

# Architecture Impact

OPA governs authorization for:

- REST APIs
- FastAPI services
- Kubernetes workloads
- AI Agents
- LangGraph workflows
- RAG pipelines
- Vector database access
- Data platform services
- Administrative portals
- Internal microservices

---

# Integration Points

OPA integrates with:

- Keycloak
- API Gateway
- FastAPI
- Kubernetes
- Envoy / Service Mesh
- GitHub Actions
- OpenTelemetry
- Prometheus
- Audit Logging
- CI/CD pipelines

---

# Policy Types

OPA evaluates policies for:

### API Authorization

- Endpoint access
- HTTP methods
- Resource ownership
- Tenant isolation

---

### Kubernetes Governance

- Admission control
- Pod security policies
- Resource quotas
- Namespace isolation

---

### AI Governance

- Model access
- Prompt restrictions
- Agent permissions
- RAG authorization
- Document visibility

---

### Data Governance

- Dataset access
- Column-level permissions
- Row-level filtering
- Sensitive data masking

---

### Infrastructure Governance

- Terraform validation
- Deployment approvals
- Environment promotion
- Compliance checks

---

# Responsibilities

OPA is responsible for:

- Authorization decisions
- Policy evaluation
- Policy enforcement
- Compliance validation
- Governance rules
- Fine-grained permissions
- Centralized policy management

OPA is not responsible for:

- User authentication
- Identity management
- Secret storage
- API routing
- Workflow orchestration

---

# Relationship with Keycloak

Keycloak and OPA solve different security problems.

### Keycloak

Responsible for:

- Authentication
- Identity management
- Single Sign-On (SSO)
- User federation
- Token issuance
- Identity lifecycle

---

### Open Policy Agent

Responsible for:

- Authorization
- Policy evaluation
- Access control
- Compliance rules
- Resource permissions
- Governance policies

Together they implement:

Authentication → Authorization → Resource Access

---

# Consequences

## Positive

- Centralized authorization
- Consistent policy enforcement
- Reduced code duplication
- Improved governance
- Better auditability
- Easier compliance
- Vendor neutrality
- Policy versioning

## Negative

- Additional infrastructure
- Rego learning curve
- Policy management lifecycle
- Operational overhead

---

# Risks

Potential risks include:

- Complex policy rules
- Policy conflicts
- Performance overhead
- Incorrect authorization
- Policy sprawl

Mitigation strategies:

- Policy testing
- Version-controlled policies
- Automated CI validation
- Policy review process
- Performance benchmarking
- Policy documentation

---

# Alternatives Rejected

### Custom Authorization Framework

Rejected because maintaining authorization logic across multiple
services increases operational complexity and reduces consistency.

### Application-Level Authorization

Rejected because embedding authorization logic within each service
creates duplication and complicates governance.

### Cloud Provider IAM

Rejected because the Enterprise AI Platform targets a cloud-agnostic
architecture rather than a provider-specific implementation.

---

# Future Considerations

Future enhancements may include:

- OPA Bundle Server
- Dynamic policy distribution
- Policy simulation
- AI-assisted policy generation
- Fine-grained AI governance
- Cross-cluster policy federation
- Confidential computing policy enforcement

---

# References

Related ADRs:

- ADR-015: FastAPI
- ADR-021: Event-Driven Architecture
- ADR-022: Microservices
- ADR-023: Domain-Driven Design
- ADR-025: Keycloak
- ADR-029: Service Mesh

Related Architecture Documents:

- Security Architecture
- Logical Architecture
- Physical Architecture
- Deployment Architecture
- Quality Attributes
- Governance Architecture
