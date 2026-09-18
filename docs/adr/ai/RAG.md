# ADR-018: Adopt Retrieval-Augmented Generation (RAG) as the Enterprise Knowledge Retrieval Architecture

**Status:** Accepted

**Date:** YYYY-MM-DD

**Decision Owners:** Enterprise Architecture Team

---

# Context

The Enterprise AI Platform enables conversational AI, enterprise search,
knowledge assistants, document intelligence, and domain-specific AI
applications.

Enterprise knowledge resides across multiple repositories, including:

- Enterprise documentation
- Policies and procedures
- Knowledge bases
- Wikis
- Source code repositories
- Data catalogs
- APIs
- Structured databases
- Data lake assets
- Business documents

Traditional Large Language Models (LLMs) rely solely on pre-trained
knowledge and cannot access proprietary or continuously changing
enterprise information without additional mechanisms.

The platform requires an architecture that enables LLMs to retrieve
relevant enterprise knowledge at inference time while maintaining
security, governance, and response quality.

---

# Problem Statement

The platform requires a knowledge retrieval architecture capable of:

- Enterprise document retrieval
- Semantic search
- Context injection
- Multi-source knowledge integration
- Low-latency retrieval
- Secure access control
- Source attribution
- Real-time knowledge updates
- Reduced hallucinations
- Enterprise scalability

---

# Decision Drivers

The selected architecture should provide:

- Accurate contextual retrieval
- Improved response quality
- Cloud portability
- Open architecture
- Vendor independence
- Integration with multiple LLM providers
- Metadata filtering
- Fine-grained security
- Observability
- Enterprise governance

---

# Options Considered

## Option 1 — Retrieval-Augmented Generation (RAG)

Advantages

- Uses current enterprise knowledge
- Reduces hallucinations
- Improves response accuracy
- Supports proprietary data
- Model independent
- Scalable architecture
- Easier governance
- Supports source attribution

Disadvantages

- Additional infrastructure
- Retrieval latency
- Embedding management
- Knowledge indexing required

---

## Option 2 — LLM Fine-Tuning

Advantages

- Domain adaptation
- No retrieval required during inference

Disadvantages

- Expensive retraining
- Knowledge becomes outdated
- Limited explainability
- Difficult governance
- High operational cost

---

## Option 3 — Prompt Engineering Only

Advantages

- Simple implementation
- Minimal infrastructure

Disadvantages

- Limited enterprise knowledge
- High hallucination risk
- Poor scalability
- No knowledge management

---

## Option 4 — Keyword Search

Advantages

- Simple
- Mature technology

Disadvantages

- Limited semantic understanding
- Poor ranking quality
- Inferior user experience
- No contextual reasoning

---

# Decision

Retrieval-Augmented Generation (RAG) is adopted as the standard
enterprise knowledge retrieval architecture.

RAG combines semantic retrieval with Large Language Models to provide
accurate, explainable, and context-aware responses while allowing the
platform to use continuously updated enterprise knowledge without
retraining foundation models.

---

# Architecture Impact

The RAG architecture consists of:

- Document ingestion
- Document parsing
- Text chunking
- Embedding generation
- Vector indexing
- Metadata management
- Semantic retrieval
- Prompt augmentation
- LLM inference
- Response generation

---

# Retrieval Strategy

The platform supports composable retrieval strategies behind the
`Retriever` contract.

The current retrieval architecture includes:

- Semantic retrieval
- Lexical retrieval
- Hybrid semantic and lexical retrieval
- Post-retrieval reranking through `RerankingRetriever`
- Retrieval quality evaluation and experiment comparison

Hybrid retrieval combines semantic and lexical candidate sets using
weighted Reciprocal Rank Fusion (RRF). It remains the default retrieval
strategy for the current retrieval-quality evaluation path.

The PostgreSQL hybrid retrieval path has a deterministic regression
benchmark covering 11 labeled retrieval cases. The current hybrid
baseline achieves Recall@5 of 1.0000, Precision@5 of 0.4727, MRR of
0.9394, and nDCG@5 of 0.9530, with all relevant items present in the
hybrid candidate sets.

Experiments varying the semantic and lexical RRF weights across the
tested ranges did not materially change the benchmark ordering or
aggregate quality metrics. No change to the default RRF weighting is
therefore justified by the current evidence.

Cross-encoder reranking is implemented as an experimental,
post-retrieval capability. It is intentionally separated from the
Enterprise LLM Gateway and does not change the core `Retriever` contract.

Controlled CrossEncoder experiments improved MRR in the current
benchmark but introduced materially higher inference latency and
benchmark-dependent changes in nDCG and recall. Candidate-depth
experiments also showed that increasing the reranking candidate set does
not guarantee improved retrieval quality.

On the enterprise-policy benchmark, which contains 20 graded retrieval
cases, the Hybrid RRF baseline achieved Recall@5 of 0.9750, Precision@5
of 0.4500, MRR of 1.0000, and nDCG@5 of 0.9380. Replacing the Hybrid
ranking with CrossEncoder reranking preserved recall and precision but
reduced MRR to 0.8417 and nDCG@5 to 0.7736. Per-query diagnostics showed
that the CrossEncoder can rank broadly related policy content above
higher-grade business-specific results, indicating a ranking-objective
mismatch rather than a candidate-coverage problem.

A further selective-reranking experiment using CrossEncoder score
margins did not identify a stable threshold that consistently improves
the benchmark. CrossEncoder score margins are model scores rather than
calibrated confidence probabilities and can be high even when the
top-ranked item is not labeled relevant.

Normalized Hybrid-score and CrossEncoder-score fusion was also evaluated
on the enterprise-policy benchmark. Introducing even a 10% CrossEncoder
contribution reduced MRR from 1.0000 to 0.9750 and nDCG@5 from 0.9380 to
0.9036. Higher CrossEncoder contributions produced further degradation.
No score-fusion policy is therefore adopted.

Hybrid semantic and lexical retrieval using RRF therefore remains the
default retrieval strategy. CrossEncoder reranking remains an optional
experimental capability and is not enabled globally by default.

Retrieval experiments are compared descriptively rather than treated as
release regressions when the retrieval implementation itself changes.
Release compatibility remains governed by the existing evaluation
lineage and release-comparison rules.

Promotion of reranking to a default retrieval strategy requires broader
and more representative enterprise datasets and evaluation across
ranking quality, latency, cost, and operational behavior.

---

# Core Components

The RAG architecture includes:

- LangGraph
- FastAPI
- Qdrant
- Enterprise LLM Gateway
- Embedding Models
- PostgreSQL
- Object Storage
- OpenTelemetry
- Prometheus & Grafana

---

# End-to-End Workflow

The enterprise RAG workflow consists of:

1. Documents are ingested from enterprise sources.
2. Documents are parsed and normalized.
3. Text is divided into optimized chunks.
4. Embeddings are generated.
5. Embeddings are stored in Qdrant.
6. Metadata is indexed.
7. User submits a query.
8. Query embedding is generated.
9. Similar vectors are retrieved.
10. Retrieved context is validated.
11. LangGraph assembles the prompt.
12. The selected LLM generates a response.
13. Source references are attached.
14. Observability data is captured.
15. Response is returned to the client.

---

# Knowledge Sources

Supported enterprise sources include:

- SharePoint
- Confluence
- GitHub
- Data Catalogs
- APIs
- PostgreSQL
- Snowflake
- Delta Lake
- PDF documents
- Office documents
- HTML content
- Internal portals

---

# Responsibilities

The RAG architecture is responsible for:

- Knowledge retrieval
- Semantic search
- Context generation
- Source attribution
- Enterprise document access
- Prompt enrichment

The RAG architecture is not responsible for:

- Model training
- Workflow scheduling
- Identity management
- Infrastructure provisioning
- Feature engineering
- Data ingestion orchestration

---

# Relationship with Other Components

### LangGraph

Responsible for:

- Agent orchestration
- Prompt assembly
- Tool execution
- Multi-step reasoning

### Qdrant

Responsible for:

- Vector storage
- Similarity search
- Metadata filtering
- Semantic retrieval

### FastAPI

Responsible for:

- API endpoints
- Request validation
- Client communication

### Enterprise LLM Gateway

Responsible for:

- Model routing
- Provider abstraction
- Failover
- Cost optimization

Together these components implement the enterprise RAG pipeline.

---

# Security Considerations

The RAG architecture implements:

- Role-Based Access Control (RBAC)
- Document-level authorization
- Metadata filtering
- Tenant isolation
- Encryption in transit
- Encryption at rest
- Audit logging
- Prompt sanitization
- Sensitive data masking

Only documents that the requesting user is authorized to access are
eligible for retrieval.

---

# Observability

The platform captures:

- Retrieval latency
- Embedding latency
- LLM latency
- Retrieved document count
- Token usage
- Prompt size
- Response quality metrics
- Retrieval accuracy
- Citation coverage
- Error rates

Telemetry is exported using OpenTelemetry and monitored through
Prometheus and Grafana.

---

# Consequences

## Positive

- Reduced hallucinations
- Current enterprise knowledge
- Explainable responses
- Improved trust
- Better governance
- Model independence
- Scalable architecture

## Negative

- Additional operational components
- Retrieval latency
- Embedding lifecycle management
- Knowledge indexing overhead

---

# Risks

Potential risks include:

- Poor chunking strategy
- Low-quality embeddings
- Missing metadata
- Unauthorized retrieval
- Large prompt sizes
- Context window limitations

Mitigation strategies:

- Standardized chunking policies
- Embedding quality validation
- Metadata governance
- RBAC enforcement
- Prompt optimization
- Retrieval evaluation
- Continuous monitoring

---

# Alternatives Rejected

### LLM Fine-Tuning

Rejected because enterprise knowledge changes frequently and retraining
foundation models is expensive and operationally complex.

### Prompt Engineering Only

Rejected because prompts alone cannot provide dynamic access to
enterprise knowledge.

### Keyword Search

Rejected because semantic retrieval provides significantly better
relevance and user experience for enterprise AI applications.

---

# Future Considerations

Potential future enhancements include:

- Graph RAG
- Multi-modal RAG
- Agentic RAG
- Adaptive retrieval strategies
- Knowledge graph integration
- Multi-vector indexing
- Personalized retrieval
- Continuous retrieval evaluation
- Broader enterprise-scale reranking evaluation

---

# References

Related ADRs:

- ADR-008: FastAPI
- ADR-012: OpenTelemetry
- ADR-013: Prometheus & Grafana
- ADR-016: LangGraph
- ADR-017: Qdrant

Related Architecture Documents:

- AI Architecture
- Logical Architecture
- Physical Architecture
- Security Architecture
- Observability Architecture
- Quality Attributes
