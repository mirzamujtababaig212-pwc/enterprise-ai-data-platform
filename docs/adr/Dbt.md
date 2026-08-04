# ADR-011: Adopt dbt as the Enterprise SQL Transformation Framework

**Status:** Accepted

**Date:** YYYY-MM-DD

**Decision Owners:** Enterprise Architecture Team

---

# Context

The Enterprise AI Platform ingests data into Delta Lake and Snowflake from
multiple operational systems through Kafka, Spark, and Airflow.

After ingestion, business-ready datasets require standardized
transformations including:

- Data cleansing
- Business rules
- Data modeling
- Dimensional modeling
- Metric definitions
- Testing
- Documentation
- Lineage

The transformation framework must integrate with the platform's
engineering standards while supporting version control, testing,
documentation, and CI/CD.

---

# Problem Statement

The platform requires a transformation framework capable of:

- SQL-based transformations
- Modular development
- Version-controlled pipelines
- Automated testing
- Documentation generation
- Data lineage visualization
- Integration with GitHub
- Integration with Airflow
- Integration with Snowflake
- Enterprise scalability

---

# Decision Drivers

The selected framework should provide:

- SQL-first development
- Software engineering practices
- Automated testing
- Documentation
- Data lineage
- Reusable models
- Cloud portability
- CI/CD integration
- Strong community support
- Enterprise adoption

---

# Options Considered

## Option 1 — dbt

Advantages

- SQL-first development
- Modular models
- Version control
- Built-in testing
- Automatic documentation
- Lineage visualization
- Jinja templating
- CI/CD friendly
- Mature ecosystem

Disadvantages

- SQL-centric
- Learning curve for Jinja
- Requires disciplined project structure

---

## Option 2 — Spark SQL Scripts

Advantages

- Flexible
- Native Spark integration
- Distributed execution

Disadvantages

- Limited documentation
- Limited testing framework
- Difficult dependency management
- Reduced lineage visibility

---

## Option 3 — Stored Procedures

Advantages

- Database-native
- Good performance

Disadvantages

- Vendor lock-in
- Difficult version control
- Poor portability
- Reduced maintainability

---

## Option 4 — Custom Python ETL

Advantages

- Flexible
- General-purpose programming

Disadvantages

- Reinvents transformation framework
- Increased maintenance
- Less accessible for analytics teams
- Limited SQL optimization

---

# Decision

dbt is selected as the enterprise SQL transformation framework.

dbt provides modular SQL development, automated testing,
documentation, lineage visualization, and software engineering
best practices for analytics engineering.

Spark remains responsible for distributed computation,
while dbt is responsible for business-oriented SQL
transformations.

---

# Architecture Impact

dbt is responsible for:

- Silver-to-Gold transformations
- Dimensional modeling
- Star schemas
- Business logic
- Metric definitions
- Data quality tests
- Documentation
- Lineage generation
- Semantic layer preparation

---

# Integration Points

dbt integrates with:

- Snowflake
- Delta Lake
- Apache Airflow
- GitHub Actions
- CI/CD pipelines
- Data Catalog
- Business Intelligence tools

---

# Responsibilities

dbt is responsible for:

- SQL transformations
- Incremental models
- Testing
- Documentation
- Lineage
- Model dependency management

dbt is not responsible for:

- Data ingestion
- Event streaming
- Distributed processing
- Workflow scheduling
- Machine learning training

---

# Relationship with Spark

Spark and dbt solve different problems.

Spark

- Large-scale distributed processing
- Streaming ETL
- Feature engineering
- CDC processing
- Heavy transformations

dbt

- SQL transformations
- Business models
- Star schemas
- Data marts
- Testing
- Documentation
- Lineage

Together they provide a layered enterprise data engineering architecture.

---

# Consequences

## Positive

- Improved maintainability
- Standardized SQL development
- Built-in testing
- Better documentation
- Automatic lineage
- Easier onboarding
- Reusable transformations

## Negative

- Additional tooling
- SQL discipline required
- Build dependency management
- Learning curve

---

# Risks

Potential risks include:

- Large monolithic models
- Poor naming standards
- Long-running transformations
- Inconsistent testing

Mitigation strategies:

- Modular model design
- Coding standards
- Incremental models
- CI validation
- Mandatory testing
- Documentation reviews

---

# Alternatives Rejected

### Spark SQL Only

Rejected because Spark focuses on distributed processing rather than
analytics engineering lifecycle management.

### Stored Procedures

Rejected because they introduce vendor lock-in and reduce
maintainability.

### Custom Python Framework

Rejected because dbt already provides a mature,
well-supported transformation framework.

---

# Future Considerations

Future enhancements may include:

- dbt Mesh
- dbt Semantic Layer
- dbt Metrics
- Data Contracts
- AI-assisted SQL generation
- OpenMetadata integration

---

# References

Related ADRs:

- ADR-002: Apache Spark
- ADR-003: Delta Lake
- ADR-004: Snowflake
- ADR-005: Apache Airflow

Related Architecture Documents:

- Logical Architecture
- Data Platform Architecture
- Quality Attributes
- Physical Architecture
