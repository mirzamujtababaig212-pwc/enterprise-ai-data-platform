Status: Proposed
Date: August 2026
Deciders: Enterprise Architecture Review Board (ARB), Lead AI/ML Infrastructure Engineer, Lead Systems Architect
Technical Domain: Compute & Compute Kernel (Layer 1 of Enterprise AI OS)
1. Context and Problem Statement
Our current Enterprise AI Platform relies on standard Kubernetes orchestration (via EKS/AKS/GKE) to deploy AI microservices and model serving pods. 
While Kubernetes effectively handles basic pod scheduling and horizontal auto-scaling (HPA), standard container orchestrators are oblivious to deep AI compute primitives—specifically Key-Value (KV) 
cache state, tensor parallel memory allocation, and hardware heterogeneous execution constraints.  
As our platform scales to support multi-agent systems, stateful long-context RAG, and multi-modal edge execution, we face three critical architectural bottlenecks:
VRAM Memory Fragmentation & Out-Of-Memory (OOM) Errors: Naïve GPU allocation allocates static VRAM blocks to concurrent requests, leading to external/internal fragmentation and 
poor batching utilization.
High TTFT (Time-To-First-Token) & Latency Overhead: Re-computing KV-caches for repeated prompt contexts (e.g., system prompts, RAG context windows, agent system state) incurs prohibitive compute latency 
and GPU expenditure.
Edge-to-Cloud Heterogeneity & Execution Disconnect: Devices at the edge (on-device SLMs, edge servers, remote gateways) cannot run large foundation models locally, 
while relying 100% on cloud GPUs creates unacceptable latency, bandwidth costs, and offline fragility.
2. Decision Outcome
We will upgrade our compute infrastructure by implementing a Two-Tier AI Compute Kernel Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CONTROL PLANE / SCHEDULER                             │
│       (Knative / Ray Serve + Dynamic Edge-Cloud Task Router)                │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
┌───────────────────────────────┐                     ┌──────────────────────┐
│     CLOUD COMPUTE KERNEL      │                     │ EDGE COMPUTE KERNEL  │
│  (Distributed GPU/vLLM/SGLang)│                     │ (vLLM Edge / Llama.cpp)│
├───────────────────────────────┤                     ├──────────────────────┤
│ • PagedAttention / RadixTree  │                     │ • Quantized (GGUF)   │
│ • Chunked Prefill & Decoding  │  Edge-Cloud Sync    │ • Local CPU/NPU/vGPU │
│ • Tensor Parallel (NCCL)      │◄───────────────────►│ • Privacy / Offline  │
│ • Dynamic KV-Cache Eviction   │    (gRPC / MCP)     │   Fallback Rules     │
└───────────────────────────────┘                     └──────────────────────┘
Key Architectural Components Adopted:
PagedAttention & KV-Cache Management (vLLM / SGLang Engine):
Integrate PagedAttention memory management within model serving engines, treating GPU VRAM like virtual memory pages in a traditional OS to eliminate memory fragmentation.
Implement RadixTree Automatic Prefix Caching (APC) to share KV-caches across concurrent requests sharing identical system prompts or enterprise context documents.
Adopt Chunked Prefill & Piggybacked Decoding to run compute-bound prefill and memory-bound decoding concurrently, reducing tail latency ($p99$).
Heterogeneous Edge-Cloud Workload Scheduling:
Deploy a Tiered Inference Pipeline: Small Language Models (SLMs e.g., Phi-4, Llama 3.2 1B-3B) run locally on edge hardware for fast filtering, pre-processing, and fallback execution.
Route high-complexity reasoning, agentic tool execution, and heavy embeddings dynamically to the centralized Cloud GPU cluster via secure gRPC/mTLS channels managed by Istio.  
Kubernetes Device Plugin & Compute Virtualization:
Integrate NVIDIA GPU Operator with MIG (Multi-Instance GPU) and Time-Slicing for fractional GPU sharing across lighter workloads.
Utilize Ray on Kubernetes (KubeRay) for auto-scaling distributed GPU workloads (Tensor/Pipeline Parallelism across multi-node clusters).
3. Considered Alternatives Option Rationale for Rejection
Option 1: Native Kubernetes HPA + Standard HuggingFace TGILacks dynamic KV-cache prefix sharing across requests. 
Higher VRAM utilization cost and significantly higher latency under concurrent agent loads.
Option 2: Pure Cloud-Only Inference (No Edge Support)
High network bandwidth costs, latency unacceptable for real-time edge processing, zero offline resilience, and non-compliance with strict local data residency rules.
Option 3: Custom In-House CUDA Memory Allocator 
Massive ongoing software engineering overhead; reinvents open-source community standards (vLLM, SGLang) without added business benefit.
4. Consequences & Trade-offs
Positive Consequences (Benefits) 3x–5x Improvement in GPU Throughput: 
PagedAttention and prefix caching increase maximum request concurrency per GPU node without OOM errors.
Significant Cost Reduction: Reduced token processing cost through prompt caching and automated offloading to edge/SLM tiers.
Zero-Downtime Multi-Cloud & Edge Portability: 
Maintains enterprise cloud-agnostic objectives (AWS/Azure/GCP/Edge) using standard containerized Ray/Kubernetes execution.  
Sub-100ms Initial Token Latency: Drastic reduction in TTFT for long-context RAG pipelines.
Negative Consequences & Mitigations Increased System Complexity: Managing stateful KV-caches and multi-tier Edge-Cloud synchronization introduces operational complexity.
Mitigation: Abstract compute kernel orchestration behind the Multi-LLM Gateway and automate container deployments via Helm and Terraform.  
Network Instability at Edge: Potential dropped connections between Edge nodes and Cloud clusters.
Mitigation: Edge execution engines must contain self-contained fallback logic (local SLM execution) when offline.
