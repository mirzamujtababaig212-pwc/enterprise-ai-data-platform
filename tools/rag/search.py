from __future__ import annotations

from typing import Any

from rag.retrieval.retriever import SemanticRetriever
from tools.models import ToolDefinition


class RAGSearchTool:
    """Tool exposing semantic retrieval to agents."""

    def __init__(self, retriever: SemanticRetriever) -> None:
        self._retriever = retriever

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="rag.search",
            description=(
                "Search the enterprise knowledge base for relevant "
                "documents and return the most relevant sources."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query.",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5,
                        "description": "Maximum number of sources to return.",
                    },
                    "min_score": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": (
                            "Optional minimum relevance score. Results below this "
                            "threshold are excluded."
                        ),
                    },
                    "metadata_filter": {
                        "type": "object",
                        "additionalProperties": True,
                        "description": (
                            "Optional equality-based metadata filters. Multiple fields "
                            "are combined with AND semantics."
                        ),
                    },
                },
                "required": ["query"],
            },
            metadata={
                "category": "retrieval",
                "read_only": True,
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("rag.search query must be a non-empty string.")

        top_k = arguments.get("top_k", 5)
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError("rag.search top_k must be an integer.")

        if top_k < 1 or top_k > 10:
            raise ValueError("rag.search top_k must be between 1 and 10.")

        min_score = arguments.get("min_score")

        if min_score is not None:
            if isinstance(min_score, bool) or not isinstance(min_score, (int, float)):
                raise TypeError("rag.search min_score must be a number.")

            if min_score < 0.0 or min_score > 1.0:
                raise ValueError("rag.search min_score must be between 0.0 and 1.0.")

        metadata_filter = arguments.get("metadata_filter")

        if metadata_filter is not None and not isinstance(metadata_filter, dict):
            raise TypeError("rag.search metadata_filter must be an object.")

        results = await self._retriever.retrieve(
            query=query.strip(),
            top_k=top_k,
            min_score=min_score,
            metadata_filter=metadata_filter,
        )

        return {
            "query": query.strip(),
            "results": [
                {
                    "chunk_id": result.chunk.id,
                    "document_id": result.chunk.document_id,
                    "content": result.chunk.content,
                    "score": result.score,
                    "metadata": result.chunk.metadata,
                }
                for result in results
            ],
            "retrieved_count": len(results),
        }
