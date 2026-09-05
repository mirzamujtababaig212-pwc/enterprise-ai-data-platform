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

        results = await self._retriever.retrieve(
            query=query.strip(),
            top_k=top_k,
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
