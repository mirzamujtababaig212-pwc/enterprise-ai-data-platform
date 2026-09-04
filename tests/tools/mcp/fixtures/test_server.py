from mcp.server.fastmcp import FastMCP


mcp = FastMCP("Enterprise AI Platform Test Server")


@mcp.tool()
def search_documents(
    query: str,
) -> dict:
    """
    Search a deterministic test document collection.
    """

    documents = [
        {
            "id": "document-1",
            "content": "Enterprise AI platform architecture.",
        },
        {
            "id": "document-2",
            "content": "Enterprise data engineering standards.",
        },
        {
            "id": "document-3",
            "content": "AI governance and security policies.",
        },
    ]

    query_lower = query.lower()

    matches = [document for document in documents if query_lower in document["content"].lower()]

    return {
        "query": query,
        "results": matches,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
