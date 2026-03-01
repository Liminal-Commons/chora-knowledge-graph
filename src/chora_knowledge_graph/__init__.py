"""chora-knowledge-graph — SQLite-backed knowledge graph with FTS5 + sqlite-vec.

    from chora_knowledge_graph import KnowledgeGraph

    g = KnowledgeGraph(
        valid_node_types={"Term", "Axiom", "Pattern"},
        valid_edge_types={"governed_by", "replaces"},
    )

    g.add_node("Term", "my-term", name="My Term", domain="example")
    g.add_node("Pattern", "my-pattern", name="My Pattern")
    g.add_edge("governed_by", "my-term", "my-pattern")

    result = g.query("my-term")
"""

from chora_knowledge_graph.graph import KnowledgeGraph
from chora_knowledge_graph.types import (
    InvalidEdgeTypeError,
    InvalidNodeTypeError,
    NodeNotFoundError,
)

__all__ = [
    "KnowledgeGraph",
    "InvalidNodeTypeError",
    "InvalidEdgeTypeError",
    "NodeNotFoundError",
]
