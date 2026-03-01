"""Exception types for the knowledge graph library."""

from __future__ import annotations


class InvalidNodeTypeError(ValueError):
    """Raised when an invalid node type is used."""

    def __init__(self, node_type: str, valid_types: frozenset[str]) -> None:
        super().__init__(
            f"Invalid node type '{node_type}'. Valid types: {sorted(valid_types)}"
        )


class InvalidEdgeTypeError(ValueError):
    """Raised when an invalid edge type is used."""

    def __init__(self, edge_type: str, valid_types: frozenset[str]) -> None:
        super().__init__(
            f"Invalid edge type '{edge_type}'. Valid types: {sorted(valid_types)}"
        )


class NodeNotFoundError(KeyError):
    """Raised when a node is not found in the graph."""

    def __init__(self, node_id: str) -> None:
        super().__init__(f"Node not found: '{node_id}'")
        self.node_id = node_id
