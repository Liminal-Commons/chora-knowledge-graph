"""Tests for KnowledgeGraph — configurable types, FTS5, sqlite-vec, persistence.

Validates configurable type registries, type validation, rejection of
invalid types, and full functionality preservation (FTS5, sqlite-vec,
persistence, reindex).

Also verifies that a second KnowledgeGraph instance with vibe-platform
types works correctly — proving the library is consumer-agnostic.
"""

import json
import math

import pytest

from chora_knowledge_graph import (
    InvalidEdgeTypeError,
    InvalidNodeTypeError,
    KnowledgeGraph,
)


# -- Helpers ----------------------------------------------------------------


def _make_embedding(dim: int, nonzero: dict[int, float]) -> list[float]:
    """Create a normalized sparse embedding for testing."""
    vec = [0.0] * dim
    for idx, val in nonzero.items():
        vec[idx] = val
    mag = math.sqrt(sum(x * x for x in vec))
    if mag > 0:
        vec = [x / mag for x in vec]
    return vec


# -- Type registries -------------------------------------------------------

# chora-core types (the original consumer)
CHORA_NODE_TYPES = frozenset(
    {
        "Term", "BoundedContext", "Service", "Standard",
        "Pattern", "Axiom", "CyclePhase", "Discovery",
    }
)
CHORA_EDGE_TYPES = frozenset(
    {
        "belongs_to", "depends_on", "derives_from", "replaces", "extends",
        "governed_by", "uses_term", "introduced_by", "produces",
        "discovered_during", "feeds_forward_to",
    }
)

# vibe-platform types (the new consumer)
VIBE_NODE_TYPES = frozenset(
    {
        "Artifact", "Section", "Space", "Zone",
        "Circle", "Session", "Definition", "Generation",
    }
)
VIBE_EDGE_TYPES = frozenset(
    {
        "contains", "renders_from", "belongs_to",
        "composed_in", "member_of", "transcribed_in",
    }
)


# -- Configurable types ---------------------------------------------------


class TestConfigurableTypes:
    """KnowledgeGraph accepts custom type registries."""

    def test_accepts_chora_types(self) -> None:
        g = KnowledgeGraph(
            valid_node_types=CHORA_NODE_TYPES,
            valid_edge_types=CHORA_EDGE_TYPES,
        )
        g.add_node("Term", "t1", name="test term")
        assert g.get_node("t1") is not None

    def test_accepts_vibe_types(self) -> None:
        g = KnowledgeGraph(
            valid_node_types=VIBE_NODE_TYPES,
            valid_edge_types=VIBE_EDGE_TYPES,
        )
        g.add_node("Artifact", "a1", name="Vibe Cafe Layout")
        g.add_node("Section", "s1", name="Entrance Zone")
        g.add_edge("contains", "a1", "s1")
        assert g.get_node("a1") is not None
        assert g.get_node("s1") is not None
        edges = g.get_edges("a1")
        assert len(edges) == 1
        assert edges[0]["type"] == "contains"

    def test_rejects_invalid_node_type(self) -> None:
        g = KnowledgeGraph(
            valid_node_types=VIBE_NODE_TYPES,
            valid_edge_types=VIBE_EDGE_TYPES,
        )
        with pytest.raises(InvalidNodeTypeError, match="Invalid node type 'Term'"):
            g.add_node("Term", "t1", name="wrong type for vibe")

    def test_rejects_invalid_edge_type(self) -> None:
        g = KnowledgeGraph(
            valid_node_types=VIBE_NODE_TYPES,
            valid_edge_types=VIBE_EDGE_TYPES,
        )
        g.add_node("Artifact", "a1", name="artifact")
        g.add_node("Section", "s1", name="section")
        with pytest.raises(InvalidEdgeTypeError, match="Invalid edge type 'governed_by'"):
            g.add_edge("governed_by", "a1", "s1")

    def test_no_type_validation_when_none(self) -> None:
        """When no type registries provided, all types are accepted."""
        g = KnowledgeGraph()
        g.add_node("AnyType", "n1", name="anything")
        g.add_node("AnotherType", "n2", name="also fine")
        g.add_edge("any_edge", "n1", "n2")
        assert g.get_node("n1") is not None
        assert g.get_node("n2") is not None

    def test_accepts_set_not_just_frozenset(self) -> None:
        """Constructor accepts plain sets too."""
        g = KnowledgeGraph(
            valid_node_types={"TypeA", "TypeB"},
            valid_edge_types={"links_to"},
        )
        g.add_node("TypeA", "a1", name="a")
        g.add_node("TypeB", "b1", name="b")
        g.add_edge("links_to", "a1", "b1")
        assert g.get_node("a1") is not None


# -- Full functionality with custom types ----------------------------------


class TestVibePlatformTypes:
    """Verify full KG functionality works with vibe-platform types."""

    @pytest.fixture()
    def vibe_graph(self) -> KnowledgeGraph:
        g = KnowledgeGraph(
            embedding_dim=768,
            valid_node_types=VIBE_NODE_TYPES,
            valid_edge_types=VIBE_EDGE_TYPES,
        )
        g.add_node(
            "Space", "vibe-cafe", name="Vibe Cafe",
            description="The default community space",
        )
        g.add_node("Zone", "talk-zone-1", name="Talk Zone 1", capacity=5, mode="convened")
        g.add_node("Zone", "floor", name="Cafe Floor", mode="spatial")
        g.add_node("Artifact", "cafe-layout", name="Cafe Layout Artifact")
        g.add_node("Circle", "core-team", name="Core Team", members=["alice", "bob"])
        g.add_node("Session", "session-001", name="Tuesday standup")
        g.add_edge("contains", "vibe-cafe", "talk-zone-1")
        g.add_edge("contains", "vibe-cafe", "floor")
        g.add_edge("renders_from", "vibe-cafe", "cafe-layout")
        g.add_edge("composed_in", "cafe-layout", "session-001")
        g.add_edge("member_of", "session-001", "core-team")
        return g

    def test_query_node_and_neighbors(self, vibe_graph: KnowledgeGraph) -> None:
        result = vibe_graph.query("vibe-cafe")
        assert result is not None
        assert result["node"]["id"] == "vibe-cafe"
        assert len(result["edges"]) == 3  # 2 contains + 1 renders_from
        neighbor_ids = {n["id"] for n in result["neighbors"]}
        assert "talk-zone-1" in neighbor_ids
        assert "cafe-layout" in neighbor_ids

    def test_query_domain_not_applicable(self, vibe_graph: KnowledgeGraph) -> None:
        """query_domain works (returns empty if no domain attr)."""
        result = vibe_graph.query_domain("vibe-platform")
        assert "nodes" in result

    def test_fts5_search(self, vibe_graph: KnowledgeGraph) -> None:
        result = vibe_graph.search_terms("cafe")
        assert result["search_method"] == "fts5"
        assert result["match_count"] >= 1
        ids = [m["id"] for m in result["matches"]]
        assert "vibe-cafe" in ids

    def test_semantic_search(self, vibe_graph: KnowledgeGraph) -> None:
        # Add embeddings
        vibe_graph._write_embedding(
            "vibe-cafe", _make_embedding(768, {0: 0.9, 1: 0.1})
        )
        vibe_graph._write_embedding(
            "talk-zone-1", _make_embedding(768, {2: 0.8, 3: 0.2})
        )

        query = _make_embedding(768, {0: 0.8, 1: 0.2})
        result = vibe_graph.semantic_search(query, top_k=2)
        assert result["search_method"] == "semantic"
        assert result["match_count"] >= 1
        assert result["matches"][0]["id"] == "vibe-cafe"

    def test_stats(self, vibe_graph: KnowledgeGraph) -> None:
        s = vibe_graph.stats()
        assert s["total_nodes"] == 6
        assert s["total_edges"] == 5
        assert s["nodes"]["Space"] == 1
        assert s["nodes"]["Zone"] == 2

    def test_persistence_roundtrip(self, vibe_graph: KnowledgeGraph) -> None:
        # Add an embedding to verify it persists
        vibe_graph._write_embedding("vibe-cafe", _make_embedding(768, {0: 1.0}))

        json_str = vibe_graph.to_json()
        g2 = KnowledgeGraph.from_json(
            json_str,
            valid_node_types=VIBE_NODE_TYPES,
            valid_edge_types=VIBE_EDGE_TYPES,
        )
        assert g2.stats()["total_nodes"] == 6
        assert g2.stats()["total_edges"] == 5
        assert g2.embedding_count() == 1
        assert g2.get_node("vibe-cafe") is not None
        node = g2.get_node("vibe-cafe")
        assert node is not None
        assert node["name"] == "Vibe Cafe"

    def test_reindex_embeddings(self, vibe_graph: KnowledgeGraph) -> None:
        call_idx = [0]

        def embed_fn(text: str) -> list[float]:
            idx = call_idx[0]
            call_idx[0] += 1
            vec = [0.0] * 768
            vec[idx % 768] = 1.0
            return vec

        result = vibe_graph.reindex_embeddings(embed_fn)
        assert result["reindexed"] == 6
        assert result["failed"] == 0
        assert vibe_graph.embedding_count() == 6


# -- Error messages --------------------------------------------------------


class TestErrorMessages:
    """Error messages include valid types for the specific registry."""

    def test_node_error_shows_valid_types(self) -> None:
        g = KnowledgeGraph(valid_node_types={"A", "B"})
        with pytest.raises(InvalidNodeTypeError) as exc_info:
            g.add_node("C", "c1")
        assert "A" in str(exc_info.value)
        assert "B" in str(exc_info.value)

    def test_edge_error_shows_valid_types(self) -> None:
        g = KnowledgeGraph(valid_node_types={"A"}, valid_edge_types={"x"})
        g.add_node("A", "a1")
        g.add_node("A", "a2")
        with pytest.raises(InvalidEdgeTypeError) as exc_info:
            g.add_edge("y", "a1", "a2")
        assert "x" in str(exc_info.value)


# -- Persistence additional tests ------------------------------------------


class TestPersistence:
    """Test JSON serialization edge cases."""

    def test_empty_graph_to_json(self) -> None:
        g = KnowledgeGraph()
        data = json.loads(g.to_json())
        assert data["nodes"] == []
        assert data["edges"] == []
        assert data["embeddings"] == {}

    def test_from_json_with_no_embeddings_key(self) -> None:
        """Backward compat: JSON without embeddings key."""
        json_str = json.dumps({"nodes": [], "edges": []})
        g = KnowledgeGraph.from_json(json_str)
        assert g.stats()["total_nodes"] == 0

    def test_close(self) -> None:
        g = KnowledgeGraph()
        g.add_node("Test", "t1", name="test")
        g.close()
