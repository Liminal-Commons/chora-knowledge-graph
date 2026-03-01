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
    NodeNotFoundError,
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


# -- Update node ---------------------------------------------------------------


class TestUpdateNode:
    """update_node merges attrs on existing nodes."""

    def test_merge_new_attr(self) -> None:
        g = KnowledgeGraph()
        g.add_node("Term", "t1", name="original")
        g.update_node("t1", description="added later")
        node = g.get_node("t1")
        assert node is not None
        assert node["name"] == "original"
        assert node["description"] == "added later"

    def test_overwrite_existing_attr(self) -> None:
        g = KnowledgeGraph()
        g.add_node("Term", "t1", name="old", version=1)
        g.update_node("t1", version=2)
        node = g.get_node("t1")
        assert node is not None
        assert node["version"] == 2
        assert node["name"] == "old"

    def test_updates_fts_index(self) -> None:
        g = KnowledgeGraph()
        g.add_node("Term", "t1", name="alpha")
        g.update_node("t1", name="beta")
        result = g.search_terms("beta")
        assert result["match_count"] >= 1
        assert any(m["id"] == "t1" for m in result["matches"])

    def test_raises_for_missing_node(self) -> None:
        g = KnowledgeGraph()
        with pytest.raises(NodeNotFoundError):
            g.update_node("nonexistent", name="fail")

    def test_preserves_type(self) -> None:
        g = KnowledgeGraph()
        g.add_node("Pattern", "p1", name="my pattern")
        g.update_node("p1", severity="high")
        node = g.get_node("p1")
        assert node is not None
        assert node["type"] == "Pattern"


# -- Delete node ---------------------------------------------------------------


class TestDeleteNode:
    """delete_node cascades to edges, FTS, and embeddings."""

    def test_removes_node(self) -> None:
        g = KnowledgeGraph()
        g.add_node("Term", "t1", name="test")
        result = g.delete_node("t1")
        assert result is True
        assert g.get_node("t1") is None

    def test_cascades_edges(self) -> None:
        g = KnowledgeGraph()
        g.add_node("Term", "t1", name="a")
        g.add_node("Term", "t2", name="b")
        g.add_edge("replaces", "t2", "t1")
        g.delete_node("t1")
        edges = g.get_edges("t2")
        assert len(edges) == 0

    def test_removes_from_fts(self) -> None:
        g = KnowledgeGraph()
        g.add_node("Term", "t1", name="unicorn")
        g.delete_node("t1")
        result = g.search_terms("unicorn")
        assert result["match_count"] == 0

    def test_removes_embedding(self) -> None:
        g = KnowledgeGraph()
        g.add_node("Term", "t1", name="test")
        g._write_embedding("t1", _make_embedding(768, {0: 1.0}))
        assert g.embedding_count() == 1
        g.delete_node("t1")
        assert g.embedding_count() == 0

    def test_returns_false_for_missing(self) -> None:
        g = KnowledgeGraph()
        assert g.delete_node("nonexistent") is False


# -- Delete edge ---------------------------------------------------------------


class TestDeleteEdge:
    """delete_edge removes specific edges."""

    def test_removes_edge(self) -> None:
        g = KnowledgeGraph()
        g.add_node("Term", "t1", name="a")
        g.add_node("Term", "t2", name="b")
        g.add_edge("replaces", "t1", "t2")
        result = g.delete_edge("t1", "t2", "replaces")
        assert result is True
        assert len(g.get_edges("t1")) == 0

    def test_returns_false_for_missing(self) -> None:
        g = KnowledgeGraph()
        assert g.delete_edge("x", "y", "z") is False

    def test_preserves_other_edges(self) -> None:
        g = KnowledgeGraph()
        g.add_node("Term", "t1", name="a")
        g.add_node("Term", "t2", name="b")
        g.add_node("Term", "t3", name="c")
        g.add_edge("replaces", "t1", "t2")
        g.add_edge("replaces", "t1", "t3")
        g.delete_edge("t1", "t2", "replaces")
        edges = g.get_edges("t1")
        assert len(edges) == 1
        assert edges[0]["target"] == "t3"


# -- List nodes ----------------------------------------------------------------


class TestListNodes:
    """list_nodes with type and attribute filtering."""

    @pytest.fixture()
    def populated_graph(self) -> KnowledgeGraph:
        g = KnowledgeGraph()
        g.add_node("Term", "t1", name="alpha", domain="auth")
        g.add_node("Term", "t2", name="beta", domain="auth")
        g.add_node("Term", "t3", name="gamma", domain="payments")
        g.add_node("Pattern", "p1", name="singleton", domain="auth")
        return g

    def test_list_all(self, populated_graph: KnowledgeGraph) -> None:
        nodes = populated_graph.list_nodes()
        assert len(nodes) == 4

    def test_filter_by_type(self, populated_graph: KnowledgeGraph) -> None:
        nodes = populated_graph.list_nodes(node_type="Term")
        assert len(nodes) == 3
        assert all(n["type"] == "Term" for n in nodes)

    def test_filter_by_attr(self, populated_graph: KnowledgeGraph) -> None:
        nodes = populated_graph.list_nodes(domain="auth")
        assert len(nodes) == 3

    def test_filter_by_type_and_attr(self, populated_graph: KnowledgeGraph) -> None:
        nodes = populated_graph.list_nodes(node_type="Term", domain="auth")
        assert len(nodes) == 2

    def test_limit_and_offset(self, populated_graph: KnowledgeGraph) -> None:
        nodes = populated_graph.list_nodes(limit=2)
        assert len(nodes) == 2
        nodes2 = populated_graph.list_nodes(limit=2, offset=2)
        assert len(nodes2) == 2
        ids1 = {n["id"] for n in nodes}
        ids2 = {n["id"] for n in nodes2}
        assert ids1.isdisjoint(ids2)


# -- Walk (BFS traversal) -----------------------------------------------------


class TestWalk:
    """walk performs BFS traversal with cycle detection."""

    @pytest.fixture()
    def chain_graph(self) -> KnowledgeGraph:
        g = KnowledgeGraph()
        g.add_node("Behavior", "b1", name="login")
        g.add_node("Behavior", "b2", name="user-exists")
        g.add_node("Behavior", "b3", name="db-connected")
        g.add_edge("depends_on", "b1", "b2")
        g.add_edge("depends_on", "b2", "b3")
        return g

    def test_walks_chain(self, chain_graph: KnowledgeGraph) -> None:
        result = chain_graph.walk("b1", edge_types=["depends_on"])
        assert len(result) == 3
        assert result[0]["node"]["id"] == "b1"
        assert result[0]["depth"] == 0
        assert result[1]["node"]["id"] == "b2"
        assert result[1]["depth"] == 1
        assert result[2]["node"]["id"] == "b3"
        assert result[2]["depth"] == 2

    def test_respects_max_depth(self, chain_graph: KnowledgeGraph) -> None:
        result = chain_graph.walk("b1", edge_types=["depends_on"], max_depth=1)
        assert len(result) == 2  # root + 1 hop

    def test_handles_cycles(self) -> None:
        g = KnowledgeGraph()
        g.add_node("N", "a", name="a")
        g.add_node("N", "b", name="b")
        g.add_edge("links", "a", "b")
        g.add_edge("links", "b", "a")
        result = g.walk("a", edge_types=["links"])
        assert len(result) == 2  # visits each once

    def test_returns_empty_for_missing_node(self) -> None:
        g = KnowledgeGraph()
        assert g.walk("nonexistent") == []

    def test_incoming_direction(self, chain_graph: KnowledgeGraph) -> None:
        result = chain_graph.walk("b3", edge_types=["depends_on"], direction="incoming")
        assert len(result) == 3
        ids = [r["node"]["id"] for r in result]
        assert "b3" in ids
        assert "b2" in ids
        assert "b1" in ids

    def test_root_only_no_matching_edges(self) -> None:
        g = KnowledgeGraph()
        g.add_node("N", "a", name="lonely")
        result = g.walk("a")
        assert len(result) == 1
        assert result[0]["node"]["id"] == "a"
        assert result[0]["edge"] is None


# -- Enhanced get_edges --------------------------------------------------------


class TestGetEdgesEnhanced:
    """get_edges with direction and edge_type filtering."""

    @pytest.fixture()
    def edge_graph(self) -> KnowledgeGraph:
        g = KnowledgeGraph()
        g.add_node("N", "a", name="a")
        g.add_node("N", "b", name="b")
        g.add_node("N", "c", name="c")
        g.add_edge("x", "a", "b")
        g.add_edge("y", "a", "c")
        g.add_edge("x", "c", "a")
        return g

    def test_default_both_directions(self, edge_graph: KnowledgeGraph) -> None:
        edges = edge_graph.get_edges("a")
        assert len(edges) == 3  # 2 outgoing + 1 incoming

    def test_outgoing_only(self, edge_graph: KnowledgeGraph) -> None:
        edges = edge_graph.get_edges("a", direction="outgoing")
        assert len(edges) == 2
        assert all(e["source"] == "a" for e in edges)

    def test_incoming_only(self, edge_graph: KnowledgeGraph) -> None:
        edges = edge_graph.get_edges("a", direction="incoming")
        assert len(edges) == 1
        assert edges[0]["source"] == "c"

    def test_filter_by_edge_type(self, edge_graph: KnowledgeGraph) -> None:
        edges = edge_graph.get_edges("a", edge_type="x")
        assert len(edges) == 2  # one outgoing (a->b) + one incoming (c->a)

    def test_filter_by_type_and_direction(self, edge_graph: KnowledgeGraph) -> None:
        edges = edge_graph.get_edges("a", edge_type="x", direction="outgoing")
        assert len(edges) == 1
        assert edges[0]["target"] == "b"
