"""KnowledgeGraph — SQLite-backed knowledge graph with FTS5 and sqlite-vec.

Configurable type registries allow different consumers to share the same
graph engine with domain-specific node and edge types:

    from chora_knowledge_graph import KnowledgeGraph

    # chora-core types
    g = KnowledgeGraph(
        valid_node_types={"Term", "Axiom", "Pattern", ...},
        valid_edge_types={"governed_by", "replaces", ...},
    )

    # vibe-platform types
    g = KnowledgeGraph(
        valid_node_types={"Artifact", "Section", "Space", "Zone", "Circle", ...},
        valid_edge_types={"contains", "renders_from", "belongs_to", ...},
    )

    # No type validation (accept anything)
    g = KnowledgeGraph()

Storage: SQLite + FTS5 (full-text search) + sqlite-vec (vector embeddings).
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import sqlite3
import struct
import time
from collections import Counter
from typing import Any

import sqlite_vec  # type: ignore[import-untyped]

from chora_knowledge_graph.types import InvalidEdgeTypeError, InvalidNodeTypeError

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Knowledge layer backed by SQLite with FTS5 and sqlite-vec.

    Args:
        db_path: Path to SQLite file, or ":memory:" for in-memory (default).
        embedding_dim: Dimension of embedding vectors (default 768).
        valid_node_types: Set of allowed node types. None = accept all.
        valid_edge_types: Set of allowed edge types. None = accept all.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        embedding_dim: int = 768,
        valid_node_types: set[str] | frozenset[str] | None = None,
        valid_edge_types: set[str] | frozenset[str] | None = None,
    ) -> None:
        self._embedding_dim = embedding_dim
        self._valid_node_types: frozenset[str] | None = (
            frozenset(valid_node_types) if valid_node_types is not None else None
        )
        self._valid_edge_types: frozenset[str] | None = (
            frozenset(valid_edge_types) if valid_edge_types is not None else None
        )
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._create_tables()

    def _create_tables(self) -> None:
        """Create schema tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                attrs TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS edges (
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                type TEXT NOT NULL,
                attrs TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (source, target, type)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                node_id,
                searchable_text,
                tokenize='unicode61'
            );
        """)

        # sqlite-vec table — created empty, ready for embeddings
        with contextlib.suppress(sqlite3.OperationalError):
            self._conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS node_embeddings
                USING vec0(id TEXT PRIMARY KEY, embedding float[{self._embedding_dim}])
            """)

        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # ── FTS5 helpers ────────────────────────────────────────────────

    def _build_searchable_text(self, node_id: str, attrs: dict[str, Any]) -> str:
        """Build searchable text from node ID + all text-bearing attributes."""
        parts = [node_id]
        for value in attrs.values():
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                parts.extend(str(item) for item in value if isinstance(item, str))
        return " ".join(parts)

    def _index_node_fts(self, node_id: str, attrs: dict[str, Any]) -> None:
        """Insert or update FTS5 index for a node."""
        searchable = self._build_searchable_text(node_id, attrs)
        # Delete old entry if exists, then insert new
        self._conn.execute("DELETE FROM nodes_fts WHERE node_id = ?", (node_id,))
        self._conn.execute(
            "INSERT INTO nodes_fts(node_id, searchable_text) VALUES (?, ?)",
            (node_id, searchable),
        )

    # ── Vector helpers ───────────────────────────────────────────────

    @staticmethod
    def _normalize(vec: list[float]) -> list[float]:
        """L2-normalize a vector. Returns zero vector if magnitude is 0."""
        mag = math.sqrt(sum(x * x for x in vec))
        if mag > 0:
            return [x / mag for x in vec]
        return vec

    def _write_embedding(self, node_id: str, embedding: list[float]) -> None:
        """Store a normalized embedding vector for a node."""
        embedding = self._normalize(embedding)
        # sqlite-vec virtual tables don't support REPLACE — delete first
        self._conn.execute(
            "DELETE FROM node_embeddings WHERE id = ?", (node_id,)
        )
        self._conn.execute(
            "INSERT INTO node_embeddings(id, embedding) VALUES (?, ?)",
            (node_id, json.dumps(embedding)),
        )
        self._conn.commit()

    def _read_embedding(self, node_id: str) -> list[float] | None:
        """Read an embedding vector for a node."""
        row = self._conn.execute(
            "SELECT embedding FROM node_embeddings WHERE id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        # sqlite-vec returns bytes; decode to list
        raw = row[0]
        if isinstance(raw, bytes):
            count = len(raw) // 4
            return list(struct.unpack(f"{count}f", raw))
        return json.loads(raw)  # type: ignore[no-any-return]

    # ── Node operations ──────────────────────────────────────────────

    def add_node(self, node_type: str, node_id: str, **attrs: Any) -> None:
        """Add a typed node to the graph.

        Args:
            node_type: Node type. Must be in valid_node_types if set.
            node_id: Unique identifier for the node.
            **attrs: Arbitrary attributes stored on the node.

        Raises:
            InvalidNodeTypeError: If node_type is not in the registry.
        """
        if self._valid_node_types is not None and node_type not in self._valid_node_types:
            raise InvalidNodeTypeError(node_type, self._valid_node_types)
        attrs_json = json.dumps(attrs, ensure_ascii=False)
        self._conn.execute(
            "INSERT OR REPLACE INTO nodes (id, type, attrs) VALUES (?, ?, ?)",
            (node_id, node_type, attrs_json),
        )
        self._index_node_fts(node_id, attrs)
        self._conn.commit()

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Get a node's attributes by ID. Returns None if not found."""
        row = self._conn.execute(
            "SELECT id, type, attrs FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        attrs: dict[str, Any] = json.loads(row[2])
        attrs["id"] = row[0]
        attrs["type"] = row[1]
        return attrs

    # ── Edge operations ──────────────────────────────────────────────

    def add_edge(self, edge_type: str, source: str, target: str, **attrs: Any) -> None:
        """Add a typed edge between two nodes.

        Args:
            edge_type: Edge type. Must be in valid_edge_types if set.
            source: Source node ID.
            target: Target node ID.
            **attrs: Arbitrary attributes stored on the edge.

        Raises:
            InvalidEdgeTypeError: If edge_type is not in the registry.
        """
        if self._valid_edge_types is not None and edge_type not in self._valid_edge_types:
            raise InvalidEdgeTypeError(edge_type, self._valid_edge_types)
        attrs_json = json.dumps(attrs, ensure_ascii=False)
        self._conn.execute(
            "INSERT OR REPLACE INTO edges (source, target, type, attrs) VALUES (?, ?, ?, ?)",
            (source, target, edge_type, attrs_json),
        )
        self._conn.commit()

    def get_edges(self, node_id: str) -> list[dict[str, Any]]:
        """Get all edges connected to a node (outgoing and incoming)."""
        edges: list[dict[str, Any]] = []

        rows = self._conn.execute(
            "SELECT source, target, type, attrs FROM edges WHERE source = ? OR target = ?",
            (node_id, node_id),
        ).fetchall()

        for source, target, edge_type, attrs_json in rows:
            edge: dict[str, Any] = json.loads(attrs_json)
            edge["source"] = source
            edge["target"] = target
            edge["type"] = edge_type
            edges.append(edge)

        return edges

    # ── Query operations ─────────────────────────────────────────────

    def query(self, node_id: str) -> dict[str, Any] | None:
        """Query a node and its 1-hop neighborhood.

        Returns:
            Dict with 'node', 'edges', and 'neighbors' keys,
            or None if node_id not found.
        """
        node = self.get_node(node_id)
        if node is None:
            return None

        edges = self.get_edges(node_id)

        # Collect unique neighbor IDs
        neighbor_ids: set[str] = set()
        for edge in edges:
            if edge["source"] != node_id:
                neighbor_ids.add(edge["source"])
            if edge["target"] != node_id:
                neighbor_ids.add(edge["target"])

        neighbors = [
            self.get_node(nid)
            for nid in sorted(neighbor_ids)
            if self.get_node(nid) is not None
        ]

        return {
            "node": node,
            "edges": edges,
            "neighbors": neighbors,
        }

    def query_domain(self, domain: str) -> dict[str, Any]:
        """Query all nodes belonging to a domain/bounded context.

        Matches nodes that have a 'domain' attribute equal to the given domain,
        or nodes with a 'name' attribute matching the domain (for BoundedContext nodes).

        Returns:
            Dict with 'domain', 'nodes', and 'edges' keys.
        """
        matching_ids: list[str] = []

        rows = self._conn.execute(
            """
            SELECT id, type, attrs FROM nodes
            WHERE json_extract(attrs, '$.domain') = ?
               OR (type = 'BoundedContext' AND json_extract(attrs, '$.name') = ?)
            """,
            (domain, domain),
        ).fetchall()

        for row in rows:
            matching_ids.append(row[0])

        nodes = [
            self.get_node(nid) for nid in matching_ids if self.get_node(nid) is not None
        ]

        # Collect edges connected to matching nodes
        matching_set = set(matching_ids)
        edges: list[dict[str, Any]] = []
        if matching_set:
            placeholders = ",".join("?" for _ in matching_set)
            sql = (
                f"SELECT source, target, type, attrs FROM edges "  # nosec B608
                f"WHERE source IN ({placeholders}) OR target IN ({placeholders})"
            )
            edge_rows = self._conn.execute(
                sql,
                list(matching_set) + list(matching_set),
            ).fetchall()
            for source, target, edge_type, attrs_json in edge_rows:
                edge: dict[str, Any] = json.loads(attrs_json)
                edge["source"] = source
                edge["target"] = target
                edge["type"] = edge_type
                edges.append(edge)

        return {
            "domain": domain,
            "nodes": nodes,
            "edges": edges,
        }

    # ── Search operations ────────────────────────────────────────────

    def search_terms(self, query_string: str) -> dict[str, Any]:
        """Search for nodes matching a query string via FTS5.

        Returns:
            Dict with 'matches', 'search_method', 'query', and 'match_count' keys.
        """
        matches: list[dict[str, Any]] = []

        if not query_string.strip():
            return {
                "query": query_string,
                "matches": [],
                "search_method": "fts5",
                "match_count": 0,
            }

        # Split into individual terms, prefix-match each, OR them together.
        safe_query = query_string.replace('"', '""')
        terms = safe_query.split()
        if len(terms) == 1:
            fts_query = f'"{terms[0]}"*'
        else:
            fts_query = " OR ".join(f'"{t}"*' for t in terms if t.strip())

        try:
            rows = self._conn.execute(
                "SELECT node_id FROM nodes_fts WHERE nodes_fts MATCH ?",
                (fts_query,),
            ).fetchall()
        except sqlite3.OperationalError:
            # Malformed FTS5 query — fall back to empty results
            rows = []

        for (node_id,) in rows:
            node = self.get_node(node_id)
            if node is not None:
                matches.append(node)

        return {
            "query": query_string,
            "matches": matches,
            "search_method": "fts5",
            "match_count": len(matches),
        }

    def semantic_search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> dict[str, Any]:
        """Search for nodes by vector similarity using sqlite-vec.

        Uses L2 distance on normalized vectors (equivalent to cosine
        similarity ranking).

        Returns:
            Dict with 'matches', 'search_method', and 'match_count' keys.
            Each match includes a 'similarity' score in [0, 1].
        """
        query_embedding = self._normalize(query_embedding)

        try:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM node_embeddings"
            ).fetchone()[0]
        except sqlite3.OperationalError:
            count = 0

        if count == 0:
            return {
                "matches": [],
                "search_method": "semantic",
                "match_count": 0,
            }

        rows = self._conn.execute(
            "SELECT id, distance FROM node_embeddings "
            "WHERE embedding MATCH ? AND k = ? "
            "ORDER BY distance",
            (json.dumps(query_embedding), top_k),
        ).fetchall()

        matches: list[dict[str, Any]] = []
        for node_id, distance in rows:
            node = self.get_node(node_id)
            if node is not None:
                # L2 distance -> cosine similarity for normalized vectors:
                # cos(theta) = 1 - d^2/2
                node["similarity"] = round(1.0 - (distance**2) / 2.0, 6)
                matches.append(node)

        return {
            "matches": matches,
            "search_method": "semantic",
            "match_count": len(matches),
        }

    # ── Embedding operations ────────────────────────────────────────

    def embedding_count(self) -> int:
        """Return the number of nodes that have stored embeddings."""
        try:
            result = self._conn.execute(
                "SELECT COUNT(*) FROM node_embeddings"
            ).fetchone()
            return result[0] if result else 0
        except sqlite3.OperationalError:
            return 0

    def reindex_embeddings(
        self, embed_fn: Any, *, batch_commit: bool = True
    ) -> dict[str, Any]:
        """Re-embed all graph nodes.

        Args:
            embed_fn: Callable[[str], list[float]] — embeds text to vector.
            batch_commit: If True, commit once at end (faster).

        Returns:
            Dict with 'reindexed', 'failed', and 'duration_ms' keys.
        """
        start = time.monotonic()
        reindexed = 0
        failed = 0

        rows = self._conn.execute("SELECT id, type, attrs FROM nodes").fetchall()
        for node_id, _node_type, attrs_json in rows:
            attrs: dict[str, Any] = json.loads(attrs_json)
            searchable = self._build_searchable_text(node_id, attrs)
            try:
                embedding = embed_fn(searchable)
                normalized = self._normalize(embedding)
                # sqlite-vec virtual tables don't support REPLACE — delete first
                self._conn.execute(
                    "DELETE FROM node_embeddings WHERE id = ?", (node_id,)
                )
                self._conn.execute(
                    "INSERT INTO node_embeddings(id, embedding) VALUES (?, ?)",
                    (node_id, json.dumps(normalized)),
                )
                reindexed += 1
            except Exception:
                logger.warning("Failed to embed node %s", node_id, exc_info=True)
                failed += 1

        if batch_commit:
            self._conn.commit()

        duration_ms = round((time.monotonic() - start) * 1000)
        return {
            "reindexed": reindexed,
            "failed": failed,
            "duration_ms": duration_ms,
        }

    # ── Stats ────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return node/edge counts by type."""
        node_types: Counter[str] = Counter()
        for ntype, count in self._conn.execute(
            "SELECT type, COUNT(*) FROM nodes GROUP BY type"
        ).fetchall():
            node_types[ntype] = count

        edge_types: Counter[str] = Counter()
        for etype, count in self._conn.execute(
            "SELECT type, COUNT(*) FROM edges GROUP BY type"
        ).fetchall():
            edge_types[etype] = count

        total_nodes = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        total_edges = self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

        return {
            "nodes": dict(node_types),
            "edges": dict(edge_types),
            "total_nodes": total_nodes,
            "total_edges": total_edges,
        }

    # ── Persistence (JSON export/import) ─────────────────────────────

    def to_json(self) -> str:
        """Serialize the graph to JSON. Includes nodes, edges, and embeddings."""
        nodes: list[dict[str, Any]] = []
        for row in self._conn.execute("SELECT id, type, attrs FROM nodes").fetchall():
            node: dict[str, Any] = json.loads(row[2])
            node["id"] = row[0]
            node["type"] = row[1]
            nodes.append(node)

        edges: list[dict[str, Any]] = []
        for row in self._conn.execute(
            "SELECT source, target, type, attrs FROM edges"
        ).fetchall():
            edge: dict[str, Any] = json.loads(row[3])
            edge["source"] = row[0]
            edge["target"] = row[1]
            edge["type"] = row[2]
            edges.append(edge)

        # Serialize embeddings from sqlite-vec
        embeddings: dict[str, list[float]] = {}
        try:
            for row in self._conn.execute(
                "SELECT id, embedding FROM node_embeddings"
            ).fetchall():
                node_id = row[0]
                raw = row[1]
                if isinstance(raw, bytes):
                    count = len(raw) // 4
                    embeddings[node_id] = list(struct.unpack(f"{count}f", raw))
                else:
                    embeddings[node_id] = json.loads(raw)
        except sqlite3.OperationalError:
            pass  # No embeddings table yet

        return json.dumps(
            {"nodes": nodes, "edges": edges, "embeddings": embeddings},
            indent=2,
            ensure_ascii=False,
        )

    @classmethod
    def from_json(
        cls,
        json_str: str,
        db_path: str = ":memory:",
        valid_node_types: set[str] | frozenset[str] | None = None,
        valid_edge_types: set[str] | frozenset[str] | None = None,
    ) -> KnowledgeGraph:
        """Deserialize a graph from JSON into a new KnowledgeGraph."""
        data = json.loads(json_str)
        g = cls(
            db_path=db_path,
            valid_node_types=valid_node_types,
            valid_edge_types=valid_edge_types,
        )

        for node_data in data.get("nodes", []):
            node_data = dict(node_data)  # Don't mutate caller's data
            node_id = node_data.pop("id")
            node_type = node_data.pop("type")
            g.add_node(node_type, node_id, **node_data)

        for edge_data in data.get("edges", []):
            edge_data = dict(edge_data)
            source = edge_data.pop("source")
            target = edge_data.pop("target")
            edge_type = edge_data.pop("type")
            g.add_edge(edge_type, source, target, **edge_data)

        # Restore embeddings
        for node_id, embedding in data.get("embeddings", {}).items():
            g._write_embedding(node_id, embedding)

        return g
