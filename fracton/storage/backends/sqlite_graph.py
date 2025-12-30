"""
SQLite Graph Backend for KRONOS

Lightweight, file-based graph storage using SQLite.
Perfect for development, single-user apps, and prototyping.

Features:
- No external server required
- ACID transactions
- Efficient indexing
- PAC delta storage as blobs
- Up to ~1M nodes
"""

from __future__ import annotations

import json
import sqlite3
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import logging

from .base import (
    GraphBackend,
    GraphNode,
    GraphEdge,
    GraphNeighborhood,
    TemporalPath,
)

logger = logging.getLogger(__name__)


class SQLiteGraphBackend(GraphBackend):
    """
    SQLite-based graph storage backend.

    Schema:
    - graphs: Metadata about named graphs
    - nodes: Memory nodes with PAC structure
    - edges: Typed relationships between nodes

    Indexes:
    - nodes: (graph_name, id), (parent_id), (timestamp)
    - edges: (graph_name, from_id), (to_id), (relation_type)
    """

    def __init__(self, db_path: Path):
        """
        Initialize SQLite backend.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: Optional[sqlite3.Connection] = None

        logger.info(f"SQLiteGraphBackend initialized: {db_path}")

    async def connect(self) -> None:
        """Establish connection and create schema."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Access columns by name

        # Create schema
        await self._create_schema()

        logger.info("SQLite connection established")

    async def close(self) -> None:
        """Close connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("SQLite connection closed")

    async def _create_schema(self) -> None:
        """Create database schema."""
        cursor = self.conn.cursor()

        # Graphs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS graphs (
                name TEXT PRIMARY KEY,
                description TEXT,
                created_at TEXT,
                node_count INTEGER DEFAULT 0,
                edge_count INTEGER DEFAULT 0
            )
        """)

        # Nodes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT NOT NULL,
                graph_name TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                fractal_signature TEXT NOT NULL,
                metadata TEXT,  -- JSON
                parent_id TEXT,
                children_ids TEXT,  -- JSON array
                potential REAL DEFAULT 1.0,
                entropy REAL DEFAULT 0.5,
                coherence REAL DEFAULT 0.5,
                phase TEXT DEFAULT 'STABLE',
                PRIMARY KEY (graph_name, id)
            )
        """)

        # Indexes on nodes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_nodes_parent
            ON nodes(graph_name, parent_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_nodes_timestamp
            ON nodes(graph_name, timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_nodes_signature
            ON nodes(fractal_signature)
        """)

        # Edges table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                graph_name TEXT NOT NULL,
                from_id TEXT NOT NULL,
                to_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                metadata TEXT,  -- JSON
                timestamp TEXT,
                PRIMARY KEY (graph_name, from_id, to_id, relation_type)
            )
        """)

        # Indexes on edges
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_edges_from
            ON edges(graph_name, from_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_edges_to
            ON edges(graph_name, to_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_edges_type
            ON edges(graph_name, relation_type)
        """)

        self.conn.commit()

    # ========================================================================
    # Node Operations
    # ========================================================================

    async def create_node(
        self,
        node: GraphNode,
        graph_name: str = "default",
    ) -> None:
        """Create node in graph."""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO nodes (
                id, graph_name, content, timestamp, fractal_signature,
                metadata, parent_id, children_ids,
                potential, entropy, coherence, phase
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            node.id,
            graph_name,
            node.content,
            node.timestamp.isoformat(),
            node.fractal_signature,
            json.dumps(node.metadata),
            node.parent_id,
            json.dumps(node.children_ids),
            node.potential,
            node.entropy,
            node.coherence,
            node.phase,
        ))

        # Update parent's children_ids if this is a child
        if node.parent_id != "-1":
            cursor.execute("""
                SELECT children_ids FROM nodes
                WHERE graph_name = ? AND id = ?
            """, (graph_name, node.parent_id))

            row = cursor.fetchone()
            if row:
                children = json.loads(row['children_ids'])
                if node.id not in children:
                    children.append(node.id)
                    cursor.execute("""
                        UPDATE nodes
                        SET children_ids = ?
                        WHERE graph_name = ? AND id = ?
                    """, (json.dumps(children), graph_name, node.parent_id))

        # Update graph node count
        cursor.execute("""
            UPDATE graphs
            SET node_count = node_count + 1
            WHERE name = ?
        """, (graph_name,))

        self.conn.commit()

    async def get_node(
        self,
        node_id: str,
        graph_name: str = "default",
    ) -> Optional[GraphNode]:
        """Retrieve node by ID."""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT * FROM nodes
            WHERE graph_name = ? AND id = ?
        """, (graph_name, node_id))

        row = cursor.fetchone()
        if not row:
            return None

        return self._row_to_node(row)

    async def update_node(
        self,
        node_id: str,
        updates: Dict[str, Any],
        graph_name: str = "default",
    ) -> None:
        """Update node metadata."""
        cursor = self.conn.cursor()

        # Build dynamic UPDATE query
        set_clauses = []
        values = []

        for key, value in updates.items():
            if key in ['metadata', 'children_ids']:
                set_clauses.append(f"{key} = ?")
                values.append(json.dumps(value))
            elif key == 'timestamp':
                set_clauses.append(f"{key} = ?")
                values.append(value.isoformat() if isinstance(value, datetime) else value)
            else:
                set_clauses.append(f"{key} = ?")
                values.append(value)

        if not set_clauses:
            return

        values.extend([graph_name, node_id])

        query = f"""
            UPDATE nodes
            SET {', '.join(set_clauses)}
            WHERE graph_name = ? AND id = ?
        """

        cursor.execute(query, values)
        self.conn.commit()

    async def delete_node(
        self,
        node_id: str,
        graph_name: str = "default",
    ) -> None:
        """Delete node from graph."""
        cursor = self.conn.cursor()

        # Get node to update parent's children list
        node = await self.get_node(node_id, graph_name)
        if node and node.parent_id != "-1":
            cursor.execute("""
                SELECT children_ids FROM nodes
                WHERE graph_name = ? AND id = ?
            """, (graph_name, node.parent_id))

            row = cursor.fetchone()
            if row:
                children = json.loads(row['children_ids'])
                if node_id in children:
                    children.remove(node_id)
                    cursor.execute("""
                        UPDATE nodes
                        SET children_ids = ?
                        WHERE graph_name = ? AND id = ?
                    """, (json.dumps(children), graph_name, node.parent_id))

        # Delete node
        cursor.execute("""
            DELETE FROM nodes
            WHERE graph_name = ? AND id = ?
        """, (graph_name, node_id))

        # Delete associated edges
        cursor.execute("""
            DELETE FROM edges
            WHERE graph_name = ? AND (from_id = ? OR to_id = ?)
        """, (graph_name, node_id, node_id))

        # Update graph counts
        cursor.execute("""
            UPDATE graphs
            SET node_count = node_count - 1
            WHERE name = ?
        """, (graph_name,))

        self.conn.commit()

    # ========================================================================
    # Edge Operations
    # ========================================================================

    async def create_edge(
        self,
        edge: GraphEdge,
        graph_name: str = "default",
    ) -> None:
        """Create edge between nodes."""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO edges (
                graph_name, from_id, to_id, relation_type,
                weight, metadata, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            graph_name,
            edge.from_id,
            edge.to_id,
            edge.relation_type,
            edge.weight,
            json.dumps(edge.metadata),
            edge.timestamp.isoformat(),
        ))

        # Update graph edge count
        cursor.execute("""
            UPDATE graphs
            SET edge_count = edge_count + 1
            WHERE name = ?
        """, (graph_name,))

        self.conn.commit()

    async def get_edges(
        self,
        node_id: str,
        direction: str = "out",
        relation_type: Optional[str] = None,
        graph_name: str = "default",
    ) -> List[GraphEdge]:
        """Get edges connected to node."""
        cursor = self.conn.cursor()

        # Build query based on direction
        if direction == "out":
            where = "from_id = ?"
        elif direction == "in":
            where = "to_id = ?"
        else:  # both
            where = "(from_id = ? OR to_id = ?)"

        query = f"""
            SELECT * FROM edges
            WHERE graph_name = ? AND {where}
        """

        params = [graph_name]
        if direction == "both":
            params.extend([node_id, node_id])
        else:
            params.append(node_id)

        if relation_type:
            query += " AND relation_type = ?"
            params.append(relation_type)

        cursor.execute(query, params)

        edges = []
        for row in cursor.fetchall():
            edges.append(self._row_to_edge(row))

        return edges

    async def delete_edge(
        self,
        from_id: str,
        to_id: str,
        relation_type: str,
        graph_name: str = "default",
    ) -> None:
        """Delete specific edge."""
        cursor = self.conn.cursor()

        cursor.execute("""
            DELETE FROM edges
            WHERE graph_name = ? AND from_id = ? AND to_id = ? AND relation_type = ?
        """, (graph_name, from_id, to_id, relation_type))

        # Update graph edge count
        cursor.execute("""
            UPDATE graphs
            SET edge_count = edge_count - 1
            WHERE name = ?
        """, (graph_name,))

        self.conn.commit()

    # ========================================================================
    # Graph Operations
    # ========================================================================

    async def create_graph(
        self,
        graph_name: str,
        description: str = "",
    ) -> None:
        """Create new named graph."""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT OR IGNORE INTO graphs (name, description, created_at)
            VALUES (?, ?, ?)
        """, (graph_name, description, datetime.now().isoformat()))

        self.conn.commit()

    async def list_graphs(self) -> List[Dict[str, Any]]:
        """List all graphs."""
        cursor = self.conn.cursor()

        cursor.execute("SELECT * FROM graphs")

        graphs = []
        for row in cursor.fetchall():
            graphs.append({
                "name": row['name'],
                "description": row['description'],
                "created_at": row['created_at'],
                "node_count": row['node_count'],
                "edge_count": row['edge_count'],
            })

        return graphs

    async def get_neighbors(
        self,
        node_id: str,
        max_hops: int = 1,
        relation_types: Optional[List[str]] = None,
        graph_name: str = "default",
    ) -> GraphNeighborhood:
        """Get neighborhood around node."""
        visited = {node_id: 0}
        nodes = []
        edges = []
        frontier = [node_id]

        # BFS traversal
        for depth in range(max_hops):
            next_frontier = []

            for current_id in frontier:
                # Get current node
                node = await self.get_node(current_id, graph_name)
                if node:
                    nodes.append(node)

                # Get outgoing edges
                node_edges = await self.get_edges(
                    current_id,
                    direction="both",
                    graph_name=graph_name
                )

                for edge in node_edges:
                    # Filter by relation type if specified
                    if relation_types and edge.relation_type not in relation_types:
                        continue

                    edges.append(edge)

                    # Add neighbor to frontier
                    neighbor_id = edge.to_id if edge.from_id == current_id else edge.from_id
                    if neighbor_id not in visited:
                        visited[neighbor_id] = depth + 1
                        next_frontier.append(neighbor_id)

            frontier = next_frontier
            if not frontier:
                break

        return GraphNeighborhood(
            center_id=node_id,
            nodes=nodes,
            edges=edges,
            depth_map=visited,
        )

    # ========================================================================
    # Temporal Operations
    # ========================================================================

    async def trace_lineage(
        self,
        node_id: str,
        direction: str = "both",
        max_depth: int = 100,
        graph_name: str = "default",
    ) -> TemporalPath:
        """Trace temporal lineage (PAC parent-child chains)."""
        path = []
        nodes = []
        edges = []
        entropy_evolution = []

        # Trace backward to root
        if direction in ["backward", "both"]:
            current_id = node_id
            depth = 0

            while current_id and depth < max_depth:
                node = await self.get_node(current_id, graph_name)
                if not node:
                    break

                path.insert(0, current_id)
                nodes.insert(0, node)
                entropy_evolution.insert(0, node.entropy)

                if node.parent_id == "-1":
                    break

                # Add edge
                edges.insert(0, GraphEdge(
                    from_id=node.parent_id,
                    to_id=current_id,
                    relation_type="PARENT_OF",
                ))

                current_id = node.parent_id
                depth += 1

        # Trace forward through children
        if direction in ["forward", "both"]:
            # Start from node_id if backward wasn't traced
            if direction == "forward":
                start_node = await self.get_node(node_id, graph_name)
                if start_node:
                    path.append(node_id)
                    nodes.append(start_node)
                    entropy_evolution.append(start_node.entropy)

            # Trace down primary child (first child in list)
            current_id = node_id
            depth = 0

            while current_id and depth < max_depth:
                node = await self.get_node(current_id, graph_name)
                if not node or not node.children_ids:
                    break

                # Follow first child
                child_id = node.children_ids[0]
                child = await self.get_node(child_id, graph_name)

                if not child:
                    break

                path.append(child_id)
                nodes.append(child)
                entropy_evolution.append(child.entropy)

                edges.append(GraphEdge(
                    from_id=current_id,
                    to_id=child_id,
                    relation_type="PARENT_OF",
                ))

                current_id = child_id
                depth += 1

        # Calculate total potential
        total_potential = sum(n.potential for n in nodes)

        return TemporalPath(
            root_id=path[0] if path else node_id,
            path=path,
            nodes=nodes,
            edges=edges,
            total_potential=total_potential,
            entropy_evolution=entropy_evolution,
        )

    async def find_contradictions(
        self,
        node_id: str,
        graph_name: str = "default",
    ) -> List[str]:
        """Find nodes that contradict given node."""
        cursor = self.conn.cursor()

        # Find edges with CONTRADICTS relationship
        cursor.execute("""
            SELECT to_id FROM edges
            WHERE graph_name = ? AND from_id = ? AND relation_type = 'CONTRADICTS'
            UNION
            SELECT from_id FROM edges
            WHERE graph_name = ? AND to_id = ? AND relation_type = 'CONTRADICTS'
        """, (graph_name, node_id, graph_name, node_id))

        return [row['to_id'] if 'to_id' in row.keys() else row['from_id']
                for row in cursor.fetchall()]

    # ========================================================================
    # Utility
    # ========================================================================

    async def health_check(self) -> bool:
        """Check if backend is healthy."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _row_to_node(self, row: sqlite3.Row) -> GraphNode:
        """Convert database row to GraphNode."""
        return GraphNode(
            id=row['id'],
            content=row['content'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            fractal_signature=row['fractal_signature'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            parent_id=row['parent_id'],
            children_ids=json.loads(row['children_ids']) if row['children_ids'] else [],
            potential=row['potential'],
            entropy=row['entropy'],
            coherence=row['coherence'],
            phase=row['phase'],
        )

    def _row_to_edge(self, row: sqlite3.Row) -> GraphEdge:
        """Convert database row to GraphEdge."""
        return GraphEdge(
            from_id=row['from_id'],
            to_id=row['to_id'],
            relation_type=row['relation_type'],
            weight=row['weight'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            timestamp=datetime.fromisoformat(row['timestamp']),
        )
