"""
Neo4j Graph Backend for KRONOS

Production-grade distributed graph database.
Perfect for large-scale, multi-user production systems.

Features:
- Distributed, highly available
- ACID transactions
- Cypher query language
- Scales to billions of nodes
- ~10K QPS for reads
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

import torch

try:
    from neo4j import AsyncGraphDatabase, AsyncDriver
    from neo4j.exceptions import Neo4jError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    AsyncGraphDatabase = None

from .base import (
    GraphBackend,
    GraphNode,
    GraphEdge,
    GraphNeighborhood,
    TemporalPath,
)

logger = logging.getLogger(__name__)


class Neo4jGraphBackend(GraphBackend):
    """
    Neo4j-based graph storage backend.

    Features:
    - Production-grade distributed database
    - Cypher query language
    - ACID transactions
    - High performance (~10K QPS)
    - Scales to billions of nodes

    Schema:
    - Nodes: (:Memory) with properties
    - Relationships: Typed edges with weights
    - Indexes: On id, fractal_signature, timestamp
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
    ):
        """
        Initialize Neo4j backend.

        Args:
            uri: Neo4j bolt URI
            user: Username
            password: Password
        """
        if not NEO4J_AVAILABLE:
            raise ImportError(
                "Neo4j driver not installed. Install with: pip install neo4j"
            )

        self.uri = uri
        self.user = user
        self.password = password
        self.driver: Optional[AsyncDriver] = None

        logger.info(f"Neo4jGraphBackend initialized: {uri}")

    async def connect(self) -> None:
        """Establish connection and create indexes."""
        self.driver = AsyncGraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password)
        )

        # Verify connectivity
        await self.driver.verify_connectivity()

        # Create indexes for performance
        await self._create_indexes()

        logger.info("Neo4j connection established")

    async def close(self) -> None:
        """Close connection."""
        if self.driver:
            await self.driver.close()
            self.driver = None
            logger.info("Neo4j connection closed")

    async def _create_indexes(self) -> None:
        """Create indexes for fast queries."""
        async with self.driver.session() as session:
            # Index on id (unique)
            try:
                await session.run("""
                    CREATE CONSTRAINT memory_id IF NOT EXISTS
                    FOR (m:Memory) REQUIRE m.id IS UNIQUE
                """)
            except Exception:
                pass  # Constraint may already exist

            # Index on fractal_signature
            try:
                await session.run("""
                    CREATE INDEX memory_signature IF NOT EXISTS
                    FOR (m:Memory) ON (m.fractal_signature)
                """)
            except Exception:
                pass

            # Index on timestamp
            try:
                await session.run("""
                    CREATE INDEX memory_timestamp IF NOT EXISTS
                    FOR (m:Memory) ON (m.timestamp)
                """)
            except Exception:
                pass

    # ========================================================================
    # Node Operations
    # ========================================================================

    async def create_node(
        self,
        node: GraphNode,
        graph_name: str = "default",
    ) -> None:
        """Create node in graph."""
        # Build properties dict
        props = {
            "id": node.id,
            "graph_name": graph_name,
            "content": node.content,
            "timestamp": node.timestamp.isoformat(),
            "fractal_signature": node.fractal_signature,
            "parent_id": node.parent_id,
            "children_ids": "|".join(node.children_ids),  # Store as pipe-separated string
            "potential": node.potential,
            "entropy": node.entropy,
            "coherence": node.coherence,
            "phase": node.phase,
        }

        # Add metadata
        for key, value in node.metadata.items():
            if isinstance(value, datetime):
                props[f"meta_{key}"] = value.isoformat()
            elif value is not None:
                props[f"meta_{key}"] = value

        # Build property string
        prop_items = ", ".join([f"{k}: ${k}" for k in props.keys()])

        async with self.driver.session() as session:
            await session.run(
                f"""
                CREATE (m:Memory {{
                    {prop_items}
                }})
                """,
                **props
            )

        logger.debug(f"Created node: {node.id} in graph {graph_name}")

    async def get_node(
        self,
        node_id: str,
        graph_name: str = "default",
    ) -> Optional[GraphNode]:
        """Retrieve node by ID."""
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (m:Memory {id: $id, graph_name: $graph})
                RETURN m
                """,
                id=node_id,
                graph=graph_name,
            )

            record = await result.single()
            if not record:
                return None

            node_dict = dict(record["m"])
            return self._dict_to_node(node_dict)

    async def update_node(
        self,
        node_id: str,
        updates: Dict[str, Any],
        graph_name: str = "default",
    ) -> None:
        """Update node metadata."""
        # Build SET clauses
        set_clauses = []
        params = {"id": node_id, "graph": graph_name}

        for key, value in updates.items():
            if key == "children_ids":
                params[key] = "|".join(value) if isinstance(value, list) else value
            elif isinstance(value, datetime):
                params[key] = value.isoformat()
            else:
                params[key] = value
            set_clauses.append(f"m.{key} = ${key}")

        if not set_clauses:
            return

        query = f"""
            MATCH (m:Memory {{id: $id, graph_name: $graph}})
            SET {', '.join(set_clauses)}
        """

        async with self.driver.session() as session:
            await session.run(query, **params)

    async def delete_node(
        self,
        node_id: str,
        graph_name: str = "default",
    ) -> None:
        """Delete node from graph."""
        async with self.driver.session() as session:
            # Delete node and all its relationships
            await session.run(
                """
                MATCH (m:Memory {id: $id, graph_name: $graph})
                DETACH DELETE m
                """,
                id=node_id,
                graph=graph_name,
            )

    # ========================================================================
    # Edge Operations
    # ========================================================================

    async def create_edge(
        self,
        edge: GraphEdge,
        graph_name: str = "default",
    ) -> None:
        """Create edge between nodes."""
        async with self.driver.session() as session:
            # Use dynamic relationship type from edge
            await session.run(
                f"""
                MATCH (from:Memory {{id: $from_id, graph_name: $graph}})
                MATCH (to:Memory {{id: $to_id, graph_name: $graph}})
                MERGE (from)-[r:{edge.relation_type}]->(to)
                SET r.weight = $weight,
                    r.timestamp = $timestamp,
                    r.graph_name = $graph
                """,
                from_id=edge.from_id,
                to_id=edge.to_id,
                weight=edge.weight,
                timestamp=edge.timestamp.isoformat(),
                graph=graph_name,
            )

        logger.debug(f"Created edge: {edge.from_id} -{edge.relation_type}-> {edge.to_id}")

    async def get_edges(
        self,
        node_id: str,
        direction: str = "out",
        relation_type: Optional[str] = None,
        graph_name: str = "default",
    ) -> List[GraphEdge]:
        """Get edges connected to node."""
        # Build relationship pattern
        if direction == "out":
            pattern = "(m)-[r]->(target)"
        elif direction == "in":
            pattern = "(target)-[r]->(m)"
        else:  # both
            pattern = "(m)-[r]-(target)"

        # Add type filter
        type_filter = f":{relation_type}" if relation_type else ""

        async with self.driver.session() as session:
            result = await session.run(
                f"""
                MATCH {pattern.replace('[r]', f'[r{type_filter}]')}
                WHERE m.id = $id AND m.graph_name = $graph
                RETURN r, target.id as target_id, startNode(r).id as from_id
                """,
                id=node_id,
                graph=graph_name,
            )

            edges = []
            async for record in result:
                rel = record["r"]
                edges.append(GraphEdge(
                    from_id=record["from_id"],
                    to_id=record["target_id"],
                    relation_type=rel.type,
                    weight=rel.get("weight", 1.0),
                    metadata={},
                    timestamp=datetime.fromisoformat(rel.get("timestamp", datetime.now().isoformat())),
                ))

            return edges

    async def delete_edge(
        self,
        from_id: str,
        to_id: str,
        relation_type: str,
        graph_name: str = "default",
    ) -> None:
        """Delete specific edge."""
        async with self.driver.session() as session:
            await session.run(
                f"""
                MATCH (from:Memory {{id: $from_id, graph_name: $graph}})-[r:{relation_type}]->(to:Memory {{id: $to_id, graph_name: $graph}})
                DELETE r
                """,
                from_id=from_id,
                to_id=to_id,
                graph=graph_name,
            )

    # ========================================================================
    # Graph Operations
    # ========================================================================

    async def create_graph(
        self,
        graph_name: str,
        description: str = "",
    ) -> None:
        """Create new named graph (metadata only in Neo4j)."""
        # Neo4j doesn't have explicit graph creation
        # We just ensure nodes have graph_name property
        logger.info(f"Graph '{graph_name}' will be created implicitly on first node")

    async def list_graphs(self) -> List[Dict[str, Any]]:
        """List all graphs."""
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (m:Memory)
                RETURN DISTINCT m.graph_name as name, count(m) as node_count
                ORDER BY name
            """)

            graphs = []
            async for record in result:
                graphs.append({
                    "name": record["name"],
                    "node_count": record["node_count"],
                    "description": "",
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
        # Build relationship type filter
        type_filter = ""
        if relation_types:
            type_filter = ":" + "|".join(relation_types)

        async with self.driver.session() as session:
            result = await session.run(
                f"""
                MATCH path = (center:Memory {{id: $id, graph_name: $graph}})-[r{type_filter}*1..{max_hops}]-(neighbor)
                WITH center, neighbor, relationships(path) as rels, length(path) as depth
                RETURN neighbor, rels, depth
                ORDER BY depth
                """,
                id=node_id,
                graph=graph_name,
            )

            nodes = []
            edges = []
            depth_map = {node_id: 0}

            async for record in result:
                neighbor_dict = dict(record["neighbor"])
                neighbor = self._dict_to_node(neighbor_dict)
                nodes.append(neighbor)
                depth_map[neighbor.id] = record["depth"]

                # Extract edges
                for rel in record["rels"]:
                    edges.append(GraphEdge(
                        from_id=rel.start_node["id"],
                        to_id=rel.end_node["id"],
                        relation_type=rel.type,
                        weight=rel.get("weight", 1.0),
                    ))

            return GraphNeighborhood(
                center_id=node_id,
                nodes=nodes,
                edges=edges,
                depth_map=depth_map,
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
        # Build pattern based on direction
        if direction == "forward":
            # Find descendants via parent_id
            query = f"""
            MATCH path = (m:Memory {{id: $id, graph_name: $graph}})<-[:PARENT_OF*1..{max_depth}]-(descendant)
            RETURN path
            ORDER BY length(path) DESC
            LIMIT 1
            """
        elif direction == "backward":
            # Find ancestors via parent_id
            query = f"""
            MATCH path = (ancestor)-[:PARENT_OF*1..{max_depth}]->(m:Memory {{id: $id, graph_name: $graph}})
            RETURN path
            ORDER BY length(path) DESC
            LIMIT 1
            """
        else:  # both
            query = f"""
            MATCH path = (ancestor)-[:PARENT_OF*1..{max_depth}]-(m:Memory {{id: $id, graph_name: $graph}})-[:PARENT_OF*1..{max_depth}]-(descendant)
            RETURN path
            ORDER BY length(path) DESC
            LIMIT 1
            """

        async with self.driver.session() as session:
            result = await session.run(query, id=node_id, graph=graph_name)
            record = await result.single()

            if not record:
                # No path found, return single node
                node = await self.get_node(node_id, graph_name)
                if node:
                    return TemporalPath(
                        root_id=node_id,
                        path=[node_id],
                        nodes=[node],
                        edges=[],
                        total_potential=node.potential,
                        entropy_evolution=[node.entropy],
                    )
                return TemporalPath(
                    root_id=node_id,
                    path=[],
                    nodes=[],
                    edges=[],
                    total_potential=0.0,
                    entropy_evolution=[],
                )

            path = record["path"]
            nodes = []
            edges = []
            path_ids = []
            entropy_evolution = []

            for node_dict in path.nodes:
                node = self._dict_to_node(dict(node_dict))
                nodes.append(node)
                path_ids.append(node.id)
                entropy_evolution.append(node.entropy)

            for rel in path.relationships:
                edges.append(GraphEdge(
                    from_id=rel.start_node["id"],
                    to_id=rel.end_node["id"],
                    relation_type=rel.type,
                    weight=rel.get("weight", 1.0),
                ))

            total_potential = sum(n.potential for n in nodes)

            return TemporalPath(
                root_id=path_ids[0] if path_ids else node_id,
                path=path_ids,
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
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (m:Memory {id: $id, graph_name: $graph})-[:CONTRADICTS]-(conflicting)
                RETURN conflicting.id as conflict_id
                """,
                id=node_id,
                graph=graph_name,
            )

            conflicts = []
            async for record in result:
                conflicts.append(record["conflict_id"])

            return conflicts

    # ========================================================================
    # Utility
    # ========================================================================

    async def health_check(self) -> bool:
        """Check if backend is healthy."""
        try:
            async with self.driver.session() as session:
                await session.run("RETURN 1")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _dict_to_node(self, node_dict: Dict[str, Any]) -> GraphNode:
        """Convert Neo4j node dict to GraphNode."""
        # Extract metadata (anything with meta_ prefix)
        metadata = {}
        for key, value in node_dict.items():
            if key.startswith("meta_"):
                metadata[key[5:]] = value

        # Parse children_ids
        children_str = node_dict.get("children_ids", "")
        children_ids = children_str.split("|") if children_str else []
        children_ids = [c for c in children_ids if c]  # Remove empty strings

        return GraphNode(
            id=node_dict["id"],
            content=node_dict["content"],
            timestamp=datetime.fromisoformat(node_dict["timestamp"]),
            fractal_signature=node_dict["fractal_signature"],
            metadata=metadata,
            parent_id=node_dict.get("parent_id", "-1"),
            children_ids=children_ids,
            potential=node_dict.get("potential", 1.0),
            entropy=node_dict.get("entropy", 0.5),
            coherence=node_dict.get("coherence", 0.5),
            phase=node_dict.get("phase", "STABLE"),
        )
