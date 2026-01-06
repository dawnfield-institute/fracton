"""
Kronos Tools for LangGraph Agent.

Provides tools that wrap Kronos memory operations for use by the agent.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
import asyncio
import logging

# Import from parent package (fracton.storage)
from ..storage import KronosMemory, NodeType, RelationType

logger = logging.getLogger(__name__)


class KronosTools:
    """
    Tool implementations for Kronos operations.
    
    These are the actual implementations called by the agent.
    """
    
    def __init__(
        self,
        storage_path: Path,
        namespace: str = "agent",
        device: str = "cpu",
        embedding_dim: int = 384,
    ):
        self.storage_path = Path(storage_path)
        self.namespace = namespace
        self.device = device
        self.embedding_dim = embedding_dim
        self._memory: Optional[KronosMemory] = None
        self._connected = False
        
    async def connect(self) -> bool:
        """Initialize Kronos memory connection."""
        if self._connected:
            return True
            
        try:
            self._memory = KronosMemory(
                storage_path=self.storage_path,
                namespace=self.namespace,
                device=self.device,
                embedding_dim=self.embedding_dim,
                embedding_model="mini",
            )
            await self._memory.connect()
            self._connected = True
            logger.info(f"Connected to Kronos at {self.storage_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    async def close(self):
        """Close Kronos connection."""
        if self._memory:
            await self._memory.close()
            self._connected = False
    
    # =========================================================================
    # Tool: scan_repo
    # =========================================================================
    
    async def scan_repo(
        self,
        repo_path: str,
        graph_name: str = "repo",
        file_patterns: List[str] = None,
        build_pac_tree: bool = True,
    ) -> Dict[str, Any]:
        """
        Scan a repository and create a knowledge graph with PAC tree structure.
        
        When build_pac_tree=True (default):
        - Creates repo root node
        - Creates directory nodes as children of root/parent dirs
        - Creates file nodes as children of directories
        - Uses PAC delta encoding between parent/child nodes
        
        Args:
            repo_path: Path to the repository
            graph_name: Name for the knowledge graph
            file_patterns: File patterns to include (e.g., ["*.py", "*.md"])
            build_pac_tree: If True, build hierarchical PAC tree structure
            
        Returns:
            Summary of scanned content
        """
        if not self._connected:
            await self.connect()
            
        repo = Path(repo_path)
        if not repo.exists():
            return {"error": f"Repository not found: {repo_path}"}
        
        # Default patterns
        if file_patterns is None:
            file_patterns = ["*.py", "*.md", "*.yaml", "*.json"]
        
        # Create graph
        await self._memory.create_graph(
            graph_name, 
            f"Knowledge graph for {repo.name}"
        )
        
        nodes_created = 0
        files_scanned = 0
        dirs_created = 0
        
        # Track directory nodes for PAC tree building
        dir_node_ids: Dict[str, str] = {}  # relative_dir_path -> node_id
        
        if build_pac_tree:
            # Create repository root node
            repo_readme = ""
            readme_path = repo / "README.md"
            if readme_path.exists():
                try:
                    repo_readme = readme_path.read_text(encoding="utf-8", errors="ignore")[:500]
                except:
                    pass
            
            root_content = f"Repository: {repo.name}\n\n{repo_readme}"
            root_id = await self._memory.store(
                content=root_content,
                graph=graph_name,
                node_type=NodeType.CONCEPT,
                parent_id=None,  # Root node
                metadata={
                    "source_type": "repository",
                    "source_path": "",
                    "source_full_path": str(repo),
                    "source_repo": str(repo),
                    "is_root": True,
                }
            )
            dir_node_ids[""] = root_id
            nodes_created += 1
            logger.info(f"Created PAC root node: {root_id[:8]}...")
        
        # Collect all files first, then process by directory
        all_files = []
        for pattern in file_patterns:
            for file_path in repo.rglob(pattern):
                # Skip common ignore patterns
                if any(p in str(file_path) for p in [
                    "__pycache__", ".git", "node_modules", ".egg-info",
                    "htmlcov", ".pytest_cache", ".venv", "venv",
                    ".changelog", ".cip", ".spec"
                ]):
                    continue
                all_files.append(file_path)
        
        # Sort by path depth to ensure parents are created first
        all_files.sort(key=lambda p: len(p.relative_to(repo).parts))
        
        for file_path in all_files:
            try:
                rel_path = file_path.relative_to(repo)
                rel_dir = str(rel_path.parent) if rel_path.parent != Path('.') else ""
                
                # Ensure directory hierarchy exists (PAC tree)
                if build_pac_tree and rel_dir and rel_dir not in dir_node_ids:
                    # Create directory nodes for each level
                    parts = rel_path.parent.parts
                    current_path = ""
                    
                    for i, part in enumerate(parts):
                        current_path = str(Path(*parts[:i+1]))
                        
                        if current_path not in dir_node_ids:
                            # Get parent directory node
                            parent_path = str(Path(*parts[:i])) if i > 0 else ""
                            parent_id = dir_node_ids.get(parent_path, root_id)
                            
                            # Create directory node as child of parent
                            dir_content = f"Directory: {part}\nPath: {current_path}"
                            
                            # Check for directory-level README
                            dir_readme_path = repo / current_path / "README.md"
                            if dir_readme_path.exists():
                                try:
                                    dir_readme = dir_readme_path.read_text(
                                        encoding="utf-8", errors="ignore"
                                    )[:300]
                                    dir_content += f"\n\n{dir_readme}"
                                except:
                                    pass
                            
                            dir_node_id = await self._memory.store(
                                content=dir_content,
                                graph=graph_name,
                                node_type=NodeType.CONCEPT,
                                parent_id=parent_id,  # PAC parent relationship!
                                metadata={
                                    "source_type": "directory",
                                    "source_path": current_path,
                                    "source_full_path": str(repo / current_path),
                                    "source_repo": str(repo),
                                    "is_directory": True,
                                }
                            )
                            dir_node_ids[current_path] = dir_node_id
                            dirs_created += 1
                            nodes_created += 1
                
                # Read file content
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                
                # Determine node type
                suffix = file_path.suffix.lower()
                if suffix == ".py":
                    node_type = NodeType.PROCEDURE
                elif suffix == ".md":
                    node_type = NodeType.DOCUMENT
                else:
                    node_type = NodeType.FACT
                
                # Get parent node (directory or root)
                parent_id = None
                if build_pac_tree:
                    parent_id = dir_node_ids.get(rel_dir, dir_node_ids.get(""))
                
                # Store summary/description, not full content
                line_count = content.count('\n') + 1
                preview = content[:500] + "..." if len(content) > 500 else content
                
                node_id = await self._memory.store(
                    content=preview,
                    graph=graph_name,
                    node_type=node_type,
                    parent_id=parent_id,  # PAC parent relationship!
                    metadata={
                        "source_type": "file",
                        "source_path": str(rel_path),
                        "source_full_path": str(file_path),
                        "source_repo": str(repo),
                        "file_extension": suffix,
                        "line_start": 1,
                        "line_end": line_count,
                        "content_size": len(content),
                        "can_open": True,
                        "uri": f"file://{file_path}",
                    }
                )
                
                nodes_created += 1
                files_scanned += 1
                
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
                continue
        
        # Save source path index for incremental updates
        await self._save_source_index(graph_name, repo)
        
        logger.info(
            f"Scanned {repo.name}: {files_scanned} files, "
            f"{dirs_created} dirs, {nodes_created} nodes (PAC tree={build_pac_tree})"
        )
        
        return {
            "success": True,
            "graph_name": graph_name,
            "files_scanned": files_scanned,
            "directories_created": dirs_created,
            "nodes_created": nodes_created,
            "repo_path": str(repo),
        }
    
    # =========================================================================
    # Incremental Update System
    # =========================================================================
    
    def _get_index_path(self, graph_name: str) -> Path:
        """Get path to source index file for a graph."""
        return self.storage_path / f"{graph_name}_source_index.json"
    
    async def _save_source_index(self, graph_name: str, repo: Path) -> None:
        """
        Build and save source path → node_id index.
        
        This enables O(1) lookup when updating files.
        """
        index = {
            "graph": graph_name,
            "repo_path": str(repo.resolve()),  # Always store absolute path
            "git_commit": self._get_git_head(repo),
            "files": {}  # source_path → node_id
        }
        
        # Query all nodes and build index from metadata
        # This is a one-time build after full scan
        try:
            results = await self._memory.query(
                query_text="*",  # Get all
                graphs=[graph_name],
                limit=10000,  # High limit
            )
            
            for r in results:
                metadata = getattr(r.node, '_user_metadata', {}) or {}
                if hasattr(r.node, 'metadata'):
                    metadata = r.node.metadata
                source_path = metadata.get("source_path")
                if source_path:
                    index["files"][source_path] = r.node.id
                    
        except Exception as e:
            logger.warning(f"Failed to build index from query: {e}")
        
        # Save index
        index_path = self._get_index_path(graph_name)
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        
        logger.info(f"Saved source index: {len(index['files'])} files")
    
    def _load_source_index(self, graph_name: str) -> Optional[Dict[str, Any]]:
        """Load source path index if it exists."""
        index_path = self._get_index_path(graph_name)
        if index_path.exists():
            with open(index_path, 'r') as f:
                return json.load(f)
        return None
    
    def _get_git_head(self, repo: Path) -> Optional[str]:
        """Get current HEAD commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:12]
        except Exception:
            pass
        return None
    
    def _get_changed_files(
        self, 
        repo: Path, 
        base_commit: str,
        file_patterns: List[str] = None,
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Get files changed between base_commit and HEAD.
        
        Returns:
            Tuple of (added, modified, deleted) file sets
        """
        if file_patterns is None:
            file_patterns = ["*.py", "*.md", "*.yaml", "*.json"]
        
        added = set()
        modified = set()
        deleted = set()
        
        try:
            # Get diff summary
            result = subprocess.run(
                ["git", "diff", "--name-status", base_commit, "HEAD"],
                cwd=repo,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
            )
            
            if result.returncode != 0:
                logger.warning(f"Git diff failed: {result.stderr}")
                return added, modified, deleted
            
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                    
                status, file_path = parts[0], parts[-1]
                
                # Check if file matches patterns
                matches = any(
                    Path(file_path).match(p) for p in file_patterns
                )
                if not matches:
                    continue
                
                # Skip ignored paths
                if any(p in file_path for p in [
                    "__pycache__", ".git", "node_modules", ".egg-info",
                    "htmlcov", ".pytest_cache", ".venv", "venv"
                ]):
                    continue
                
                if status.startswith('A'):
                    added.add(file_path)
                elif status.startswith('M'):
                    modified.add(file_path)
                elif status.startswith('D'):
                    deleted.add(file_path)
                elif status.startswith('R'):
                    # Rename: old path deleted, new path added
                    old_path = parts[1] if len(parts) > 2 else None
                    new_path = parts[-1]
                    if old_path:
                        deleted.add(old_path)
                    added.add(new_path)
                    
        except Exception as e:
            logger.error(f"Failed to get changed files: {e}")
        
        return added, modified, deleted
    
    async def update_repo(
        self,
        repo_path: str,
        graph_name: str,
        file_patterns: List[str] = None,
        force_full: bool = False,
    ) -> Dict[str, Any]:
        """
        Incrementally update a repository graph.
        
        Only re-embeds files that changed since last index.
        
        Args:
            repo_path: Path to the repository
            graph_name: Name of the graph to update
            file_patterns: File patterns to include
            force_full: If True, do full re-scan instead of incremental
            
        Returns:
            Summary of updates
        """
        if not self._connected:
            await self.connect()
        
        repo = Path(repo_path)
        if not repo.exists():
            return {"error": f"Repository not found: {repo_path}"}
        
        if file_patterns is None:
            file_patterns = ["*.py", "*.md", "*.yaml", "*.json"]
        
        # Load existing index
        index = self._load_source_index(graph_name)
        
        # If no index or force_full, do full scan
        if index is None or force_full:
            logger.info(f"No index found, doing full scan of {repo_path}")
            return await self.scan_repo(repo_path, graph_name, file_patterns)
        
        base_commit = index.get("git_commit")
        current_commit = self._get_git_head(repo)
        
        if base_commit == current_commit:
            return {
                "success": True,
                "message": "Already up to date",
                "graph_name": graph_name,
                "commit": current_commit,
            }
        
        # Get changed files
        added, modified, deleted = self._get_changed_files(
            repo, base_commit, file_patterns
        )
        
        logger.info(f"Changes since {base_commit}: +{len(added)} ~{len(modified)} -{len(deleted)}")
        
        nodes_created = 0
        nodes_updated = 0
        nodes_deleted = 0
        
        # Process deletions
        for file_path in deleted:
            node_id = index["files"].get(file_path)
            if node_id:
                try:
                    await self._memory.graph_backend.delete_node(node_id, graph_name)
                    del index["files"][file_path]
                    nodes_deleted += 1
                except Exception as e:
                    logger.warning(f"Failed to delete node for {file_path}: {e}")
        
        # Process additions and modifications
        for file_path in added | modified:
            full_path = repo / file_path
            if not full_path.exists():
                continue
                
            try:
                content = full_path.read_text(encoding="utf-8", errors="ignore")
                
                # Determine node type
                suffix = full_path.suffix.lower()
                if suffix == ".py":
                    node_type = NodeType.PROCEDURE
                elif suffix == ".md":
                    node_type = NodeType.DOCUMENT
                else:
                    node_type = NodeType.FACT
                
                line_count = content.count('\n') + 1
                preview = content[:500] + "..." if len(content) > 500 else content
                
                existing_node_id = index["files"].get(file_path)
                
                if existing_node_id and file_path in modified:
                    # Update existing node
                    await self._memory.update_node(
                        graph=graph_name,
                        node_id=existing_node_id,
                        updates={
                            "content": preview,
                            "line_end": line_count,
                            "content_size": len(content),
                        }
                    )
                    nodes_updated += 1
                else:
                    # Create new node
                    node_id = await self._memory.store(
                        content=preview,
                        graph=graph_name,
                        node_type=node_type,
                        metadata={
                            "source_type": "file",
                            "source_path": file_path,
                            "source_full_path": str(full_path),
                            "source_repo": str(repo),
                            "file_extension": suffix,
                            "line_start": 1,
                            "line_end": line_count,
                            "content_size": len(content),
                            "can_open": True,
                            "uri": f"file://{full_path}",
                        }
                    )
                    index["files"][file_path] = node_id
                    nodes_created += 1
                    
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
        
        # Update index with new commit
        index["git_commit"] = current_commit
        
        # Save updated index
        index_path = self._get_index_path(graph_name)
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        
        return {
            "success": True,
            "graph_name": graph_name,
            "base_commit": base_commit,
            "current_commit": current_commit,
            "nodes_created": nodes_created,
            "nodes_updated": nodes_updated,
            "nodes_deleted": nodes_deleted,
            "total_files": len(index["files"]),
        }
    
    async def get_graph_status(self, graph_name: str) -> Dict[str, Any]:
        """
        Get status of a graph including commit tracking.
        
        Returns:
            Graph metadata and sync status
        """
        index = self._load_source_index(graph_name)
        
        if index is None:
            return {
                "graph_name": graph_name,
                "indexed": False,
                "message": "No source index found. Run scan_repo first.",
            }
        
        repo = Path(index.get("repo_path", ""))
        current_commit = self._get_git_head(repo) if repo.exists() else None
        indexed_commit = index.get("git_commit")
        
        return {
            "graph_name": graph_name,
            "indexed": True,
            "repo_path": index.get("repo_path"),
            "indexed_commit": indexed_commit,
            "current_commit": current_commit,
            "in_sync": indexed_commit == current_commit,
            "file_count": len(index.get("files", {})),
        }

    # =========================================================================
    # Tool: query_graph
    # =========================================================================
    
    async def query_graph(
        self,
        query: str,
        graphs: List[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Query the knowledge graph using SEC resonance.
        
        Args:
            query: Natural language query
            graphs: Graphs to search (None = all)
            limit: Maximum results
            
        Returns:
            Matching nodes with scores
        """
        if not self._connected:
            await self.connect()
        
        if graphs is None:
            # Get all graphs from backend
            graph_list = await self._memory.list_graphs()
            graphs = [g.get("name", g) if isinstance(g, dict) else g for g in graph_list]
        
        try:
            results = await self._memory.query(
                query_text=query,
                graphs=graphs,
                limit=limit,
            )
            
            # Format results with source locations prominently displayed
            formatted_results = []
            for r in results:
                # Get metadata if available
                metadata = getattr(r.node, '_user_metadata', {}) or r.node.metadata if hasattr(r.node, 'metadata') else {}
                
                result_entry = {
                    "id": r.node.id,
                    "content_preview": r.node.content[:300],  # Short preview
                    "score": round(r.score, 4),
                    "similarity": round(r.similarity, 4),
                    "graph": r.node.graph,
                    "node_type": r.node.node_type.value if hasattr(r.node.node_type, 'value') else str(r.node.node_type),
                    # Source location - the key data
                    "source": {
                        "type": metadata.get("source_type", "unknown"),
                        "path": metadata.get("source_path", None),
                        "full_path": metadata.get("source_full_path", None),
                        "line_start": metadata.get("line_start", None),
                        "line_end": metadata.get("line_end", None),
                        "uri": metadata.get("uri", None),
                    }
                }
                formatted_results.append(result_entry)
            
            return {
                "success": True,
                "query": query,
                "results": formatted_results,
                "count": len(formatted_results),
            }
        except Exception as e:
            return {"error": str(e)}
    
    # =========================================================================
    # Tool: ingest_diff
    # =========================================================================
    
    async def ingest_diff(
        self,
        diff_content: str,
        graph_name: str = "commits",
        commit_hash: str = None,
        commit_message: str = None,
    ) -> Dict[str, Any]:
        """
        Ingest a git diff and add to knowledge graph.
        
        Args:
            diff_content: The git diff content
            graph_name: Graph to add to
            commit_hash: Optional commit hash
            commit_message: Optional commit message
            
        Returns:
            Summary of ingested changes
        """
        if not self._connected:
            await self.connect()
        
        # Ensure graph exists (create_graph is idempotent)
        try:
            await self._memory.create_graph(graph_name, "Git commit history")
        except Exception:
            pass  # Graph may already exist
        
        # Parse diff into chunks (simplified)
        files_changed = []
        current_file = None
        chunks = []
        
        for line in diff_content.split('\n'):
            if line.startswith('diff --git'):
                if current_file:
                    chunks.append({
                        "file": current_file,
                        "content": '\n'.join(files_changed[-20:]) if files_changed else ""
                    })
                parts = line.split(' ')
                if len(parts) >= 4:
                    current_file = parts[2].lstrip('a/')
                files_changed = []
            elif line.startswith(('+', '-')) and not line.startswith(('+++', '---')):
                files_changed.append(line)
        
        # Add final chunk
        if current_file:
            chunks.append({
                "file": current_file,
                "content": '\n'.join(files_changed[-20:]) if files_changed else ""
            })
        
        # Create commit node
        commit_content = f"Commit: {commit_hash or 'unknown'}\n"
        commit_content += f"Message: {commit_message or 'No message'}\n"
        commit_content += f"Files changed: {len(chunks)}\n\n"
        commit_content += diff_content[:3000]  # Include truncated diff
        
        commit_id = await self._memory.store(
            content=commit_content,
            graph=graph_name,
            node_type=NodeType.COMMIT,
            metadata={
                # Source location for commits
                "source_type": "commit",
                "commit_hash": commit_hash,
                "commit_message": commit_message,
                "files_changed": [c["file"] for c in chunks],
                "files_count": len(chunks),
                # For GitHub linking
                "can_link": True if commit_hash else False,
            }
        )
        
        return {
            "success": True,
            "commit_id": commit_id,
            "files_parsed": len(chunks),
            "graph": graph_name,
        }
    
    # =========================================================================
    # Tool: trace_lineage
    # =========================================================================
    
    async def trace_lineage(
        self,
        node_id: str,
        graph: str,
        direction: str = "both",
    ) -> Dict[str, Any]:
        """
        Trace the evolution lineage of a node.
        
        Args:
            node_id: The node to trace from
            graph: The graph containing the node
            direction: "forward", "backward", or "both"
            
        Returns:
            Lineage path with entropy evolution
        """
        if not self._connected:
            await self.connect()
        
        try:
            trace = await self._memory.trace_evolution(
                graph=graph,
                node_id=node_id,
                direction=direction,
            )
            return {
                "success": True,
                "trace": trace,
            }
        except Exception as e:
            return {"error": str(e)}
    
    # =========================================================================
    # Tool: get_context
    # =========================================================================
    
    async def get_context(
        self,
        query: str,
        graphs: List[str] = None,
        max_tokens: int = 4000,
    ) -> Dict[str, Any]:
        """
        Get relevant context for a query (for RAG).
        
        Args:
            query: The query to find context for
            graphs: Graphs to search
            max_tokens: Approximate max tokens to return
            
        Returns:
            Concatenated relevant context
        """
        if not self._connected:
            await self.connect()
        
        results = await self.query_graph(query, graphs, limit=10)
        
        if "error" in results:
            return results
        
        # Build context string with source references
        context_parts = []
        sources = []
        total_chars = 0
        char_limit = max_tokens * 4  # Rough token -> char estimate
        
        for r in results["results"]:
            content = r["content_preview"]
            source = r.get("source", {})
            
            if total_chars + len(content) > char_limit:
                break
            
            # Include source path in context
            source_ref = source.get("path", r["id"][:8])
            context_parts.append(f"[Source: {source_ref}]\n{content}")
            total_chars += len(content)
            
            # Track sources for reference
            sources.append({
                "path": source.get("path"),
                "full_path": source.get("full_path"),
                "type": source.get("type"),
                "lines": f"{source.get('line_start', '?')}-{source.get('line_end', '?')}",
            })
        
        return {
            "success": True,
            "query": query,
            "context": "\n\n---\n\n".join(context_parts),
            "sources": sources,
            "source_count": len(sources),
        }
    
    # =========================================================================
    # Tool: list_graphs
    # =========================================================================
    
    async def list_graphs(self) -> Dict[str, Any]:
        """List all available knowledge graphs."""
        if not self._connected:
            await self.connect()
        
        graphs = await self._memory.list_graphs()
        return {
            "success": True,
            "graphs": graphs,
        }
    
    # =========================================================================
    # Tool: get_stats
    # =========================================================================
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Kronos memory statistics."""
        if not self._connected:
            await self.connect()
        
        stats = await self._memory.get_stats()
        return {
            "success": True,
            "stats": stats,
        }
    
    # =========================================================================
    # Tool: add_source
    # =========================================================================
    
    async def add_source(
        self,
        content: str,
        graph: str,
        source_type: str,
        source_path: str,
        title: str = None,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Add a source (paper, URL, document) to the knowledge graph.
        
        The key insight: nodes are POINTERS to sources, not the full content.
        This enables search to return actual source locations.
        
        Args:
            content: Brief description or abstract (for embedding)
            graph: Graph to add to
            source_type: "paper", "url", "document", "code", "data"
            source_path: Path or URL to the actual source
            title: Optional title
            metadata: Additional metadata
            
        Returns:
            Created node info
        """
        if not self._connected:
            await self.connect()
        
        # Ensure graph exists by creating it (create_graph is idempotent in most backends)
        try:
            await self._memory.create_graph(graph, f"Sources: {graph}")
        except Exception:
            pass  # Graph may already exist
        
        # Determine node type
        type_map = {
            "paper": NodeType.PAPER,
            "url": NodeType.DOCUMENT,
            "document": NodeType.DOCUMENT,
            "code": NodeType.PROCEDURE,
            "data": NodeType.FACT,
        }
        node_type = type_map.get(source_type, NodeType.FACT)
        
        # Build source metadata
        source_metadata = {
            "source_type": source_type,
            "source_path": source_path,
            "title": title or source_path,
            "can_open": source_type in ["file", "url"],
        }
        
        # Merge with additional metadata
        if metadata:
            source_metadata.update(metadata)
        
        # For papers, try to extract DOI
        if source_type == "paper" and "doi" not in source_metadata:
            # Check if source_path looks like a DOI
            if "10." in source_path:
                source_metadata["doi"] = source_path
                source_metadata["uri"] = f"https://doi.org/{source_path}"
        
        # For URLs, store the full URI
        if source_type == "url":
            source_metadata["uri"] = source_path
        
        node_id = await self._memory.store(
            content=content,
            graph=graph,
            node_type=node_type,
            metadata=source_metadata,
        )
        
        return {
            "success": True,
            "node_id": node_id,
            "graph": graph,
            "source_type": source_type,
            "source_path": source_path,
        }


# Tool definitions for LangGraph/Anthropic
TOOL_DEFINITIONS = [
    {
        "name": "scan_repo",
        "description": "Scan a repository and create a knowledge graph from its contents. Use this when asked to learn about or analyze a codebase.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Path to the repository to scan"
                },
                "graph_name": {
                    "type": "string",
                    "description": "Name for the knowledge graph (default: 'repo')"
                },
                "file_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File patterns to include (e.g., ['*.py', '*.md'])"
                }
            },
            "required": ["repo_path"]
        }
    },
    {
        "name": "query_graph",
        "description": "Query the knowledge graph to find relevant information. Use this to answer questions about ingested content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query"
                },
                "graphs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific graphs to search (optional, searches all if not specified)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return (default: 10)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "ingest_diff",
        "description": "Ingest a git diff and add it to the knowledge graph. Use this to learn about code changes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "diff_content": {
                    "type": "string",
                    "description": "The git diff content"
                },
                "graph_name": {
                    "type": "string",
                    "description": "Graph to add to (default: 'commits')"
                },
                "commit_hash": {
                    "type": "string",
                    "description": "The commit hash"
                },
                "commit_message": {
                    "type": "string",
                    "description": "The commit message"
                }
            },
            "required": ["diff_content"]
        }
    },
    {
        "name": "trace_lineage",
        "description": "Trace the evolution lineage of a node to see how ideas evolved.",
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "The node ID to trace from"
                },
                "graph": {
                    "type": "string",
                    "description": "The graph containing the node"
                },
                "direction": {
                    "type": "string",
                    "enum": ["forward", "backward", "both"],
                    "description": "Direction to trace (default: 'both')"
                }
            },
            "required": ["node_id", "graph"]
        }
    },
    {
        "name": "get_context",
        "description": "Get relevant context from the knowledge graph for a query. Useful for answering questions with supporting information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to find context for"
                },
                "graphs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Graphs to search"
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Approximate max tokens to return (default: 4000)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "list_graphs",
        "description": "List all available knowledge graphs.",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "get_stats",
        "description": "Get statistics about the Kronos memory system.",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "add_source",
        "description": "Add a source (paper, URL, document) to the knowledge graph. Nodes are pointers to sources, enabling search to return actual source locations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Brief description or abstract (used for semantic embedding)"
                },
                "graph": {
                    "type": "string",
                    "description": "Graph to add the source to"
                },
                "source_type": {
                    "type": "string",
                    "enum": ["paper", "url", "document", "code", "data"],
                    "description": "Type of source"
                },
                "source_path": {
                    "type": "string",
                    "description": "Path, URL, or DOI to the actual source"
                },
                "title": {
                    "type": "string",
                    "description": "Optional title for the source"
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional metadata (e.g., author, date, tags)"
                }
            },
            "required": ["content", "graph", "source_type", "source_path"]
        }
    }
]
