"""
Session Graph Manager.

Manages ephemeral session graphs for context building.
Uses Kronos for structured short-term memory.

Supports serialization for caching between agent calls.
"""

import json
import uuid
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict


@dataclass
class SourceChunk:
    """A chunk of content with source lineage."""
    id: str
    content: str
    source_path: str
    source_id: str  # ID of node in long-term graph
    score: float
    graph: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    
    
@dataclass 
class SessionGraph:
    """
    In-memory session graph for context building.
    
    Lightweight - doesn't use full Kronos persistence.
    Just tracks chunks and their relationships for synthesis.
    """
    session_id: str
    chunks: Dict[str, SourceChunk] = field(default_factory=dict)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    
    def add_chunk(self, chunk: SourceChunk) -> str:
        """Add a chunk to the session graph."""
        self.chunks[chunk.id] = chunk
        return chunk.id
    
    def add_edge(
        self, 
        from_id: str, 
        to_id: str, 
        edge_type: str = "related",
        weight: float = 1.0,
    ) -> None:
        """Add a relationship between chunks."""
        self.edges.append({
            "from": from_id,
            "to": to_id,
            "type": edge_type,
            "weight": weight,
        })
    
    def get_sources(self) -> List[Dict[str, Any]]:
        """Get all unique source files referenced."""
        sources = {}
        for chunk in self.chunks.values():
            if chunk.source_path not in sources:
                sources[chunk.source_path] = {
                    "path": chunk.source_path,
                    "graph": chunk.graph,
                    "chunks": [],
                    "max_score": chunk.score,
                }
            sources[chunk.source_path]["chunks"].append(chunk.id)
            sources[chunk.source_path]["max_score"] = max(
                sources[chunk.source_path]["max_score"], 
                chunk.score
            )
        
        # Sort by relevance
        return sorted(
            sources.values(), 
            key=lambda x: x["max_score"], 
            reverse=True
        )
    
    def get_context_for_synthesis(self, max_chunks: int = 20) -> str:
        """
        Build context string for synthesis agent.
        
        Returns chunks ordered by score with source annotations.
        """
        # Sort chunks by score
        sorted_chunks = sorted(
            self.chunks.values(),
            key=lambda c: c.score,
            reverse=True
        )[:max_chunks]
        
        context_parts = []
        for chunk in sorted_chunks:
            source_ref = f"[{chunk.graph}:{chunk.source_path}"
            if chunk.line_start:
                source_ref += f"#L{chunk.line_start}"
                if chunk.line_end and chunk.line_end != chunk.line_start:
                    source_ref += f"-L{chunk.line_end}"
            source_ref += f"] (score: {chunk.score:.3f})"
            
            context_parts.append(f"{source_ref}\n{chunk.content}\n")
        
        return "\n---\n".join(context_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize session graph."""
        return {
            "session_id": self.session_id,
            "chunk_count": len(self.chunks),
            "edge_count": len(self.edges),
            "sources": self.get_sources(),
            "created_at": self.created_at,
        }
    
    def to_cache(self) -> Dict[str, Any]:
        """Full serialization for caching between agent calls."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "chunks": {
                cid: {
                    "id": c.id,
                    "content": c.content,
                    "source_path": c.source_path,
                    "source_id": c.source_id,
                    "score": c.score,
                    "graph": c.graph,
                    "line_start": c.line_start,
                    "line_end": c.line_end,
                }
                for cid, c in self.chunks.items()
            },
            "edges": self.edges,
        }
    
    @classmethod
    def from_cache(cls, data: Dict[str, Any]) -> "SessionGraph":
        """Restore session from cache."""
        session = cls(
            session_id=data["session_id"],
            created_at=data.get("created_at", time.time()),
        )
        for cid, chunk_data in data.get("chunks", {}).items():
            session.chunks[cid] = SourceChunk(**chunk_data)
        session.edges = data.get("edges", [])
        return session
    
    def save(self, path: Union[str, Path]) -> Path:
        """Save session cache to file."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_cache(), f, indent=2)
        return path
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "SessionGraph":
        """Load session from cache file."""
        with open(path, 'r') as f:
            return cls.from_cache(json.load(f))


class SessionManager:
    """Manages multiple session graphs."""
    
    def __init__(self):
        self.sessions: Dict[str, SessionGraph] = {}
    
    def create_session(self) -> SessionGraph:
        """Create a new session graph."""
        session_id = str(uuid.uuid4())[:8]
        session = SessionGraph(session_id=session_id)
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[SessionGraph]:
        """Get an existing session."""
        return self.sessions.get(session_id)
    
    def close_session(self, session_id: str) -> None:
        """Close and clean up a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def cleanup_old_sessions(self, max_age_seconds: float = 3600) -> int:
        """Remove sessions older than max_age."""
        now = time.time()
        old_sessions = [
            sid for sid, s in self.sessions.items()
            if now - s.created_at > max_age_seconds
        ]
        for sid in old_sessions:
            del self.sessions[sid]
        return len(old_sessions)
