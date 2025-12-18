"""
Temporal Index - Time-Based Pattern Retrieval
==============================================

Maintains a temporal index for efficient time-range queries
on PAC nodes.

Index Structure:
    {
        "version": "1.0",
        "entries": [
            {"doc_id": "abc123", "timestamp": "2024-12-17T14:30:00Z", "node_id": 123},
            ...
        ],
        "by_date": {
            "2024-12-17": ["abc123", "def456", ...],
            ...
        }
    }
"""

import json
from pathlib import Path
from datetime import datetime, date
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict


class TemporalIndex:
    """
    Time-based index for PAC nodes.
    
    Provides:
    - Add entries with timestamp
    - Query by time range
    - Query by date
    - Efficient in-memory caching with disk persistence
    """
    
    def __init__(self, index_path: Path):
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory index
        self._entries: List[Dict[str, Any]] = []
        self._by_date: Dict[str, List[str]] = defaultdict(list)
        self._by_doc_id: Dict[str, Dict[str, Any]] = {}
        
        # Load existing
        self._load()
    
    def _load(self):
        """Load index from disk."""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    data = json.load(f)
                
                self._entries = data.get("entries", [])
                self._by_date = defaultdict(list, data.get("by_date", {}))
                
                # Rebuild doc_id lookup
                for entry in self._entries:
                    self._by_doc_id[entry["doc_id"]] = entry
                    
            except (json.JSONDecodeError, KeyError):
                # Corrupted index, start fresh
                self._entries = []
                self._by_date = defaultdict(list)
    
    def _save(self):
        """Save index to disk."""
        data = {
            "version": "1.0",
            "entries": self._entries,
            "by_date": dict(self._by_date),
        }
        
        with open(self.index_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def add(
        self, 
        doc_id: str, 
        node_id: int,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add entry to temporal index.
        
        Args:
            doc_id: FDO document ID
            node_id: PAC node ID
            timestamp: When created (default: now)
            metadata: Additional metadata
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        entry = {
            "doc_id": doc_id,
            "node_id": node_id,
            "timestamp": timestamp.isoformat(),
            **(metadata or {})
        }
        
        # Update indices
        self._entries.append(entry)
        self._by_doc_id[doc_id] = entry
        
        date_key = timestamp.date().isoformat()
        self._by_date[date_key].append(doc_id)
        
        # Persist
        self._save()
    
    def remove(self, doc_id: str) -> bool:
        """Remove entry from index."""
        if doc_id not in self._by_doc_id:
            return False
        
        entry = self._by_doc_id[doc_id]
        
        # Remove from entries list
        self._entries = [e for e in self._entries if e["doc_id"] != doc_id]
        
        # Remove from date index
        timestamp = datetime.fromisoformat(entry["timestamp"])
        date_key = timestamp.date().isoformat()
        if date_key in self._by_date:
            self._by_date[date_key] = [d for d in self._by_date[date_key] if d != doc_id]
        
        # Remove from doc_id lookup
        del self._by_doc_id[doc_id]
        
        self._save()
        return True
    
    def query_range(
        self, 
        start: datetime, 
        end: datetime
    ) -> List[Tuple[str, int]]:
        """
        Query entries in time range.
        
        Returns:
            List of (doc_id, node_id) tuples
        """
        results = []
        
        for entry in self._entries:
            ts = datetime.fromisoformat(entry["timestamp"])
            if start <= ts <= end:
                results.append((entry["doc_id"], entry["node_id"]))
        
        return results
    
    def query_date(self, query_date: date) -> List[str]:
        """Get all doc_ids for a specific date."""
        date_key = query_date.isoformat()
        return list(self._by_date.get(date_key, []))
    
    def query_recent(self, count: int = 10) -> List[Tuple[str, int]]:
        """Get N most recent entries."""
        sorted_entries = sorted(
            self._entries,
            key=lambda e: e["timestamp"],
            reverse=True
        )[:count]
        
        return [(e["doc_id"], e["node_id"]) for e in sorted_entries]
    
    def get_entry(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get full entry by doc_id."""
        return self._by_doc_id.get(doc_id)
    
    def __len__(self) -> int:
        return len(self._entries)
    
    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._by_doc_id
    
    def clear(self):
        """Clear all entries."""
        self._entries = []
        self._by_date = defaultdict(list)
        self._by_doc_id = {}
        self._save()
    
    def rebuild_from_entries(self, entries: List[Dict[str, Any]]):
        """Rebuild index from list of entries."""
        self._entries = entries
        self._by_date = defaultdict(list)
        self._by_doc_id = {}
        
        for entry in entries:
            self._by_doc_id[entry["doc_id"]] = entry
            timestamp = datetime.fromisoformat(entry["timestamp"])
            date_key = timestamp.date().isoformat()
            self._by_date[date_key].append(entry["doc_id"])
        
        self._save()
