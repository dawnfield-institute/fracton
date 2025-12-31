"""
Multi-Agent Pipeline.

SearchAgent → SessionGraph → SynthesisAgent → Response with citations
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from anthropic import Anthropic

from .config import config
from .tools import KronosTools
from .session import SessionManager, SessionGraph, SourceChunk

logger = logging.getLogger(__name__)


# =============================================================================
# Search Agent - Queries Kronos, populates session graph
# =============================================================================

SEARCH_SYSTEM_PROMPT = """You are a search specialist. Your job is to:
1. Analyze the user's query
2. Generate effective search queries
3. Review retrieved sources

You have access to knowledge graphs containing:
- dawn_theory: Research papers, experiments, theoretical foundations
- fracton: Implementation code, storage systems
- cip_core: Infrastructure, tools

When you receive search results, evaluate their relevance and identify the most important sources.
Be concise - just return the search queries to run."""


class SearchAgent:
    """
    Agent that searches Kronos and populates session graph with sources.
    """
    
    def __init__(
        self,
        tools: KronosTools,
        session_manager: SessionManager,
    ):
        self.tools = tools
        self.session_manager = session_manager
        self.client = Anthropic(api_key=config.anthropic_api_key)
        self.model = config.model
    
    async def search(
        self,
        query: str,
        session: SessionGraph,
        max_results: int = 15,
    ) -> Dict[str, Any]:
        """
        Search for relevant sources and populate session graph.
        
        Returns summary of what was found.
        """
        # Direct semantic search - no LLM needed for basic queries
        results = await self.tools.query_graph(
            query=query,
            graphs=["dawn_theory", "fracton", "cip_core"],
            limit=max_results,
        )
        
        if not results.get("success"):
            return {"error": results.get("error", "Search failed")}
        
        # Populate session graph with chunks
        chunks_added = 0
        for r in results.get("results", []):
            source = r.get("source", {})
            
            chunk = SourceChunk(
                id=r.get("id", f"chunk_{chunks_added}"),
                content=r.get("content_preview", ""),
                source_path=source.get("path") or source.get("full_path", "unknown"),
                source_id=r.get("id", ""),
                score=r.get("score", 0.0),
                graph=r.get("graph", "unknown"),
                line_start=source.get("line_start"),
                line_end=source.get("line_end"),
            )
            session.add_chunk(chunk)
            chunks_added += 1
        
        return {
            "success": True,
            "query": query,
            "chunks_added": chunks_added,
            "sources": session.get_sources(),
        }


# =============================================================================
# Synthesis Agent - Builds response from session graph
# =============================================================================

SYNTHESIS_SYSTEM_PROMPT = """You are a research synthesis specialist. Your job is to:
1. Review the provided source chunks carefully
2. Synthesize accurate, grounded responses
3. ALWAYS cite sources for claims

CRITICAL RULES:
- Only make claims supported by the provided sources
- If information is not in the sources, say "I don't have information about X in my sources"
- Include source citations in format: [graph:path#lines]
- Do not invent numbers, percentages, or specific claims not in sources
- If sources seem contradictory, note the contradiction

Format your response with:
1. Clear explanation of the topic
2. Specific details from sources WITH citations
3. Any caveats or limitations
4. Source list at the end"""


class SynthesisAgent:
    """
    Agent that synthesizes responses from session graph context.
    """
    
    def __init__(self):
        self.client = Anthropic(api_key=config.anthropic_api_key)
        self.model = config.model
    
    async def synthesize(
        self,
        query: str,
        session: SessionGraph,
        max_chunks: int = 15,
    ) -> str:
        """
        Synthesize a response from the session graph.
        
        Returns grounded response with citations.
        """
        # Build context from session graph
        context = session.get_context_for_synthesis(max_chunks=max_chunks)
        
        if not context:
            return "I couldn't find any relevant sources for your query."
        
        # Get source summary for the prompt
        sources = session.get_sources()
        source_list = "\n".join([
            f"- [{s['graph']}] {s['path']} ({len(s['chunks'])} chunks, score: {s['max_score']:.3f})"
            for s in sources[:10]
        ])
        
        # Build synthesis prompt
        user_prompt = f"""User Query: {query}

Available Sources:
{source_list}

Source Content:
{context}

---

Synthesize a comprehensive response to the user's query using ONLY the information from these sources. Cite each claim with the source reference."""

        # Call Claude for synthesis
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=SYNTHESIS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        
        return response.content[0].text


# =============================================================================
# Pipeline Orchestrator
# =============================================================================

class AgentPipeline:
    """
    Orchestrates the SearchAgent → SynthesisAgent pipeline.
    """
    
    def __init__(
        self,
        storage_path: Path,
        namespace: str = "dawn_models",
    ):
        self.tools = KronosTools(
            storage_path=storage_path,
            namespace=namespace,
            device=config.device,
        )
        self.session_manager = SessionManager()
        self.search_agent = SearchAgent(self.tools, self.session_manager)
        self.synthesis_agent = SynthesisAgent()
        self._connected = False
    
    async def initialize(self):
        """Initialize the pipeline."""
        await self.tools.connect()
        self._connected = True
        logger.info("AgentPipeline initialized")
    
    async def shutdown(self):
        """Shutdown the pipeline."""
        await self.tools.close()
        self._connected = False
    
    async def query(self, user_query: str) -> Dict[str, Any]:
        """
        Process a user query through the full pipeline.
        
        Returns response with metadata.
        """
        import time
        start = time.perf_counter()
        
        # Create session for this query
        session = self.session_manager.create_session()
        
        try:
            # Phase 1: Search
            search_start = time.perf_counter()
            search_result = await self.search_agent.search(
                query=user_query,
                session=session,
            )
            search_time = time.perf_counter() - search_start
            
            if not search_result.get("success"):
                return {
                    "response": f"Search failed: {search_result.get('error')}",
                    "session": session.to_dict(),
                    "timings": {"search": search_time},
                }
            
            # Phase 2: Synthesize
            synth_start = time.perf_counter()
            response = await self.synthesis_agent.synthesize(
                query=user_query,
                session=session,
            )
            synth_time = time.perf_counter() - synth_start
            
            total_time = time.perf_counter() - start
            
            return {
                "response": response,
                "session": session.to_dict(),
                "timings": {
                    "search": round(search_time, 3),
                    "synthesis": round(synth_time, 3),
                    "total": round(total_time, 3),
                },
            }
            
        finally:
            # Clean up session
            self.session_manager.close_session(session.session_id)
    
    async def list_graphs(self) -> Dict[str, Any]:
        """List available knowledge graphs."""
        return await self.tools.list_graphs()
