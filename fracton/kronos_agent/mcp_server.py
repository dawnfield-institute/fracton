"""
MCP Server for Kronos Tools.

Exposes Kronos tools via Model Context Protocol for external access.
This is the next-generation CIP server.
"""

import asyncio
import json
import logging
from typing import Any
from pathlib import Path

# MCP SDK imports (you'll need: pip install mcp)
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Server = None

from .tools import KronosTools, TOOL_DEFINITIONS
from .config import config

logger = logging.getLogger(__name__)


class KronosMCPServer:
    """
    MCP Server exposing Kronos tools.
    
    Can be used by any MCP client (Claude Desktop, VS Code, etc.)
    """
    
    def __init__(
        self,
        storage_path: Path = None,
        namespace: str = "mcp",
    ):
        if not MCP_AVAILABLE:
            raise ImportError(
                "MCP SDK not installed. Run: pip install mcp"
            )
        
        self.tools = KronosTools(
            storage_path=storage_path or config.storage_path,
            namespace=namespace,
            device=config.device,
            embedding_dim=config.embedding_dim,
        )
        
        self.server = Server("kronos-mcp")
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup MCP request handlers."""
        
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """Return available tools."""
            return [
                Tool(
                    name=t["name"],
                    description=t["description"],
                    inputSchema=t["input_schema"],
                )
                for t in TOOL_DEFINITIONS
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            """Execute a tool."""
            
            # Ensure connected
            if not self.tools._connected:
                await self.tools.connect()
            
            # Execute tool
            if name == "scan_repo":
                result = await self.tools.scan_repo(**arguments)
            elif name == "query_graph":
                result = await self.tools.query_graph(**arguments)
            elif name == "ingest_diff":
                result = await self.tools.ingest_diff(**arguments)
            elif name == "trace_lineage":
                result = await self.tools.trace_lineage(**arguments)
            elif name == "get_context":
                result = await self.tools.get_context(**arguments)
            elif name == "list_graphs":
                result = await self.tools.list_graphs()
            elif name == "get_stats":
                result = await self.tools.get_stats()
            else:
                result = {"error": f"Unknown tool: {name}"}
            
            return [TextContent(
                type="text",
                text=json.dumps(result, default=str, indent=2)
            )]
    
    async def run(self):
        """Run the MCP server."""
        logger.info("Starting Kronos MCP Server...")
        
        # Connect to Kronos
        await self.tools.connect()
        
        try:
            # Run server (stdio mode for now)
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options()
                )
        finally:
            await self.tools.close()


async def main():
    """Run the MCP server."""
    logging.basicConfig(level=logging.INFO)
    
    server = KronosMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
