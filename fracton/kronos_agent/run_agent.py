#!/usr/bin/env python3
"""
Run the Kronos Agent interactively.

Usage:
    python -m kronos_agent.run_agent
    
Or from internal/:
    python kronos_agent/run_agent.py
"""

import asyncio
import sys
from pathlib import Path

from .agent import main

if __name__ == "__main__":
    asyncio.run(main())
