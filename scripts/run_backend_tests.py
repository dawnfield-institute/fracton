#!/usr/bin/env python
"""
KRONOS Backend Test Runner

Runs comprehensive tests for all backend implementations with
automatic service detection and detailed reporting.

Usage:
    python scripts/run_backend_tests.py                    # Run all available tests
    python scripts/run_backend_tests.py --lightweight      # Only SQLite + ChromaDB
    python scripts/run_backend_tests.py --production       # Only Neo4j + Qdrant
    python scripts/run_backend_tests.py --fast             # Skip slow integration tests
    python scripts/run_backend_tests.py --coverage         # Generate coverage report
"""

import sys
import subprocess
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
import argparse


# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(80)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}\n")


def print_status(text: str, status: str):
    """Print status message."""
    if status == "PASS":
        symbol = f"{Colors.GREEN}✓{Colors.RESET}"
    elif status == "SKIP":
        symbol = f"{Colors.YELLOW}⊘{Colors.RESET}"
    elif status == "FAIL":
        symbol = f"{Colors.RED}✗{Colors.RESET}"
    else:
        symbol = f"{Colors.BLUE}•{Colors.RESET}"

    print(f"  {symbol} {text}")


def check_dependency(package: str) -> bool:
    """Check if a Python package is installed."""
    try:
        __import__(package)
        return True
    except ImportError:
        return False


async def check_neo4j() -> bool:
    """Check if Neo4j is running."""
    try:
        from neo4j import AsyncGraphDatabase
        driver = AsyncGraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password")
        )
        await driver.verify_connectivity()
        await driver.close()
        return True
    except:
        return False


async def check_qdrant() -> bool:
    """Check if Qdrant is running."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        client.get_collections()
        return True
    except:
        return False


def check_environment() -> Dict[str, bool]:
    """Check which backends are available."""
    print_header("Environment Check")

    env = {}

    # Required dependencies
    print("Checking required dependencies...")
    env['pytest'] = check_dependency('pytest')
    print_status(f"pytest", "PASS" if env['pytest'] else "FAIL")

    env['torch'] = check_dependency('torch')
    print_status(f"PyTorch", "PASS" if env['torch'] else "FAIL")

    # Lightweight backends (built-in)
    print("\nChecking lightweight backends...")
    env['sqlite'] = True  # Built-in to Python
    print_status("SQLite (built-in)", "PASS")

    env['chromadb'] = check_dependency('chromadb')
    print_status("ChromaDB", "PASS" if env['chromadb'] else "SKIP")

    # Production backends
    print("\nChecking production backends...")
    env['neo4j_driver'] = check_dependency('neo4j')
    print_status("Neo4j driver", "PASS" if env['neo4j_driver'] else "SKIP")

    env['qdrant_client'] = check_dependency('qdrant_client')
    print_status("Qdrant client", "PASS" if env['qdrant_client'] else "SKIP")

    # Check running services
    print("\nChecking running services...")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    env['neo4j_running'] = loop.run_until_complete(check_neo4j()) if env.get('neo4j_driver') else False
    print_status("Neo4j server (localhost:7687)", "PASS" if env['neo4j_running'] else "SKIP")

    env['qdrant_running'] = loop.run_until_complete(check_qdrant()) if env.get('qdrant_client') else False
    print_status("Qdrant server (localhost:6333)", "PASS" if env['qdrant_running'] else "SKIP")

    loop.close()

    return env


def run_tests(test_files: List[str], extra_args: List[str] = None) -> subprocess.CompletedProcess:
    """Run pytest with specified test files."""
    cmd = ["python", "-m", "pytest"] + test_files + ["-v"]

    if extra_args:
        cmd.extend(extra_args)

    return subprocess.run(cmd, capture_output=False)


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Run KRONOS backend tests")
    parser.add_argument("--lightweight", action="store_true", help="Only test lightweight backends")
    parser.add_argument("--production", action="store_true", help="Only test production backends")
    parser.add_argument("--fast", action="store_true", help="Skip slow integration tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Change to fracton directory
    repo_root = Path(__file__).parent.parent
    import os
    os.chdir(repo_root)

    # Check environment
    env = check_environment()

    if not env['pytest'] or not env['torch']:
        print(f"\n{Colors.RED}ERROR: Missing required dependencies!{Colors.RESET}")
        print("Install with: pip install pytest pytest-asyncio torch")
        return 1

    # Determine which tests to run
    print_header("Test Selection")

    test_files = []
    skipped = []

    # Base tests (always run)
    test_files.append("tests/backends/test_base.py")
    print_status("Base interface tests", "SELECTED")

    # Lightweight backends
    if not args.production:
        test_files.append("tests/backends/test_sqlite_graph.py")
        print_status("SQLite graph tests", "SELECTED")

        if env['chromadb']:
            test_files.append("tests/backends/test_chromadb_vectors.py")
            print_status("ChromaDB vector tests", "SELECTED")
        else:
            skipped.append("ChromaDB tests (package not installed)")
            print_status("ChromaDB vector tests", "SKIP")

    # Production backends
    if not args.lightweight:
        if env['neo4j_running']:
            test_files.append("tests/backends/test_neo4j_graph.py")
            print_status("Neo4j graph tests", "SELECTED")
        else:
            skipped.append("Neo4j tests (server not running)")
            print_status("Neo4j graph tests", "SKIP")

        if env['qdrant_running']:
            test_files.append("tests/backends/test_qdrant_vectors.py")
            print_status("Qdrant vector tests", "SELECTED")
        else:
            skipped.append("Qdrant tests (server not running)")
            print_status("Qdrant vector tests", "SKIP")

    # Integration tests
    if not args.fast and not args.lightweight and not args.production:
        test_files.append("tests/backends/test_backend_integration.py")
        print_status("Integration tests", "SELECTED")

    # Show skipped tests
    if skipped:
        print(f"\n{Colors.YELLOW}Skipped tests:{Colors.RESET}")
        for skip in skipped:
            print(f"  • {skip}")

    # Build pytest args
    extra_args = []

    if args.coverage:
        extra_args.extend(["--cov=fracton.storage.backends", "--cov-report=html", "--cov-report=term"])

    if args.verbose:
        extra_args.append("-vv")

    # Run tests
    print_header("Running Tests")

    print(f"Test command: pytest {' '.join(test_files)} {' '.join(extra_args)}\n")

    result = run_tests(test_files, extra_args)

    # Summary
    print_header("Test Summary")

    if result.returncode == 0:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ ALL TESTS PASSED{Colors.RESET}\n")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ SOME TESTS FAILED{Colors.RESET}\n")
        print(f"Exit code: {result.returncode}")
        return result.returncode


if __name__ == "__main__":
    sys.exit(main())
