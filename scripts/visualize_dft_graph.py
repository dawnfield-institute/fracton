"""
Visualize DFT Knowledge Graph Structure
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')


def load_graph(graph_path: Path):
    """Load graph from JSON."""
    with open(graph_path, 'r') as f:
        return json.load(f)


def print_tree(data, node_id, prefix="", is_last=True, visited=None, max_depth=4, current_depth=0):
    """Print tree structure recursively."""
    if visited is None:
        visited = set()

    if node_id in visited or current_depth >= max_depth:
        return

    visited.add(node_id)
    node = data["nodes"][node_id]

    # Print current node
    connector = "└── " if is_last else "├── "
    print(f"{prefix}{connector}{node['name']} (d={node['actualization_depth']})")

    # Get children
    children = node.get("child_actualizations", [])
    if not children or current_depth >= max_depth - 1:
        return

    # Print children
    extension = "    " if is_last else "│   "
    for i, child_id in enumerate(children):
        is_last_child = (i == len(children) - 1)
        print_tree(data, child_id, prefix + extension, is_last_child, visited, max_depth, current_depth + 1)


def analyze_graph(data):
    """Analyze and print graph structure."""
    nodes = data["nodes"]
    edges = data["edges"]

    print("[GRAPH ANALYSIS]")
    print(f"Total Nodes: {len(nodes)}")
    print(f"Total Edges: {len(edges)}")
    print(f"Root Nodes: {len(data['roots'])}\n")

    # Depth distribution
    depth_dist = defaultdict(int)
    for node in nodes.values():
        depth_dist[node["actualization_depth"]] += 1

    print("[DEPTH DISTRIBUTION]")
    for depth in sorted(depth_dist.keys()):
        count = depth_dist[depth]
        bar = "█" * min(count, 50)
        print(f"  Depth {depth}: {count:2d} {bar}")

    # Confluence analysis
    print("\n[CONFLUENCE NODES] (Multiple parents)")
    confluence_nodes = [
        (n["name"], len(n["parent_potentials"]), n["actualization_depth"])
        for n in nodes.values()
        if len(n["parent_potentials"]) > 1
    ]
    confluence_nodes.sort(key=lambda x: x[2])  # Sort by depth

    for name, num_parents, depth in confluence_nodes:
        print(f"  {name} ({num_parents} parents, depth {depth})")

    # High-impact nodes (many children)
    print("\n[HIGH-IMPACT NODES] (Many children)")
    high_impact = [
        (n["name"], len(n["child_actualizations"]), n["actualization_depth"])
        for n in nodes.values()
        if len(n["child_actualizations"]) >= 5
    ]
    high_impact.sort(key=lambda x: x[1], reverse=True)

    for name, num_children, depth in high_impact[:10]:
        print(f"  {name}: {num_children} children (depth {depth})")

    # Leaf nodes (applications)
    print("\n[LEAF NODES] (Applications, no children)")
    leaves = [
        (n["name"], n["actualization_depth"])
        for n in nodes.values()
        if len(n["child_actualizations"]) == 0
    ]
    leaves.sort(key=lambda x: x[1])

    for name, depth in leaves[:15]:
        print(f"  {name} (depth {depth})")

    # Print tree from each root
    print("\n[TREE STRUCTURE] (First 4 levels)")
    for root_id in data["roots"]:
        print(f"\n{data['nodes'][root_id]['name']}:")
        visited = {root_id}
        children = data["nodes"][root_id].get("child_actualizations", [])
        for i, child_id in enumerate(children):
            is_last = (i == len(children) - 1)
            print_tree(data, child_id, "", is_last, visited.copy(), max_depth=4)


def find_path(data, from_id, to_id):
    """Find path between two concepts."""
    from collections import deque

    nodes = data["nodes"]
    queue = deque([(from_id, [from_id])])
    visited = {from_id}

    while queue:
        current_id, path = queue.popleft()

        if current_id == to_id:
            return path

        # Check children
        for child_id in nodes[current_id].get("child_actualizations", []):
            if child_id not in visited:
                visited.add(child_id)
                queue.append((child_id, path + [child_id]))

    return None


def interactive_exploration(data):
    """Interactive graph exploration."""
    print("\n[INTERACTIVE EXPLORATION]")
    print("Commands:")
    print("  path <from> <to> - Find path between concepts")
    print("  info <id>        - Show concept details")
    print("  quit             - Exit")

    while True:
        cmd = input("\n> ").strip()
        if not cmd:
            continue

        if cmd == "quit":
            break

        parts = cmd.split()
        if parts[0] == "path" and len(parts) == 3:
            path = find_path(data, parts[1], parts[2])
            if path:
                names = [data["nodes"][nid]["name"] for nid in path]
                print(f"Path: {' -> '.join(names)}")
            else:
                print("No path found")

        elif parts[0] == "info" and len(parts) == 2:
            node = data["nodes"].get(parts[1])
            if node:
                print(f"\n{node['name']}")
                print(f"Definition: {node['definition']}")
                print(f"Depth: {node['actualization_depth']}")
                print(f"Parents: {len(node['parent_potentials'])}")
                print(f"Children: {len(node['child_actualizations'])}")
            else:
                print("Node not found")


if __name__ == "__main__":
    graph_path = Path(__file__).parent.parent / "data" / "dft_knowledge_graph.json"
    print(f"Loading graph from {graph_path}...\n")

    data = load_graph(graph_path)
    analyze_graph(data)

    # Optional: Interactive mode
    # interactive_exploration(data)
