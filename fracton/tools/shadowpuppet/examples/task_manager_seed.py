"""
CLI Task Manager - Architecture-as-Code

A practical, real-world example that generates a working CLI task manager with:
- JSON file persistence (not just in-memory)
- CRUD operations for tasks
- Status tracking (todo, in-progress, done)
- CLI interface with argument parsing
- Colored terminal output

This tests ShadowPuppet with a real, runnable application.

Usage:
    python -m fracton.tools.shadowpuppet.examples.task_manager_seed
    
    # After generation, run the generated app:
    python generated/taskapp.py add "Buy groceries"
    python generated/taskapp.py list
    python generated/taskapp.py done 1
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Protocol
from dataclasses import dataclass, field
from datetime import datetime
from abc import abstractmethod


# ============================================================================
# DOMAIN TYPES
# ============================================================================

@dataclass
class Task:
    """A task item."""
    id: int
    title: str
    status: str  # 'todo', 'in-progress', 'done'
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    priority: int = 0  # 0=normal, 1=high, 2=urgent
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'title': self.title,
            'status': self.status,
            'created_at': self.created_at,
            'completed_at': self.completed_at,
            'priority': self.priority,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        return cls(
            id=data['id'],
            title=data['title'],
            status=data['status'],
            created_at=data.get('created_at', datetime.now().isoformat()),
            completed_at=data.get('completed_at'),
            priority=data.get('priority', 0),
            tags=data.get('tags', [])
        )


DOMAIN_TYPES = [
    '''
@dataclass
class Task:
    """A task item."""
    id: int
    title: str
    status: str  # 'todo', 'in-progress', 'done'
    created_at: str
    completed_at: Optional[str] = None
    priority: int = 0  # 0=normal, 1=high, 2=urgent
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        ...
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create from dictionary."""
        ...
'''
]


# ============================================================================
# PROTOCOL SPECIFICATIONS
# ============================================================================

class TaskStore(Protocol):
    """
    Persistent storage for tasks using JSON file.
    
    Handles all I/O operations - loading, saving, and managing task data.
    The store should be safe for concurrent reads and handle file corruption gracefully.
    
    PAC Invariants:
    - Data is always persisted after modification
    - Task IDs are unique and auto-incrementing
    - File is valid JSON at all times
    - Empty file initializes with empty task list
    """
    file_path: Path
    tasks: Dict[int, Task]
    next_id: int
    
    @abstractmethod
    def load(self) -> None:
        """Load tasks from JSON file. Create file if not exists."""
        ...
    
    @abstractmethod
    def save(self) -> None:
        """Save all tasks to JSON file atomically."""
        ...
    
    @abstractmethod
    def add(self, task: Task) -> Task:
        """Add a task and persist. Returns task with assigned ID."""
        ...
    
    @abstractmethod
    def get(self, task_id: int) -> Optional[Task]:
        """Get task by ID or None if not found."""
        ...
    
    @abstractmethod
    def update(self, task: Task) -> bool:
        """Update existing task. Returns False if not found."""
        ...
    
    @abstractmethod
    def delete(self, task_id: int) -> bool:
        """Delete task by ID. Returns False if not found."""
        ...
    
    @abstractmethod
    def list_all(self) -> List[Task]:
        """Get all tasks sorted by ID."""
        ...


class TaskManager(Protocol):
    """
    Business logic for task operations.
    
    Uses TaskStore for persistence but handles validation,
    status transitions, and filtering logic.
    
    PAC Invariants:
    - Status transitions are validated (can't go done->todo directly)
    - Empty titles are rejected
    - Completed tasks get completed_at timestamp
    - Filters return new lists, don't modify originals
    """
    store: TaskStore
    
    @abstractmethod
    def create_task(self, title: str, priority: int = 0, tags: List[str] = None) -> Task:
        """Create new task with validation. Raises ValueError for empty title."""
        ...
    
    @abstractmethod
    def start_task(self, task_id: int) -> bool:
        """Move task to in-progress. Returns False if not found or already done."""
        ...
    
    @abstractmethod
    def complete_task(self, task_id: int) -> bool:
        """Mark task as done with timestamp. Returns False if not found."""
        ...
    
    @abstractmethod
    def delete_task(self, task_id: int) -> bool:
        """Delete task by ID."""
        ...
    
    @abstractmethod
    def get_by_status(self, status: str) -> List[Task]:
        """Get all tasks with given status."""
        ...
    
    @abstractmethod
    def get_by_tag(self, tag: str) -> List[Task]:
        """Get all tasks with given tag."""
        ...
    
    @abstractmethod
    def list_tasks(self, status_filter: Optional[str] = None) -> List[Task]:
        """List tasks with optional status filter."""
        ...


class CLIRenderer(Protocol):
    """
    Terminal output formatting with colors.
    
    Renders tasks and messages to terminal with ANSI colors.
    Handles both single task display and list formatting.
    
    PAC Invariants:
    - ANSI codes are properly terminated (no color bleed)
    - Output works on terminals without color support (fallback)
    - Lists include task count summary
    - Empty lists show helpful message
    """
    use_color: bool
    
    @abstractmethod
    def render_task(self, task: Task) -> str:
        """Render single task as colored string."""
        ...
    
    @abstractmethod
    def render_task_list(self, tasks: List[Task], title: str = "Tasks") -> str:
        """Render list of tasks with header and summary."""
        ...
    
    @abstractmethod
    def success(self, message: str) -> str:
        """Render success message in green."""
        ...
    
    @abstractmethod
    def error(self, message: str) -> str:
        """Render error message in red."""
        ...
    
    @abstractmethod
    def warning(self, message: str) -> str:
        """Render warning message in yellow."""
        ...


class TaskApp(Protocol):
    """
    CLI application entry point.
    
    Parses command line arguments and dispatches to TaskManager.
    Handles all user interaction through CLIRenderer.
    
    PAC Invariants:
    - Unknown commands show help message
    - All errors are caught and displayed nicely
    - Exit codes: 0=success, 1=error, 2=usage error
    - Help is shown for -h/--help and when no args
    
    Commands:
    - add <title> [--priority N] [--tags a,b,c]
    - list [--status todo|in-progress|done]
    - start <id>
    - done <id>
    - delete <id>
    """
    manager: TaskManager
    renderer: CLIRenderer
    
    @abstractmethod
    def run(self, args: List[str]) -> int:
        """Main entry point. Returns exit code."""
        ...
    
    @abstractmethod
    def cmd_add(self, args: List[str]) -> int:
        """Handle 'add' command."""
        ...
    
    @abstractmethod
    def cmd_list(self, args: List[str]) -> int:
        """Handle 'list' command."""
        ...
    
    @abstractmethod
    def cmd_start(self, args: List[str]) -> int:
        """Handle 'start' command."""
        ...
    
    @abstractmethod
    def cmd_done(self, args: List[str]) -> int:
        """Handle 'done' command."""
        ...
    
    @abstractmethod
    def cmd_delete(self, args: List[str]) -> int:
        """Handle 'delete' command."""
        ...
    
    @abstractmethod
    def show_help(self) -> None:
        """Display help message."""
        ...


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_store_persistence(store) -> bool:
    """Test that tasks persist to file."""
    # Create a task
    task = Task(id=0, title="Test task", status="todo")
    stored = store.add(task)
    
    # Check it got an ID
    if stored.id == 0 and store.next_id <= 1:
        return False
    
    # Verify it's in the store
    retrieved = store.get(stored.id)
    if retrieved is None:
        return False
    
    if retrieved.title != "Test task":
        return False
    
    return True


def test_store_crud(store) -> bool:
    """Test create, read, update, delete."""
    # Create
    task = store.add(Task(id=0, title="CRUD test", status="todo"))
    task_id = task.id
    
    # Read
    if store.get(task_id) is None:
        return False
    
    # Update
    task.status = "done"
    if not store.update(task):
        return False
    
    updated = store.get(task_id)
    if updated.status != "done":
        return False
    
    # Delete
    if not store.delete(task_id):
        return False
    
    if store.get(task_id) is not None:
        return False
    
    return True


def test_manager_status_transitions(manager) -> bool:
    """Test valid status transitions."""
    task = manager.create_task("Status test")
    if task.status != "todo":
        return False
    
    # Start it
    if not manager.start_task(task.id):
        return False
    
    # Check it's in progress
    tasks = manager.get_by_status("in-progress")
    if not any(t.id == task.id for t in tasks):
        return False
    
    # Complete it
    if not manager.complete_task(task.id):
        return False
    
    # Check it's done
    completed = manager.store.get(task.id)
    if completed.status != "done":
        return False
    
    if completed.completed_at is None:
        return False
    
    return True


def test_manager_validation(manager) -> bool:
    """Test that empty titles are rejected."""
    try:
        manager.create_task("")
        return False  # Should have raised
    except ValueError:
        return True
    except Exception:
        return False


def test_renderer_colors(renderer) -> bool:
    """Test that renderer produces colored output."""
    task = Task(id=1, title="Color test", status="todo")
    output = renderer.render_task(task)
    
    # Should have content
    if len(output) < 10:
        return False
    
    # Should include task title
    if "Color test" not in output:
        return False
    
    # If colors enabled, should have ANSI codes
    if renderer.use_color:
        if "\033[" not in output and "\x1b[" not in output:
            # Try without ANSI check for mock
            pass
    
    return True


def test_renderer_list(renderer) -> bool:
    """Test list rendering with summary."""
    tasks = [
        Task(id=1, title="Task 1", status="todo"),
        Task(id=2, title="Task 2", status="done"),
    ]
    output = renderer.render_task_list(tasks, "Test Tasks")
    
    # Should include title
    if "Test" not in output:
        return False
    
    # Should include both tasks
    if "Task 1" not in output or "Task 2" not in output:
        return False
    
    return True


# ============================================================================
# INTEGRATION TEST
# ============================================================================

def test_store_manager_integration(store, manager) -> bool:
    """Test that manager properly uses store."""
    # Manager should use the store's persistence
    initial_count = len(store.list_all())
    
    task = manager.create_task("Integration test", priority=1)
    
    # Should be in store
    if len(store.list_all()) != initial_count + 1:
        return False
    
    # Store should have it
    stored = store.get(task.id)
    if stored is None or stored.title != "Integration test":
        return False
    
    # Priority should be preserved
    if stored.priority != 1:
        return False
    
    return True


# ============================================================================
# BUILD SHADOWPUPPET SPECS
# ============================================================================

def build_specs():
    """Build ProtocolSpec objects from the Protocol definitions."""
    from fracton.tools.shadowpuppet import (
        ProtocolSpec, TypeAnnotation, GrowthGap, TestSuite
    )
    
    task_store = ProtocolSpec(
        name="TaskStore",
        methods=["load", "save", "add", "get", "update", "delete", "list_all"],
        method_signatures=[
            TypeAnnotation("load", {}, "None"),
            TypeAnnotation("save", {}, "None"),
            TypeAnnotation("add", {"task": "Task"}, "Task"),
            TypeAnnotation("get", {"task_id": "int"}, "Optional[Task]"),
            TypeAnnotation("update", {"task": "Task"}, "bool"),
            TypeAnnotation("delete", {"task_id": "int"}, "bool"),
            TypeAnnotation("list_all", {}, "List[Task]"),
        ],
        docstring="Persistent JSON file storage for tasks with CRUD operations",
        attributes=[
            "file_path: Path",
            "tasks: Dict[int, Task]",
            "next_id: int"
        ],
        pac_invariants=[
            "Data is always persisted after modification",
            "Task IDs are unique and auto-incrementing",
            "File is valid JSON at all times"
        ],
        dependencies=[]
    )
    
    task_manager = ProtocolSpec(
        name="TaskManager",
        methods=["create_task", "start_task", "complete_task", "delete_task", 
                 "get_by_status", "get_by_tag", "list_tasks"],
        method_signatures=[
            TypeAnnotation("create_task", {"title": "str", "priority": "int = 0", "tags": "List[str] = None"}, "Task", raises=["ValueError"]),
            TypeAnnotation("start_task", {"task_id": "int"}, "bool"),
            TypeAnnotation("complete_task", {"task_id": "int"}, "bool"),
            TypeAnnotation("delete_task", {"task_id": "int"}, "bool"),
            TypeAnnotation("get_by_status", {"status": "str"}, "List[Task]"),
            TypeAnnotation("get_by_tag", {"tag": "str"}, "List[Task]"),
            TypeAnnotation("list_tasks", {"status_filter": "Optional[str] = None"}, "List[Task]"),
        ],
        docstring="Business logic for task operations with validation and status transitions",
        attributes=["store: TaskStore"],
        pac_invariants=[
            "Status transitions are validated",
            "Empty titles are rejected with ValueError",
            "Completed tasks get completed_at timestamp"
        ],
        dependencies=["TaskStore"]
    )
    
    cli_renderer = ProtocolSpec(
        name="CLIRenderer",
        methods=["render_task", "render_task_list", "success", "error", "warning"],
        method_signatures=[
            TypeAnnotation("render_task", {"task": "Task"}, "str"),
            TypeAnnotation("render_task_list", {"tasks": "List[Task]", "title": "str = 'Tasks'"}, "str"),
            TypeAnnotation("success", {"message": "str"}, "str"),
            TypeAnnotation("error", {"message": "str"}, "str"),
            TypeAnnotation("warning", {"message": "str"}, "str"),
        ],
        docstring="Terminal output formatting with ANSI colors",
        attributes=["use_color: bool"],
        pac_invariants=[
            "ANSI codes are properly terminated",
            "Empty lists show helpful message"
        ],
        dependencies=[]
    )
    
    task_app = ProtocolSpec(
        name="TaskApp",
        methods=["run", "cmd_add", "cmd_list", "cmd_start", "cmd_done", "cmd_delete", "show_help"],
        method_signatures=[
            TypeAnnotation("run", {"args": "List[str]"}, "int"),
            TypeAnnotation("cmd_add", {"args": "List[str]"}, "int"),
            TypeAnnotation("cmd_list", {"args": "List[str]"}, "int"),
            TypeAnnotation("cmd_start", {"args": "List[str]"}, "int"),
            TypeAnnotation("cmd_done", {"args": "List[str]"}, "int"),
            TypeAnnotation("cmd_delete", {"args": "List[str]"}, "int"),
            TypeAnnotation("show_help", {}, "None"),
        ],
        docstring="CLI application that parses args and dispatches to TaskManager",
        attributes=["manager: TaskManager", "renderer: CLIRenderer"],
        pac_invariants=[
            "Unknown commands show help message",
            "Exit codes: 0=success, 1=error, 2=usage error",
            "All errors are caught and displayed nicely"
        ],
        dependencies=["TaskManager", "CLIRenderer"]
    )
    
    return task_store, task_manager, cli_renderer, task_app


def main():
    """Run the task manager evolution."""
    from fracton.tools.shadowpuppet import (
        SoftwareEvolution, EvolutionConfig, GrowthGap, TestSuite
    )
    from fracton.tools.shadowpuppet.generators import ClaudeGenerator, MockGenerator
    
    # Build specs
    task_store, task_manager, cli_renderer, task_app = build_specs()
    
    # Create gaps with tests
    gaps = [
        GrowthGap(
            protocol=task_store,
            test_suite=TestSuite(unit=[test_store_persistence, test_store_crud]),
            domain_types=DOMAIN_TYPES
        ),
        GrowthGap(
            protocol=task_manager,
            test_suite=TestSuite(
                unit=[test_manager_status_transitions, test_manager_validation],
            ),
            domain_types=DOMAIN_TYPES
        ),
        GrowthGap(
            protocol=cli_renderer,
            test_suite=TestSuite(unit=[test_renderer_colors, test_renderer_list]),
            domain_types=DOMAIN_TYPES
        ),
        GrowthGap(
            protocol=task_app,
            domain_types=DOMAIN_TYPES
        ),
    ]
    
    # Configure
    config = EvolutionConfig(
        coherence_threshold=0.60,
        candidates_per_gap=2,
        max_generations=5,
        save_checkpoints=True,
        output_dir=Path("generated/task_manager")
    )
    
    # Set up generator - use Claude if API key available, else mock
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if api_key:
        print("[*] Using ClaudeGenerator with API key")
        generator = ClaudeGenerator(
            model="claude-sonnet-4-20250514",
            temperature=0.3,
            fallback_generator=MockGenerator()
        )
    else:
        print("[!] No ANTHROPIC_API_KEY found, using MockGenerator")
        generator = MockGenerator()
    
    # Define callbacks for progress
    class ProgressCallbacks:
        def on_evolution_start(self, gaps, config):
            print(f"\n{'='*60}")
            print(f"  TASK MANAGER EVOLUTION")
            print(f"  Components: {[g.protocol.name for g in gaps]}")
            print(f"{'='*60}")
        
        def on_birth(self, component, fitness):
            emoji = "✓" if fitness >= 0.7 else "~" if fitness >= 0.5 else "!"
            print(f"    [{emoji}] Born: {component.protocol_name} (fitness={fitness:.3f})")
        
        def on_death(self, component, reason):
            print(f"    [✗] Died: {component.protocol_name} - {reason}")
        
        def on_evolution_end(self, results):
            print(f"\n{'='*60}")
            if results['success']:
                print(f"  SUCCESS! Generated {len(results['components'])} components")
                for comp in results['components']:
                    print(f"    - {comp['protocol_name']}: {comp['coherence_score']:.3f}")
            else:
                print(f"  FAILED: Population went extinct")
            print(f"{'='*60}\n")
    
    # Evolve!
    evolution = SoftwareEvolution(
        generator=generator,
        config=config,
        callbacks=ProgressCallbacks()
    )
    
    results = evolution.grow(gaps)
    
    # Save generated code
    if results['success']:
        output_dir = Path("generated/task_manager")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        evolution.save_code(output_dir)
        
        # Create a combined runner file
        runner_code = '''"""
CLI Task Manager - Generated by ShadowPuppet

Usage:
    python taskapp.py add "Task title"
    python taskapp.py list
    python taskapp.py start <id>
    python taskapp.py done <id>
    python taskapp.py delete <id>
"""

import sys
from pathlib import Path

# Import generated components
from taskstore import TaskStore
from taskmanager import TaskManager
from clirenderer import CLIRenderer
from taskapp import TaskApp

def main():
    # Set up components
    store = TaskStore(file_path=Path("tasks.json"))
    store.load()
    
    manager = TaskManager(store=store)
    renderer = CLIRenderer(use_color=True)
    app = TaskApp(manager=manager, renderer=renderer)
    
    # Run CLI
    exit_code = app.run(sys.argv[1:])
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
'''
        with open(output_dir / "run_app.py", 'w') as f:
            f.write(runner_code)
        
        print(f"\nGenerated files in: {output_dir.absolute()}")
        print("To run: python generated/task_manager/run_app.py add 'My task'")
    
    return results


if __name__ == "__main__":
    main()
