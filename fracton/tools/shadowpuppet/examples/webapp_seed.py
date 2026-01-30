"""
Web Application Architecture-as-Code

Defines a simple frontend + API architecture using Python protocols.
ShadowPuppet will generate implementations from these specifications.

Usage:
    python -m fracton.tools.shadowpuppet.examples.webapp_seed
"""

from typing import Dict, List, Any, Optional, Protocol
from dataclasses import dataclass
from abc import abstractmethod

# ============================================================================
# DOMAIN TYPES
# ============================================================================

@dataclass
class Request:
    """HTTP request representation."""
    method: str  # GET, POST, PUT, DELETE
    path: str
    headers: Dict[str, str]
    body: Optional[Dict[str, Any]] = None
    query_params: Optional[Dict[str, str]] = None


@dataclass
class Response:
    """HTTP response representation."""
    status_code: int
    headers: Dict[str, str]
    body: Any
    
    @staticmethod
    def json(data: Any, status: int = 200) -> 'Response':
        return Response(
            status_code=status,
            headers={'Content-Type': 'application/json'},
            body=data
        )
    
    @staticmethod
    def error(message: str, status: int = 400) -> 'Response':
        return Response.json({'error': message}, status)


@dataclass
class User:
    """User entity."""
    id: str
    username: str
    email: str
    created_at: str


# ============================================================================
# PROTOCOL SPECIFICATIONS (Architecture-as-Code)
# ============================================================================

class APIRouter(Protocol):
    """
    REST API router with CRUD operations.
    
    Handles HTTP routing and dispatches to appropriate handlers.
    Should support middleware, route parameters, and error handling.
    
    PAC Invariants:
    - All routes return Response objects
    - Errors use standard HTTP status codes
    - Route handlers are pure functions (no side effects in routing)
    """
    routes: Dict[str, callable]
    middleware: List[callable]
    
    @abstractmethod
    def get(self, path: str) -> callable:
        """Register GET route handler."""
        ...
    
    @abstractmethod
    def post(self, path: str) -> callable:
        """Register POST route handler."""
        ...
    
    @abstractmethod
    def put(self, path: str) -> callable:
        """Register PUT route handler."""
        ...
    
    @abstractmethod
    def delete(self, path: str) -> callable:
        """Register DELETE route handler."""
        ...
    
    @abstractmethod
    def handle(self, request: Request) -> Response:
        """Route request to appropriate handler."""
        ...


class UserService(Protocol):
    """
    User management service.
    
    Handles user CRUD operations with validation.
    Should integrate with storage backend.
    
    PAC Invariants:
    - User IDs are unique and immutable
    - Email addresses are validated before storage
    - Passwords are never stored in plaintext
    """
    
    @abstractmethod
    def create_user(self, username: str, email: str, password: str) -> User:
        """Create new user with validation."""
        ...
    
    @abstractmethod
    def get_user(self, user_id: str) -> Optional[User]:
        """Retrieve user by ID."""
        ...
    
    @abstractmethod
    def update_user(self, user_id: str, data: Dict[str, Any]) -> User:
        """Update user fields."""
        ...
    
    @abstractmethod
    def delete_user(self, user_id: str) -> bool:
        """Delete user by ID."""
        ...
    
    @abstractmethod
    def list_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        """List users with pagination."""
        ...


class TemplateRenderer(Protocol):
    """
    HTML template renderer for frontend.
    
    Renders templates with context data.
    Should support template inheritance and partials.
    
    PAC Invariants:
    - All output is properly HTML-escaped
    - Templates are cached after first load
    - Missing variables raise clear errors
    """
    template_dir: str
    cache: Dict[str, str]
    
    @abstractmethod
    def render(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render template with context."""
        ...
    
    @abstractmethod
    def load_template(self, template_name: str) -> str:
        """Load template from file."""
        ...
    
    @abstractmethod
    def escape_html(self, value: str) -> str:
        """Escape HTML special characters."""
        ...


class StaticFileServer(Protocol):
    """
    Static file server for frontend assets.
    
    Serves CSS, JS, images from static directory.
    Should handle caching headers and MIME types.
    
    PAC Invariants:
    - Only serves files from designated static directory
    - Sets appropriate Content-Type headers
    - Supports cache control headers
    """
    static_dir: str
    mime_types: Dict[str, str]
    
    @abstractmethod
    def serve(self, path: str) -> Response:
        """Serve static file."""
        ...
    
    @abstractmethod
    def get_mime_type(self, path: str) -> str:
        """Get MIME type for file."""
        ...


class WebApp(Protocol):
    """
    Main web application orchestrator.
    
    Combines API router, template renderer, and static server.
    Entry point for handling all HTTP requests.
    
    PAC Invariants:
    - All components share consistent configuration
    - Errors are logged and return appropriate responses
    - Requests are processed in order (no race conditions)
    """
    router: APIRouter
    renderer: TemplateRenderer
    static_server: StaticFileServer
    user_service: UserService
    
    @abstractmethod
    def handle_request(self, request: Request) -> Response:
        """Handle incoming HTTP request."""
        ...
    
    @abstractmethod
    def start(self, host: str, port: int) -> None:
        """Start the web server."""
        ...
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the web server."""
        ...


# ============================================================================
# WORLDSEED METADATA
# ============================================================================

WORLDSEED_METADATA = {
    'identity': {
        'purpose': 'Simple web application with frontend and API',
        'domain': 'Web development',
        'version': '1.0.0'
    },
    'pac_invariants': [
        'All HTTP responses have valid status codes',
        'User data is validated before storage',
        'Static files are served from designated directory only',
        'Templates escape HTML by default'
    ],
    'protocols': [
        'APIRouter',
        'UserService', 
        'TemplateRenderer',
        'StaticFileServer',
        'WebApp'
    ],
    'fibonacci_constraints': {
        'max_depth': 2,
        'max_components': 5
    }
}


# ============================================================================
# EVOLUTION RUNNER
# ============================================================================

def run_evolution():
    """Run ShadowPuppet evolution on this architecture."""
    from fracton.tools.shadowpuppet import (
        SoftwareEvolution,
        ProtocolSpec,
        GrowthGap,
        EvolutionConfig,
        MockGenerator
    )
    from pathlib import Path
    
    # Define protocols
    protocols = [
        ProtocolSpec(
            name="APIRouter",
            methods=["get", "post", "put", "delete", "handle"],
            docstring="REST API router with CRUD operations",
            attributes=["routes", "middleware"],
            pac_invariants=[
                "All routes return Response objects",
                "Errors use standard HTTP status codes"
            ]
        ),
        ProtocolSpec(
            name="UserService",
            methods=["create_user", "get_user", "update_user", "delete_user", "list_users"],
            docstring="User management service with validation",
            pac_invariants=[
                "User IDs are unique",
                "Email addresses are validated"
            ]
        ),
        ProtocolSpec(
            name="TemplateRenderer",
            methods=["render", "load_template", "escape_html"],
            docstring="HTML template renderer for frontend",
            attributes=["template_dir", "cache"],
            pac_invariants=[
                "All output is HTML-escaped",
                "Templates are cached"
            ]
        ),
        ProtocolSpec(
            name="StaticFileServer",
            methods=["serve", "get_mime_type"],
            docstring="Static file server for frontend assets",
            attributes=["static_dir", "mime_types"],
            pac_invariants=[
                "Only serves from static directory",
                "Sets appropriate Content-Type"
            ]
        ),
        ProtocolSpec(
            name="WebApp",
            methods=["handle_request", "start", "stop"],
            docstring="Main web application orchestrator",
            attributes=["router", "renderer", "static_server", "user_service"],
            pac_invariants=[
                "All components share configuration",
                "Errors are logged"
            ]
        ),
    ]
    
    # Create gaps
    gaps = [GrowthGap(protocol=p) for p in protocols]
    
    # Configure evolution
    config = EvolutionConfig(
        coherence_threshold=0.65,
        candidates_per_gap=2,
        max_generations=5,
        output_dir=Path("webapp_evolution")
    )
    
    # Run evolution (use MockGenerator for demo, swap for Claude/Copilot)
    evolution = SoftwareEvolution(
        generator=MockGenerator(),
        config=config,
        pac_invariants=WORLDSEED_METADATA['pac_invariants']
    )
    
    print("=" * 60)
    print("WebApp Architecture Evolution")
    print("=" * 60)
    
    results = evolution.grow(gaps)
    
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Success: {results['success']}")
    print(f"Generations: {results['generations']}")
    print(f"Final population: {results['final_population']}")
    
    if results['best_component']:
        print(f"\nBest component: {results['best_component']['id']}")
        print(f"Coherence: {results['best_component']['coherence_score']:.3f}")
    
    # Show genealogy
    print("\nGenealogy:")
    evolution.genealogy.print_tree()
    
    # Save generated code
    evolution.save_code(Path("webapp_generated"))
    
    return evolution


if __name__ == "__main__":
    run_evolution()
