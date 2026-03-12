"""
E-Commerce Multi-Seed Architecture Example

A realistic microservice-style system with three seeds:
1. UserService - Authentication and user management
2. ProductService - Product catalog and inventory
3. OrderService - Order processing (depends on both User + Product)

This demonstrates:
- Cross-seed dependencies
- Public interface contracts
- Integration tests spanning services
- Evolutionary coordination across service boundaries

Usage:
    python -m fracton.tools.shadowpuppet.examples.ecommerce_multiseed
    
After evolution:
    generated/multi_seed/
    ├── userservice/
    │   ├── userrepository.py
    │   ├── authservice.py
    │   └── interfaces.json
    ├── productservice/
    │   ├── productrepository.py
    │   ├── inventoryservice.py
    │   └── interfaces.json
    ├── orderservice/
    │   ├── orderrepository.py
    │   ├── orderprocessor.py
    │   └── interfaces.json
    └── connectors.json
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from fracton.tools.shadowpuppet.multiseed import (
    SeedArchitecture,
    MultiSeedEvolution,
    SeedInterface
)
from fracton.tools.shadowpuppet.protocols import ProtocolSpec, GrowthGap
from fracton.tools.shadowpuppet.evolution import EvolutionConfig
from fracton.tools.shadowpuppet.generators import ClaudeGenerator, MockGenerator


# ============================================================================
# DOMAIN TYPES (shared across all services)
# ============================================================================

DOMAIN_TYPES = [
    '''
@dataclass
class User:
    """User entity."""
    id: str
    email: str
    name: str
    password_hash: str
    created_at: str
    role: str = "customer"  # customer, admin
''',
    '''
@dataclass
class Product:
    """Product entity."""
    id: str
    name: str
    description: str
    price: float
    stock: int
    category: str
    created_at: str
''',
    '''
@dataclass
class Order:
    """Order entity."""
    id: str
    user_id: str
    items: List[Dict[str, Any]]  # [{product_id, quantity, price}]
    total: float
    status: str  # pending, confirmed, shipped, delivered, cancelled
    created_at: str
    updated_at: Optional[str] = None
'''
]


# ============================================================================
# SEED 1: USER SERVICE
# ============================================================================

def create_user_service_seed() -> SeedArchitecture:
    """
    User management service.
    
    Exposed interfaces:
    - UserRepository: CRUD for users
    - AuthService: Login, token validation
    
    No external dependencies (foundational service).
    """
    
    # Protocol: UserRepository
    user_repo_protocol = ProtocolSpec(
        name="UserRepository",
        methods=[
            "create_user",
            "get_user",
            "get_user_by_email",
            "update_user",
            "delete_user",
            "list_users"
        ],
        docstring="User persistence with CRUD operations",
        attributes=[
            "users: Dict[str, User]",
            "email_index: Dict[str, str]"
        ],
        pac_invariants=[
            "User IDs are unique and immutable",
            "Email addresses are unique across users",
            "Passwords are never stored in plaintext"
        ],
        dependencies=[]
    )
    
    # Protocol: AuthService
    auth_protocol = ProtocolSpec(
        name="AuthService",
        methods=[
            "login",
            "validate_token",
            "hash_password",
            "verify_password",
            "generate_token"
        ],
        docstring="Authentication and authorization service",
        attributes=[
            "user_repo: UserRepository",
            "secret_key: str"
        ],
        pac_invariants=[
            "Tokens expire after defined period",
            "Failed login attempts are rate-limited",
            "Passwords are hashed with salt"
        ],
        dependencies=["UserRepository"]
    )
    
    # Tests
    def test_user_creation(repo):
        """Test user creation and retrieval."""
        user = repo.create_user("test@example.com", "Test User", "password123")
        retrieved = repo.get_user(user.id)
        return retrieved is not None and retrieved.email == "test@example.com"
    
    def test_auth_flow(auth):
        """Test login and token validation."""
        token = auth.login("test@example.com", "password123")
        is_valid = auth.validate_token(token)
        return is_valid
    
    # Build seed
    return SeedArchitecture(
        name="UserService",
        gaps=[
            GrowthGap(
                protocol=user_repo_protocol,
                domain_types=DOMAIN_TYPES,
                test_suite=None  # TODO: Add TestSuite wrapper
            ),
            GrowthGap(
                protocol=auth_protocol,
                domain_types=DOMAIN_TYPES
            )
        ],
        exposed_interfaces=["UserRepository", "AuthService"],
        dependencies={},  # No external dependencies
        pac_invariants=[
            "Total users conserved across operations",
            "Authentication must use UserRepository only"
        ]
    )


# ============================================================================
# SEED 2: PRODUCT SERVICE
# ============================================================================

def create_product_service_seed() -> SeedArchitecture:
    """
    Product catalog and inventory service.
    
    Exposed interfaces:
    - ProductRepository: CRUD for products
    - InventoryService: Stock management
    
    No external dependencies (independent domain).
    """
    
    # Protocol: ProductRepository
    product_repo_protocol = ProtocolSpec(
        name="ProductRepository",
        methods=[
            "create_product",
            "get_product",
            "update_product",
            "delete_product",
            "list_products",
            "search_products"
        ],
        docstring="Product persistence and search",
        attributes=[
            "products: Dict[str, Product]",
            "category_index: Dict[str, List[str]]"
        ],
        pac_invariants=[
            "Product IDs are unique",
            "Prices are always non-negative",
            "Stock quantities are non-negative integers"
        ],
        dependencies=[]
    )
    
    # Protocol: InventoryService
    inventory_protocol = ProtocolSpec(
        name="InventoryService",
        methods=[
            "check_stock",
            "reserve_stock",
            "release_stock",
            "update_stock",
            "get_low_stock_products"
        ],
        docstring="Inventory management and stock tracking",
        attributes=[
            "product_repo: ProductRepository",
            "reservations: Dict[str, int]"
        ],
        pac_invariants=[
            "Stock conservation: reserved + available = total",
            "Stock cannot go negative",
            "Reservations must be released or committed"
        ],
        dependencies=["ProductRepository"]
    )
    
    return SeedArchitecture(
        name="ProductService",
        gaps=[
            GrowthGap(
                protocol=product_repo_protocol,
                domain_types=DOMAIN_TYPES
            ),
            GrowthGap(
                protocol=inventory_protocol,
                domain_types=DOMAIN_TYPES
            )
        ],
        exposed_interfaces=["ProductRepository", "InventoryService"],
        dependencies={},  # Independent service
        pac_invariants=[
            "Total product count conserved",
            "Stock levels consistent across operations"
        ]
    )


# ============================================================================
# SEED 3: ORDER SERVICE
# ============================================================================

def create_order_service_seed() -> SeedArchitecture:
    """
    Order processing service.
    
    Exposed interfaces:
    - OrderRepository: CRUD for orders
    - OrderProcessor: Order workflow
    
    External dependencies:
    - UserService.AuthService: Validate user tokens
    - ProductService.InventoryService: Reserve stock
    
    This is the integration layer that coordinates across services.
    """
    
    # Protocol: OrderRepository
    order_repo_protocol = ProtocolSpec(
        name="OrderRepository",
        methods=[
            "create_order",
            "get_order",
            "update_order",
            "cancel_order",
            "list_user_orders",
            "list_orders_by_status"
        ],
        docstring="Order persistence and queries",
        attributes=[
            "orders: Dict[str, Order]",
            "user_index: Dict[str, List[str]]"
        ],
        pac_invariants=[
            "Order IDs are unique",
            "Order totals match sum of item prices",
            "Cancelled orders cannot be modified"
        ],
        dependencies=[]
    )
    
    # Protocol: OrderProcessor
    order_processor_protocol = ProtocolSpec(
        name="OrderProcessor",
        methods=[
            "create_order",
            "confirm_order",
            "ship_order",
            "deliver_order",
            "cancel_order",
            "validate_order"
        ],
        docstring="Order workflow and business logic",
        attributes=[
            "order_repo: OrderRepository",
            "auth_service: AuthService",  # External from UserService
            "inventory_service: InventoryService"  # External from ProductService
        ],
        pac_invariants=[
            "Orders progress through valid state transitions only",
            "Stock is reserved when order created",
            "Stock is released when order cancelled",
            "User must be authenticated to create order"
        ],
        dependencies=["OrderRepository", "AuthService", "InventoryService"]
    )
    
    return SeedArchitecture(
        name="OrderService",
        gaps=[
            GrowthGap(
                protocol=order_repo_protocol,
                domain_types=DOMAIN_TYPES
            ),
            GrowthGap(
                protocol=order_processor_protocol,
                domain_types=DOMAIN_TYPES
            )
        ],
        exposed_interfaces=["OrderRepository", "OrderProcessor"],
        dependencies={
            "UserService": ["AuthService"],
            "ProductService": ["InventoryService"]
        },
        pac_invariants=[
            "Total order value conserved across updates",
            "Stock reservations match order items"
        ],
        cross_seed_tests=[
            # These would test the full workflow across all three services
            lambda: test_full_order_flow()
        ]
    )


def test_full_order_flow():
    """
    Integration test spanning all three services.
    
    Flow:
    1. UserService: Create user and login
    2. ProductService: Create product with stock
    3. OrderService: Create order (validates user, reserves stock)
    4. OrderService: Confirm order
    5. Check: Stock reduced, order confirmed
    """
    # TODO: Implement actual multi-service test
    # Would need to instantiate all evolved components
    return True


# ============================================================================
# MAIN EVOLUTION
# ============================================================================

def main():
    """Run multi-seed evolution."""
    
    print("="*70)
    print("E-Commerce Multi-Seed Architecture Evolution")
    print("="*70)
    
    # Create seeds
    user_seed = create_user_service_seed()
    product_seed = create_product_service_seed()
    order_seed = create_order_service_seed()
    
    print(f"\n[*] Defined {len([user_seed, product_seed, order_seed])} seeds:")
    print(f"    1. {user_seed.name}: {len(user_seed.gaps)} components")
    print(f"    2. {product_seed.name}: {len(product_seed.gaps)} components")
    print(f"    3. {order_seed.name}: {len(order_seed.gaps)} components (depends on 1 & 2)")
    
    # Configure evolution
    config = EvolutionConfig(
        coherence_threshold=0.65,
        candidates_per_gap=2,  # Fewer candidates for faster demo
        max_generations=5,
        save_checkpoints=True
    )
    
    # Use MockGenerator for demo (swap with ClaudeGenerator for real evolution)
    generator = MockGenerator()
    # generator = ClaudeGenerator(
    #     model="claude-sonnet-4-20250514",
    #     fallback_generator=MockGenerator()
    # )
    
    # Create multi-seed evolution orchestrator
    multi_evolution = MultiSeedEvolution(
        seeds=[user_seed, product_seed, order_seed],
        generator=generator,
        global_config=config,
        output_dir=Path("generated/multi_seed")
    )
    
    # Evolve!
    print("\n[*] Starting multi-seed evolution...")
    results = multi_evolution.evolve(
        max_generations=5,
        cross_seed_iterations=2  # Two passes to refine cross-seed contracts
    )
    
    # Print results
    print("\n" + "="*70)
    print("Evolution Complete!")
    print("="*70)
    
    stats = results['stats']
    print(f"\nTotal Seeds: {stats.total_seeds}")
    print(f"Total Components: {stats.total_components}")
    print(f"Total Generations: {stats.total_generations}")
    
    print("\nPer-Seed Results:")
    for seed_name, seed_stat in stats.seed_stats.items():
        print(f"  {seed_name}:")
        print(f"    Population: {seed_stat.population}")
        print(f"    Best Coherence: {seed_stat.max_coherence:.3f}")
        print(f"    Mean Coherence: {seed_stat.mean_coherence:.3f}")
    
    print(f"\nCross-Seed Connectors: {len(results['connectors'])}")
    for connector in results['connectors']:
        print(f"  {connector.provider_seed} -> {connector.consumer_seed}")
        for iface in connector.interfaces:
            print(f"    - {iface.component_name} ({len(iface.methods)} methods)")
    
    print(f"\nGenerated code saved to: {multi_evolution.output_dir}")
    
    # Show file structure
    print("\nDirectory structure:")
    for seed_name in results['evolution_order']:
        seed = results['seeds'][seed_name]
        print(f"  {seed_name}/")
        for component in seed.components:
            print(f"    ├── {component.protocol_name.lower()}.py")
        if seed.public_interfaces:
            print(f"    └── interfaces.json")
    
    return results


if __name__ == "__main__":
    results = main()
