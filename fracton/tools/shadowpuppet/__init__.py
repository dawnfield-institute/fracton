"""
ShadowPuppet: Architecture-as-Code Evolution Framework

A model-agnostic framework for evolving software through:
- Architecture-as-code (Python protocols define structure)
- Template-guided generation (parent code as context)
- Coherence evaluation (structural/semantic/energetic fitness)
- Genealogy tracking (full provenance trees)

The metaphor:
- Protocol is the puppet (structure, joints, constraints)
- Generator is the puppeteer (brings it to life)
- Shadow is the generated code (the projection that runs)
- Evolution selects better puppets (architecture improvement)

Works with any code generator:
- GitHub Copilot
- Claude Code / Claude API
- Local models
- Template fallback

Example:
    from fracton.tools.shadowpuppet import (
        SoftwareEvolution,
        ProtocolSpec,
        GrowthGap,
        CopilotGenerator,
        ClaudeGenerator
    )
    
    # Define architecture as Python protocols
    api_protocol = ProtocolSpec(
        name="APIRouter",
        methods=["get", "post", "put", "delete"],
        docstring="REST API router with CRUD operations",
        pac_invariants=["All routes return JSON", "Errors use standard HTTP codes"]
    )
    
    # Create evolution with your generator
    evolution = SoftwareEvolution(
        generator=CopilotGenerator(),
        coherence_threshold=0.75
    )
    
    # Evolve!
    results = evolution.grow([GrowthGap(protocol=api_protocol)])
"""

from .protocols import ProtocolSpec, GrowthGap, ComponentOrganism, TestSuite
from .coherence import CoherenceEvaluator
from .evolution import SoftwareEvolution, CodeEnvironment, EvolutionConfig
from .genealogy import GenealogyTree

# Generators
from .generators import (
    CodeGenerator,
    CopilotGenerator,
    ClaudeGenerator,
    MockGenerator
)

__all__ = [
    # Core types
    'ProtocolSpec',
    'GrowthGap', 
    'ComponentOrganism',
    'TestSuite',
    
    # Evaluation
    'CoherenceEvaluator',
    
    # Evolution
    'SoftwareEvolution',
    'CodeEnvironment',
    'EvolutionConfig',
    
    # Genealogy
    'GenealogyTree',
    
    # Generators
    'CodeGenerator',
    'CopilotGenerator',
    'ClaudeGenerator',
    'MockGenerator',
]

__version__ = '0.1.0'
