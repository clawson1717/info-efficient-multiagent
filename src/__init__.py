"""
Info-Efficient Multi-Agent Reasoning (IEMAR)

A multi-agent reasoning system that dynamically allocates compute and 
communication bandwidth based on each agent's measured information capacity.
"""

from .capacity import (
    InformationCapacityEstimator,
    CapacityResult,
    create_estimator,
)
from .agent import (
    AgentResponse,
    AgentConfig,
    ReasoningAgent,
    SpecializedAgent,
    create_agent,
    create_agent_pool,
)

__version__ = "0.1.0"

__all__ = [
    "InformationCapacityEstimator",
    "CapacityResult",
    "create_estimator",
    "AgentResponse",
    "AgentConfig",
    "ReasoningAgent",
    "SpecializedAgent",
    "create_agent",
    "create_agent_pool",
]
