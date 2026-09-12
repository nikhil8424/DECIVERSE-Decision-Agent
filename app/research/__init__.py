"""Research-oriented architecture for DECIVERSE - AI-Powered Social Decision Simulator.

This module provides the research framework layers built on top of the foundational
MiroFish/OASIS simulation technology. The research contribution is the decision-oriented
framework, scenario comparison, social impact analysis, and decision intelligence layer.

Foundational Technologies:
- MiroFish: Multi-agent social simulation framework
- OASIS: Social media simulation environment (camel-oasis==0.2.5, camel-ai==0.2.78)

Research Contribution:
- Community Digital Twin framework
- Scenario-based decision simulation
- Emergent behaviour analysis
- Social impact evaluation
- Multi-scenario decision comparison
- Decision intelligence with uncertainty quantification
"""

from .community_context_engine import CommunityContextEngine
from .stakeholder_digital_twin import StakeholderDigitalTwin
from .scenario_designer import ScenarioDesigner, ScenarioManager
from .emergent_behaviour_analyzer import EmergentBehaviourAnalyzer
from .social_impact_model import SocialImpactModel, SocialImpactResult
from .decision_comparison_engine import DecisionComparisonEngine
from .decision_intelligence_engine import DecisionIntelligenceEngine

__all__ = [
    'CommunityContextEngine',
    'StakeholderDigitalTwin',
    'ScenarioDesigner',
    'ScenarioManager',
    'EmergentBehaviourAnalyzer',
    'SocialImpactModel',
    'SocialImpactResult',
    'DecisionComparisonEngine',
    'DecisionIntelligenceEngine',
]
