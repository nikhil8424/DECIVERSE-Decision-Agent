"""Emergent Behaviour Analyzer - Research layer for behaviour analysis.

This layer introduces an explicit analysis layer between raw simulation output
and final reporting. It processes simulation events and derives community-level
behaviour patterns.

Foundation: Analyzes simulation data from DECIVERSE / OASIS.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from datetime import datetime
import re
import statistics

from ..services.simulation_runner import SimulationRunner
from ..utils.logger import get_logger

logger = get_logger('mirofish.research.emergent_behaviour')


@dataclass
class BehaviourMetrics:
    """Calculated emergent behaviour metrics."""
    
    # Consensus metrics
    consensus_score: float = 0.0
    opinion_distribution: Dict[str, float] = field(default_factory=dict)
    
    # Polarization metrics
    polarization_score: float = 0.0
    sentiment_divergence: float = 0.0
    
    # Sentiment metrics
    overall_sentiment: float = 0.0  # -1 to 1
    sentiment_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Conflict metrics
    conflict_intensity: float = 0.0
    disagreement_count: int = 0
    
    # Information diffusion metrics
    information_spread_rate: float = 0.0
    viral_content_count: int = 0
    
    # Agent influence metrics
    influence_distribution: Dict[str, float] = field(default_factory=dict)
    top_influencers: List[Dict[str, Any]] = field(default_factory=list)
    
    # Community/group behaviour
    cluster_cohesion: float = 0.0
    group_alignment: Dict[str, float] = field(default_factory=dict)
    
    # Adoption/support behaviour
    adoption_rate: float = 0.0
    support_ratio: float = 0.0
    
    # Metadata
    calculated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    sample_size: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'consensus_score': self.consensus_score,
            'opinion_distribution': self.opinion_distribution,
            'polarization_score': self.polarization_score,
            'sentiment_divergence': self.sentiment_divergence,
            'overall_sentiment': self.overall_sentiment,
            'sentiment_distribution': self.sentiment_distribution,
            'conflict_intensity': self.conflict_intensity,
            'disagreement_count': self.disagreement_count,
            'information_spread_rate': self.information_spread_rate,
            'viral_content_count': self.viral_content_count,
            'influence_distribution': self.influence_distribution,
            'top_influencers': self.top_influencers,
            'cluster_cohesion': self.cluster_cohesion,
            'group_alignment': self.group_alignment,
            'adoption_rate': self.adoption_rate,
            'support_ratio': self.support_ratio,
            'calculated_at': self.calculated_at,
            'sample_size': self.sample_size,
        }


class EmergentBehaviourAnalyzer:
    """Research-oriented Emergent Behaviour Analyzer.
    
    This component analyzes simulation events to derive community-level
    behaviour patterns and metrics.
    
    The analyzer:
    1. Processes raw simulation actions and events
    2. Calculates consensus and polarization metrics
    3. Analyzes sentiment patterns
    4. Identifies conflict and disagreement
    5. Measures information diffusion
    6. Evaluates agent influence
    7. Assesses community/group behaviour
    8. Tracks adoption and support patterns
    
    Note: Metrics are calculated from available simulation data.
    Some metrics may be estimated or limited by data availability.
    """
    
    def __init__(self):
        logger.info("EmergentBehaviourAnalyzer initialized")
    
    def analyze_simulation(
        self,
        simulation_id: str,
    ) -> BehaviourMetrics:
        """Analyze simulation to extract emergent behaviour metrics.
        
        Args:
            simulation_id: Simulation identifier
            
        Returns:
            BehaviourMetrics with calculated metrics
        """
        logger.info(f"Analyzing emergent behaviour for simulation {simulation_id}")
        
        try:
            # Gather simulation data
            timeline = SimulationRunner.get_timeline(simulation_id)
            agent_stats = SimulationRunner.get_agent_stats(simulation_id)
            actions = SimulationRunner.get_all_actions(simulation_id)
            
            metrics = BehaviourMetrics()
            metrics.sample_size = len(actions)
            
            # Calculate metrics from available data
            self._calculate_sentiment_metrics(metrics, actions)
            self._calculate_conflict_metrics(metrics, actions)
            self._calculate_influence_metrics(metrics, agent_stats)
            self._calculate_consensus_metrics(metrics, actions, agent_stats)
            self._calculate_polarization_metrics(metrics, actions)
            self._calculate_diffusion_metrics(metrics, actions, timeline)
            self._calculate_adoption_metrics(metrics, actions)
            
            logger.info(f"Behaviour analysis complete: {metrics.sample_size} actions analyzed")
            return metrics
            
        except Exception as e:
            logger.error(f"Behaviour analysis failed: {e}")
            # Return empty metrics on failure
            return BehaviourMetrics(sample_size=0)
    
    def _calculate_sentiment_metrics(
        self,
        metrics: BehaviourMetrics,
        actions: List[Any],
    ) -> None:
        """Calculate sentiment-based metrics from actions."""
        sentiment_values = []
        sentiment_counts = Counter()
        
        for action in actions:
            # Extract sentiment from action content if available
            content = action.action_args.get('content', '')
            if content:
                positive_keywords = {'support', 'agree', 'good', 'great', 'excellent', 'positive', 'benefit'}
                negative_keywords = {'oppose', 'disagree', 'bad', 'terrible', 'negative', 'concern', 'problem'}
                
                content_words = set(re.findall(r'\b[a-zA-Z]+\b', content.lower()))
                pos_count = len(positive_keywords.intersection(content_words))
                neg_count = len(negative_keywords.intersection(content_words))
                
                if pos_count > neg_count:
                    sentiment_values.append(0.5 + min(pos_count * 0.1, 0.5))
                    sentiment_counts['positive'] += 1
                elif neg_count > pos_count:
                    sentiment_values.append(-0.5 - min(neg_count * 0.1, 0.5))
                    sentiment_counts['negative'] += 1
                else:
                    sentiment_values.append(0.0)
                    sentiment_counts['neutral'] += 1
        
        if sentiment_values:
            metrics.overall_sentiment = statistics.mean(sentiment_values)
            metrics.sentiment_distribution = dict(sentiment_counts)
    
    def _calculate_conflict_metrics(
        self,
        metrics: BehaviourMetrics,
        actions: List[Any],
    ) -> None:
        """Calculate conflict-related metrics."""
        conflict_actions = ['DISAGREE', 'OPPOSE', 'CRITICIZE', 'ATTACK']
        disagreement_count = sum(
            1 for a in actions
            if a.action_type in conflict_actions or
            any(kw in a.action_args.get('content', '').lower() for kw in ['disagree', 'oppose', 'conflict'])
        )
        
        metrics.disagreement_count = disagreement_count
        if actions:
            metrics.conflict_intensity = disagreement_count / len(actions)
    
    def _calculate_influence_metrics(
        self,
        metrics: BehaviourMetrics,
        agent_stats: List[Dict[str, Any]],
    ) -> None:
        """Calculate agent influence metrics."""
        if not agent_stats:
            return
        
        # Sort by total actions as a proxy for influence
        sorted_agents = sorted(agent_stats, key=lambda x: x.get('total_actions', 0), reverse=True)
        
        total_actions = sum(a.get('total_actions', 0) for a in agent_stats)
        
        # Calculate influence distribution
        for agent in sorted_agents[:10]:  # Top 10 influencers
            agent_name = agent.get('agent_name', 'Unknown')
            actions = agent.get('total_actions', 0)
            influence_score = actions / total_actions if total_actions > 0 else 0
            
            metrics.influence_distribution[agent_name] = influence_score
            metrics.top_influencers.append({
                'agent_name': agent_name,
                'influence_score': influence_score,
                'total_actions': actions,
            })
    
    def _calculate_consensus_metrics(
        self,
        metrics: BehaviourMetrics,
        actions: List[Any],
        agent_stats: List[Dict[str, Any]],
    ) -> None:
        """Calculate consensus metrics."""
        # Consensus is approximated by sentiment agreement
        if metrics.overall_sentiment != 0:
            # High absolute sentiment indicates stronger consensus
            metrics.consensus_score = abs(metrics.overall_sentiment)
        
        # Opinion distribution based on sentiment categories
        if metrics.sentiment_distribution:
            total = sum(metrics.sentiment_distribution.values())
            for category, count in metrics.sentiment_distribution.items():
                metrics.opinion_distribution[category] = count / total if total > 0 else 0
    
    def _calculate_polarization_metrics(
        self,
        metrics: BehaviourMetrics,
        actions: List[Any],
    ) -> None:
        """Calculate polarization metrics."""
        # Polarization is approximated by sentiment divergence
        if metrics.sentiment_distribution:
            pos = metrics.sentiment_distribution.get('positive', 0)
            neg = metrics.sentiment_distribution.get('negative', 0)
            total = pos + neg
            
            if total > 0:
                # Polarization is high when opinions are evenly split (50/50)
                balance = min(pos, neg) / total
                metrics.polarization_score = balance * 2.0  # 0.0 when unanimous, 1.0 when evenly split
                metrics.sentiment_divergence = abs(pos - neg) / total
    
    def _calculate_diffusion_metrics(
        self,
        metrics: BehaviourMetrics,
        actions: List[Any],
        timeline: List[Dict[str, Any]],
    ) -> None:
        """Calculate information diffusion metrics."""
        # Viral content: posts that get many interactions
        action_counts = Counter(a.agent_id for a in actions if a.action_type == 'CREATE_POST')
        
        # Count reposts/shares as diffusion
        share_actions = sum(1 for a in actions if a.action_type in ['REPOST', 'SHARE', 'RETWEET'])
        
        if actions:
            metrics.viral_content_count = len([a for a in actions if a.action_type == 'CREATE_POST'])
            metrics.information_spread_rate = share_actions / len(actions) if len(actions) > 0 else 0
    
    def _calculate_adoption_metrics(
        self,
        metrics: BehaviourMetrics,
        actions: List[Any],
    ) -> None:
        """Calculate adoption and support metrics."""
        support_actions = ['SUPPORT', 'ENDORSE', 'AGREE', 'LIKE', 'LIKE_POST']
        adopt_actions = ['ADOPT', 'IMPLEMENT', 'JOIN', 'SUPPORT', 'ENDORSE', 'AGREE']
        
        support_count = sum(1 for a in actions if a.action_type in support_actions)
        adopt_count = sum(1 for a in actions if a.action_type in adopt_actions)
        
        if actions:
            metrics.support_ratio = support_count / len(actions)
            metrics.adoption_rate = adopt_count / len(actions)
    
    def compare_behaviour(
        self,
        simulation_ids: List[str],
    ) -> Dict[str, Any]:
        """Compare emergent behaviour across multiple simulations.
        
        Args:
            simulation_ids: List of simulation identifiers to compare
            
        Returns:
            Comparison summary
        """
        logger.info(f"Comparing behaviour across {len(simulation_ids)} simulations")
        
        metrics_list = []
        for sim_id in simulation_ids:
            metrics = self.analyze_simulation(sim_id)
            metrics_list.append({
                'simulation_id': sim_id,
                'metrics': metrics.to_dict(),
            })
        
        comparison = {
            'simulation_count': len(simulation_ids),
            'comparisons': metrics_list,
        }
        
        return comparison
