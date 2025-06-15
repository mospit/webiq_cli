import asyncio
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
import json
from enum import Enum

from .real_time_adaptation_system import (
    AdaptationEvent, SystemState, AdaptationSeverity, AdaptationTrigger,
    LiveMonitoringSystem, InterventionType, InterventionPlan
)
from .dynamic_strategy_adjuster import DynamicStrategyAdjuster
from .predictive_intervention_engine import PredictiveInterventionEngine, ExecutionTrajectory

logger = logging.getLogger(__name__)

class ContextType(Enum):
    """Types of execution contexts"""
    WEBSITE_INTERACTION = "website_interaction"
    DATA_EXTRACTION = "data_extraction"
    FORM_SUBMISSION = "form_submission"
    NAVIGATION = "navigation"
    CONTENT_ANALYSIS = "content_analysis"
    AUTOMATION_TASK = "automation_task"
    TESTING = "testing"
    MONITORING = "monitoring"

class SiteCategory(Enum):
    """Categories of websites"""
    E_COMMERCE = "e_commerce"
    SOCIAL_MEDIA = "social_media"
    NEWS_PORTAL = "news_portal"
    CORPORATE = "corporate"
    GOVERNMENT = "government"
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    FINANCIAL = "financial"
    HEALTHCARE = "healthcare"
    TECHNOLOGY = "technology"

class UserPriority(Enum):
    """User priority preferences"""
    SPEED = "speed"
    ACCURACY = "accuracy"
    RELIABILITY = "reliability"
    COST_EFFICIENCY = "cost_efficiency"
    COMPREHENSIVE = "comprehensive"

@dataclass
class ExecutionContext:
    """Represents current execution context"""
    context_id: str
    context_type: ContextType
    site_category: Optional[SiteCategory] = None
    user_priority: Optional[UserPriority] = None
    complexity_level: str = "medium"  # low, medium, high
    time_constraints: Optional[timedelta] = None
    resource_constraints: Dict[str, Any] = field(default_factory=dict)
    quality_requirements: Dict[str, float] = field(default_factory=dict)
    environmental_factors: Dict[str, Any] = field(default_factory=dict)
    historical_performance: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ContextualStrategy:
    """Represents a context-optimized strategy"""
    strategy_id: str
    context_type: ContextType
    site_category: Optional[SiteCategory]
    user_priority: UserPriority
    strategy_parameters: Dict[str, Any]
    expected_performance: Dict[str, float]
    confidence_score: float
    adaptation_rules: List[Dict[str, Any]]
    created_at: datetime = field(default_factory=datetime.now)

class IntelligentContextAdapter:
    """Context-aware strategy optimization and adaptation"""
    
    def __init__(self, monitoring_system: LiveMonitoringSystem,
                 strategy_adjuster: DynamicStrategyAdjuster,
                 predictive_engine: PredictiveInterventionEngine):
        self.monitoring_system = monitoring_system
        self.strategy_adjuster = strategy_adjuster
        self.predictive_engine = predictive_engine
        
        # Context tracking
        self.current_context: Optional[ExecutionContext] = None
        self.context_history: List[ExecutionContext] = []
        
        # Strategy optimization
        self.contextual_strategies: Dict[str, ContextualStrategy] = {}
        self.strategy_performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Context analysis
        self.context_patterns: Dict[str, Dict[str, Any]] = {}
        self.adaptation_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Learning mechanisms
        self.context_learning_data: Dict[str, List[Any]] = defaultdict(list)
        self.strategy_effectiveness_scores: Dict[str, float] = {}
        
        # Initialize context-specific strategies
        self._initialize_contextual_strategies()
        
        # Register with monitoring system
        self.monitoring_system.register_context_callback(self.analyze_context_change)
    
    def _initialize_contextual_strategies(self):
        """Initialize context-specific strategies"""
        
        # E-commerce + Speed priority
        self.contextual_strategies["ecommerce_speed"] = ContextualStrategy(
            strategy_id="ecommerce_speed",
            context_type=ContextType.WEBSITE_INTERACTION,
            site_category=SiteCategory.E_COMMERCE,
            user_priority=UserPriority.SPEED,
            strategy_parameters={
                "preferred_model": "gemini-2.5-flash",
                "timeout_multiplier": 0.8,
                "retry_attempts": 2,
                "parallel_workers": 3,
                "verification_level": "low",
                "cache_aggressive": True
            },
            expected_performance={
                "response_time": 1.5,
                "success_rate": 0.92,
                "accuracy": 0.88
            },
            confidence_score=0.85,
            adaptation_rules=[
                {
                    "condition": "response_time > 2.0",
                    "action": "reduce_verification",
                    "parameters": {"verification_level": "minimal"}
                },
                {
                    "condition": "error_rate > 0.1",
                    "action": "increase_retries",
                    "parameters": {"additional_retries": 1}
                }
            ]
        )
        
        # Financial + Accuracy priority
        self.contextual_strategies["financial_accuracy"] = ContextualStrategy(
            strategy_id="financial_accuracy",
            context_type=ContextType.DATA_EXTRACTION,
            site_category=SiteCategory.FINANCIAL,
            user_priority=UserPriority.ACCURACY,
            strategy_parameters={
                "preferred_model": "gemini-2.5-pro",
                "timeout_multiplier": 2.0,
                "retry_attempts": 5,
                "parallel_workers": 1,
                "verification_level": "high",
                "double_check_critical_data": True
            },
            expected_performance={
                "response_time": 4.0,
                "success_rate": 0.98,
                "accuracy": 0.99
            },
            confidence_score=0.92,
            adaptation_rules=[
                {
                    "condition": "accuracy < 0.95",
                    "action": "increase_verification",
                    "parameters": {"verification_level": "maximum"}
                },
                {
                    "condition": "critical_data_error",
                    "action": "switch_to_manual_verification",
                    "parameters": {"manual_check": True}
                }
            ]
        )
        
        # Government + Reliability priority
        self.contextual_strategies["government_reliability"] = ContextualStrategy(
            strategy_id="government_reliability",
            context_type=ContextType.FORM_SUBMISSION,
            site_category=SiteCategory.GOVERNMENT,
            user_priority=UserPriority.RELIABILITY,
            strategy_parameters={
                "preferred_model": "gemini-2.5-pro",
                "timeout_multiplier": 3.0,
                "retry_attempts": 8,
                "parallel_workers": 1,
                "verification_level": "high",
                "conservative_approach": True,
                "extensive_error_handling": True
            },
            expected_performance={
                "response_time": 5.0,
                "success_rate": 0.99,
                "accuracy": 0.97
            },
            confidence_score=0.88,
            adaptation_rules=[
                {
                    "condition": "any_error",
                    "action": "conservative_fallback",
                    "parameters": {"approach": "step_by_step"}
                }
            ]
        )
        
        logger.info(f"Initialized {len(self.contextual_strategies)} contextual strategies")
    
    async def analyze_context_change(self, context_data: Dict[str, Any]):
        """Analyze context change and adapt strategy accordingly"""
        
        # Extract context information
        new_context = await self._extract_execution_context(context_data)
        
        # Check if context has changed significantly
        context_change = await self._detect_context_change(new_context)
        
        if context_change["significant_change"]:
            logger.info(f"Significant context change detected: {context_change['changes']}")
            
            # Update current context
            self.current_context = new_context
            self.context_history.append(new_context)
            
            # Optimize strategy for new context
            optimization_result = await self.optimize_strategy_for_context(new_context)
            
            # Apply context-aware adaptations
            if optimization_result["adaptations_needed"]:
                await self._apply_contextual_adaptations(optimization_result["adaptations"])
            
            return {
                "context_updated": True,
                "optimization_result": optimization_result
            }
        
        return {"context_updated": False}
    
    async def _extract_execution_context(self, context_data: Dict[str, Any]) -> ExecutionContext:
        """Extract execution context from provided data"""
        
        context_id = context_data.get("context_id", f"ctx_{int(time.time())}")
        
        # Determine context type
        context_type = await self._determine_context_type(context_data)
        
        # Determine site category
        site_category = await self._determine_site_category(context_data)
        
        # Determine user priority
        user_priority = await self._determine_user_priority(context_data)
        
        # Extract complexity level
        complexity_level = context_data.get("complexity_level", "medium")
        
        # Extract constraints
        time_constraints = None
        if "time_limit" in context_data:
            time_constraints = timedelta(seconds=context_data["time_limit"])
        
        resource_constraints = context_data.get("resource_constraints", {})
        quality_requirements = context_data.get("quality_requirements", {})
        environmental_factors = context_data.get("environmental_factors", {})
        
        # Get historical performance for this context
        historical_performance = await self._get_historical_performance(
            context_type, site_category, user_priority
        )
        
        return ExecutionContext(
            context_id=context_id,
            context_type=context_type,
            site_category=site_category,
            user_priority=user_priority,
            complexity_level=complexity_level,
            time_constraints=time_constraints,
            resource_constraints=resource_constraints,
            quality_requirements=quality_requirements,
            environmental_factors=environmental_factors,
            historical_performance=historical_performance
        )
    
    async def _determine_context_type(self, context_data: Dict[str, Any]) -> ContextType:
        """Determine context type from data"""
        
        task_type = context_data.get("task_type", "")
        url = context_data.get("url", "")
        actions = context_data.get("planned_actions", [])
        
        # Analyze task characteristics
        if "form" in task_type.lower() or any("submit" in str(action).lower() for action in actions):
            return ContextType.FORM_SUBMISSION
        elif "extract" in task_type.lower() or "scrape" in task_type.lower():
            return ContextType.DATA_EXTRACTION
        elif "navigate" in task_type.lower() or "browse" in task_type.lower():
            return ContextType.NAVIGATION
        elif "analyze" in task_type.lower() or "content" in task_type.lower():
            return ContextType.CONTENT_ANALYSIS
        elif "test" in task_type.lower():
            return ContextType.TESTING
        elif "monitor" in task_type.lower():
            return ContextType.MONITORING
        elif "automate" in task_type.lower():
            return ContextType.AUTOMATION_TASK
        else:
            return ContextType.WEBSITE_INTERACTION
    
    async def _determine_site_category(self, context_data: Dict[str, Any]) -> Optional[SiteCategory]:
        """Determine site category from data"""
        
        url = context_data.get("url", "").lower()
        site_description = context_data.get("site_description", "").lower()
        
        # URL-based detection
        if any(keyword in url for keyword in ["shop", "store", "buy", "cart", "amazon", "ebay"]):
            return SiteCategory.E_COMMERCE
        elif any(keyword in url for keyword in ["facebook", "twitter", "instagram", "linkedin"]):
            return SiteCategory.SOCIAL_MEDIA
        elif any(keyword in url for keyword in ["news", "cnn", "bbc", "reuters"]):
            return SiteCategory.NEWS_PORTAL
        elif any(keyword in url for keyword in ["gov", "government", "official"]):
            return SiteCategory.GOVERNMENT
        elif any(keyword in url for keyword in ["edu", "university", "school"]):
            return SiteCategory.EDUCATIONAL
        elif any(keyword in url for keyword in ["bank", "finance", "trading", "investment"]):
            return SiteCategory.FINANCIAL
        elif any(keyword in url for keyword in ["health", "medical", "hospital"]):
            return SiteCategory.HEALTHCARE
        elif any(keyword in url for keyword in ["tech", "software", "api", "github"]):
            return SiteCategory.TECHNOLOGY
        
        # Description-based detection
        if any(keyword in site_description for keyword in ["commerce", "shopping", "retail"]):
            return SiteCategory.E_COMMERCE
        elif any(keyword in site_description for keyword in ["social", "community", "network"]):
            return SiteCategory.SOCIAL_MEDIA
        elif any(keyword in site_description for keyword in ["corporate", "business", "company"]):
            return SiteCategory.CORPORATE
        
        return None
    
    async def _determine_user_priority(self, context_data: Dict[str, Any]) -> Optional[UserPriority]:
        """Determine user priority from data"""
        
        explicit_priority = context_data.get("user_priority")
        if explicit_priority:
            try:
                return UserPriority(explicit_priority.lower())
            except ValueError:
                pass
        
        # Infer from context
        time_constraints = context_data.get("time_limit")
        quality_requirements = context_data.get("quality_requirements", {})
        
        if time_constraints and time_constraints < 60:  # Less than 1 minute
            return UserPriority.SPEED
        elif quality_requirements.get("accuracy", 0) > 0.95:
            return UserPriority.ACCURACY
        elif quality_requirements.get("reliability", 0) > 0.95:
            return UserPriority.RELIABILITY
        elif context_data.get("cost_sensitive", False):
            return UserPriority.COST_EFFICIENCY
        else:
            return UserPriority.COMPREHENSIVE
    
    async def _get_historical_performance(self, context_type: ContextType,
                                        site_category: Optional[SiteCategory],
                                        user_priority: Optional[UserPriority]) -> Dict[str, Any]:
        """Get historical performance for similar contexts"""
        
        # This would query actual historical data
        # For now, return mock data
        return {
            "average_response_time": 2.5,
            "average_success_rate": 0.94,
            "common_issues": ["timeout", "element_not_found"],
            "optimal_strategies": ["retry_enhancement", "timeout_adjustment"]
        }
    
    async def _detect_context_change(self, new_context: ExecutionContext) -> Dict[str, Any]:
        """Detect if context has changed significantly"""
        
        if not self.current_context:
            return {
                "significant_change": True,
                "changes": ["initial_context"]
            }
        
        changes = []
        
        # Check for context type change
        if new_context.context_type != self.current_context.context_type:
            changes.append("context_type")
        
        # Check for site category change
        if new_context.site_category != self.current_context.site_category:
            changes.append("site_category")
        
        # Check for user priority change
        if new_context.user_priority != self.current_context.user_priority:
            changes.append("user_priority")
        
        # Check for complexity level change
        if new_context.complexity_level != self.current_context.complexity_level:
            changes.append("complexity_level")
        
        # Check for significant constraint changes
        if new_context.time_constraints != self.current_context.time_constraints:
            changes.append("time_constraints")
        
        significant_change = len(changes) > 0
        
        return {
            "significant_change": significant_change,
            "changes": changes
        }
    
    async def optimize_strategy_for_context(self, context: ExecutionContext) -> Dict[str, Any]:
        """Optimize strategy for given context"""
        
        # Find best matching contextual strategy
        best_strategy = await self._find_best_contextual_strategy(context)
        
        if not best_strategy:
            # Create new strategy for this context
            best_strategy = await self._create_contextual_strategy(context)
        
        # Analyze current strategy vs optimal strategy
        current_strategy = self.strategy_adjuster.get_current_strategy_modifications()
        strategy_gap = await self._analyze_strategy_gap(current_strategy, best_strategy)
        
        # Determine needed adaptations
        adaptations = await self._determine_contextual_adaptations(strategy_gap, context)
        
        return {
            "optimal_strategy": best_strategy,
            "strategy_gap": strategy_gap,
            "adaptations_needed": len(adaptations) > 0,
            "adaptations": adaptations,
            "expected_improvement": await self._estimate_improvement(adaptations, context)
        }
    
    async def _find_best_contextual_strategy(self, context: ExecutionContext) -> Optional[ContextualStrategy]:
        """Find best matching contextual strategy"""
        
        best_match = None
        best_score = 0.0
        
        for strategy_id, strategy in self.contextual_strategies.items():
            score = await self._calculate_context_match_score(context, strategy)
            
            if score > best_score:
                best_score = score
                best_match = strategy
        
        # Only return if match score is above threshold
        if best_score > 0.7:
            return best_match
        
        return None
    
    async def _calculate_context_match_score(self, context: ExecutionContext,
                                           strategy: ContextualStrategy) -> float:
        """Calculate how well a strategy matches the context"""
        score = 0.0
        
        # Context type match (40% weight)
        if context.context_type == strategy.context_type:
            score += 0.4
        
        # Site category match (30% weight)
        if context.site_category == strategy.site_category:
            score += 0.3
        elif context.site_category is None or strategy.site_category is None:
            score += 0.15  # Partial match for unknown category
        
        # User priority match (20% weight)
        if context.user_priority == strategy.user_priority:
            score += 0.2
        elif context.user_priority is None:
            score += 0.1  # Partial match for unknown priority
        
        # Historical performance match (10% weight)
        if strategy.strategy_id in self.strategy_effectiveness_scores:
            effectiveness = self.strategy_effectiveness_scores[strategy.strategy_id]
            score += 0.1 * effectiveness
        else:
            score += 0.05  # Default for unknown effectiveness
        
        return score
    
    async def _create_contextual_strategy(self, context: ExecutionContext) -> ContextualStrategy:
        """Create new contextual strategy for given context"""
        
        strategy_id = f"{context.context_type.value}_{context.site_category.value if context.site_category else 'unknown'}_{context.user_priority.value if context.user_priority else 'default'}"
        
        # Determine strategy parameters based on context
        strategy_parameters = await self._determine_strategy_parameters(context)
        
        # Estimate expected performance
        expected_performance = await self._estimate_strategy_performance(strategy_parameters, context)
        
        # Create adaptation rules
        adaptation_rules = await self._create_adaptation_rules(context)
        
        strategy = ContextualStrategy(
            strategy_id=strategy_id,
            context_type=context.context_type,
            site_category=context.site_category,
            user_priority=context.user_priority or UserPriority.COMPREHENSIVE,
            strategy_parameters=strategy_parameters,
            expected_performance=expected_performance,
            confidence_score=0.6,  # Lower confidence for new strategy
            adaptation_rules=adaptation_rules
        )
        
        # Store new strategy
        self.contextual_strategies[strategy_id] = strategy
        
        logger.info(f"Created new contextual strategy: {strategy_id}")
        
        return strategy
    
    async def _determine_strategy_parameters(self, context: ExecutionContext) -> Dict[str, Any]:
        """Determine strategy parameters based on context"""
        parameters = {
            "preferred_model": "gemini-2.5-flash",
            "timeout_multiplier": 1.0,
            "retry_attempts": 3,
            "parallel_workers": 2,
            "verification_level": "medium"
        }
        
        # Adjust based on user priority
        if context.user_priority == UserPriority.SPEED:
            parameters.update({
                "preferred_model": "gemini-2.5-flash",
                "timeout_multiplier": 0.8,
                "retry_attempts": 2,
                "parallel_workers": 3,
                "verification_level": "low"
            })
        elif context.user_priority == UserPriority.ACCURACY:
            parameters.update({
                "preferred_model": "gemini-2.5-pro",
                "timeout_multiplier": 1.5,
                "retry_attempts": 5,
                "parallel_workers": 1,
                "verification_level": "high"
            })
        elif context.user_priority == UserPriority.RELIABILITY:
            parameters.update({
                "preferred_model": "gemini-2.5-pro",
                "timeout_multiplier": 2.0,
                "retry_attempts": 8,
                "parallel_workers": 1,
                "verification_level": "high",
                "conservative_approach": True
            })
        
        # Adjust based on site category
        if context.site_category == SiteCategory.FINANCIAL:
            parameters.update({
                "verification_level": "high",
                "double_check_critical_data": True,
                "retry_attempts": max(parameters["retry_attempts"], 5)
            })
        elif context.site_category == SiteCategory.GOVERNMENT:
            parameters.update({
                "conservative_approach": True,
                "extensive_error_handling": True,
                "timeout_multiplier": max(parameters["timeout_multiplier"], 2.0)
            })
        
        # Adjust based on complexity
        if context.complexity_level == "high":
            parameters.update({
                "timeout_multiplier": parameters["timeout_multiplier"] * 1.5,
                "retry_attempts": parameters["retry_attempts"] + 2,
                "verification_level": "high"
            })
        elif context.complexity_level == "low":
            parameters.update({
                "timeout_multiplier": parameters["timeout_multiplier"] * 0.8,
                "parallel_workers": min(parameters["parallel_workers"] + 1, 5)
            })
        
        # Adjust based on time constraints
        if context.time_constraints and context.time_constraints < timedelta(minutes=2):
            parameters.update({
                "preferred_model": "gemini-2.5-flash",
                "timeout_multiplier": 0.7,
                "verification_level": "low",
                "parallel_workers": min(parameters["parallel_workers"] + 2, 5)
            })
        
        return parameters
    
    async def _estimate_strategy_performance(self, parameters: Dict[str, Any],
                                           context: ExecutionContext) -> Dict[str, float]:
        """Estimate expected performance for strategy parameters"""
        
        # Base performance estimates
        base_response_time = 2.5
        base_success_rate = 0.92
        base_accuracy = 0.90
        
        # Adjust based on parameters
        if parameters.get("preferred_model") == "gemini-2.5-flash":
            base_response_time *= 0.7
            base_accuracy *= 0.95
        elif parameters.get("preferred_model") == "gemini-2.5-pro":
            base_response_time *= 1.3
            base_accuracy *= 1.05
        
        timeout_multiplier = parameters.get("timeout_multiplier", 1.0)
        base_response_time *= timeout_multiplier
        
        retry_attempts = parameters.get("retry_attempts", 3)
        base_success_rate = min(0.99, base_success_rate + (retry_attempts - 3) * 0.02)
        
        verification_level = parameters.get("verification_level", "medium")
        if verification_level == "high":
            base_accuracy *= 1.05
            base_response_time *= 1.2
        elif verification_level == "low":
            base_accuracy *= 0.95
            base_response_time *= 0.9
        
        return {
            "response_time": max(0.5, base_response_time),
            "success_rate": min(0.99, max(0.5, base_success_rate)),
            "accuracy": min(0.99, max(0.5, base_accuracy))
        }
    
    async def _create_adaptation_rules(self, context: ExecutionContext) -> List[Dict[str, Any]]:
        """Create adaptation rules for context"""
        rules = []
        
        # Universal rules
        rules.append({
            "condition": "error_rate > 0.15",
            "action": "increase_retries",
            "parameters": {"additional_retries": 2}
        })
        
        rules.append({
            "condition": "response_time > 5.0",
            "action": "optimize_performance",
            "parameters": {"switch_to_faster_model": True}
        })
        
        # Context-specific rules
        if context.user_priority == UserPriority.SPEED:
            rules.append({
                "condition": "response_time > 3.0",
                "action": "reduce_verification",
                "parameters": {"verification_level": "minimal"}
            })
        
        if context.site_category == SiteCategory.FINANCIAL:
            rules.append({
                "condition": "accuracy < 0.95",
                "action": "increase_verification",
                "parameters": {"verification_level": "maximum"}
            })
        
        return rules
    
    async def _analyze_strategy_gap(self, current_strategy: Dict[str, Any],
                                  optimal_strategy: ContextualStrategy) -> Dict[str, Any]:
        """Analyze gap between current and optimal strategy"""
        
        gaps = {}
        optimal_params = optimal_strategy.strategy_parameters
        
        for param, optimal_value in optimal_params.items():
            current_value = current_strategy.get(param)
            
            if current_value != optimal_value:
                gaps[param] = {
                    "current": current_value,
                    "optimal": optimal_value,
                    "gap_type": "value_mismatch"
                }
        
        # Check for missing parameters
        for param in optimal_params:
            if param not in current_strategy:
                gaps[param] = {
                    "current": None,
                    "optimal": optimal_params[param],
                    "gap_type": "missing_parameter"
                }
        
        return gaps
    
    async def _determine_contextual_adaptations(self, strategy_gap: Dict[str, Any],
                                              context: ExecutionContext) -> List[Dict[str, Any]]:
        """Determine adaptations needed to close strategy gap"""
        
        adaptations = []
        
        for param, gap_info in strategy_gap.items():
            adaptation = await self._create_adaptation_for_gap(param, gap_info, context)
            if adaptation:
                adaptations.append(adaptation)
        
        return adaptations
    
    async def _create_adaptation_for_gap(self, param: str, gap_info: Dict[str, Any],
                                       context: ExecutionContext) -> Optional[Dict[str, Any]]:
        """Create adaptation to address specific parameter gap"""
        
        optimal_value = gap_info["optimal"]
        current_value = gap_info["current"]
        
        if param == "preferred_model":
            return {
                "type": "model_optimization",
                "action": "switch_model",
                "parameters": {"target_model": optimal_value},
                "priority": "high" if context.user_priority in [UserPriority.ACCURACY, UserPriority.SPEED] else "medium"
            }
        
        elif param == "timeout_multiplier":
            if optimal_value > (current_value or 1.0):
                return {
                    "type": "timeout_adjustment",
                    "action": "increase_timeouts",
                    "parameters": {"multiplier": optimal_value},
                    "priority": "medium"
                }
            else:
                return {
                    "type": "timeout_adjustment",
                    "action": "decrease_timeouts",
                    "parameters": {"multiplier": optimal_value},
                    "priority": "low"
                }
        
        elif param == "retry_attempts":
            if optimal_value > (current_value or 3):
                return {
                    "type": "retry_enhancement",
                    "action": "increase_retry_attempts",
                    "parameters": {"target_attempts": optimal_value},
                    "priority": "medium"
                }
            else:
                return {
                    "type": "retry_enhancement",
                    "action": "decrease_retry_attempts",
                    "parameters": {"target_attempts": optimal_value},
                    "priority": "low"
                }
        
        elif param == "parallel_workers":
            return {
                "type": "parallel_execution_adjustment",
                "action": "adjust_workers",
                "parameters": {"target_workers": optimal_value},
                "priority": "low"
            }
        
        elif param == "verification_level":
            return {
                "type": "verification_adjustment",
                "action": "set_verification_level",
                "parameters": {"level": optimal_value},
                "priority": "medium"
            }
        
        return None
    
    async def _estimate_improvement(self, adaptations: List[Dict[str, Any]],
                                  context: ExecutionContext) -> Dict[str, float]:
        """Estimate improvement from applying adaptations"""
        
        improvement = {
            "response_time_improvement": 0.0,
            "success_rate_improvement": 0.0,
            "accuracy_improvement": 0.0,
            "overall_improvement": 0.0
        }
        
        for adaptation in adaptations:
            adaptation_type = adaptation["type"]
            
            if adaptation_type == "model_optimization":
                target_model = adaptation["parameters"].get("target_model")
                if target_model == "gemini-2.5-flash":
                    improvement["response_time_improvement"] += 0.15
                elif target_model == "gemini-2.5-pro":
                    improvement["accuracy_improvement"] += 0.05
            
            elif adaptation_type == "timeout_adjustment":
                improvement["success_rate_improvement"] += 0.02
            
            elif adaptation_type == "retry_enhancement":
                improvement["success_rate_improvement"] += 0.03
        
        # Calculate overall improvement
        improvement["overall_improvement"] = np.mean([
            improvement["response_time_improvement"],
            improvement["success_rate_improvement"],
            improvement["accuracy_improvement"]
        ])
        
        return improvement
    
    async def _apply_contextual_adaptations(self, adaptations: List[Dict[str, Any]]):
        """Apply contextual adaptations through strategy adjuster"""
        
        for adaptation in adaptations:
            # Create adaptation event
            adaptation_event = AdaptationEvent(
                event_id=f"context_adapt_{int(time.time())}",
                trigger=AdaptationTrigger.CONTEXT_CHANGE,
                severity=AdaptationSeverity.MEDIUM,
                confidence_score=0.8,
                suggested_adaptations=[adaptation],
                context={"source": "intelligent_context_adapter"}
            )
            
            # Apply through strategy adjuster
            result = await self.strategy_adjuster.apply_adaptation(adaptation_event)
            logger.info(f"Applied contextual adaptation: {adaptation['type']} - {result['status']}")
    
    async def learn_from_execution_outcome(self, context: ExecutionContext,
                                         strategy_used: Dict[str, Any],
                                         outcome: Dict[str, Any]):
        """Learn from execution outcomes to improve future context adaptation"""
        
        # Record strategy performance
        strategy_id = self._get_strategy_id_for_context(context)
        
        performance_record = {
            "timestamp": datetime.now(),
            "context": context,
            "strategy": strategy_used,
            "outcome": outcome,
            "effectiveness_score": await self._calculate_effectiveness_score(outcome)
        }
        
        self.strategy_performance_history[strategy_id].append(performance_record)
        
        # Update strategy effectiveness scores
        await self._update_strategy_effectiveness_scores(strategy_id, performance_record)
        
        # Update context patterns
        await self._update_context_patterns(context, outcome)
        
        # Adapt strategies based on learning
        await self._adapt_strategies_from_learning()
        
        logger.info(f"Learned from execution outcome for context: {context.context_id}")
    
    def _get_strategy_id_for_context(self, context: ExecutionContext) -> str:
        """Get strategy ID for given context"""
        return f"{context.context_type.value}_{context.site_category.value if context.site_category else 'unknown'}_{context.user_priority.value if context.user_priority else 'default'}"
    
    async def _calculate_effectiveness_score(self, outcome: Dict[str, Any]) -> float:
        """Calculate effectiveness score from outcome"""
        success_rate = outcome.get("success_rate", 0.0)
        response_time = outcome.get("response_time", 10.0)
        accuracy = outcome.get("accuracy", 0.0)
        
        # Normalize response time (lower is better)
        normalized_response_time = max(0, 1 - (response_time - 1.0) / 10.0)
        
        # Calculate weighted effectiveness score
        effectiveness = (
            success_rate * 0.4 +
            normalized_response_time * 0.3 +
            accuracy * 0.3
        )
        
        return min(1.0, max(0.0, effectiveness))
    
    async def _update_strategy_effectiveness_scores(self, strategy_id: str,
                                                  performance_record: Dict[str, Any]):
        """Update strategy effectiveness scores"""
        effectiveness = performance_record["effectiveness_score"]
        
        if strategy_id in self.strategy_effectiveness_scores:
            # Exponential moving average
            current_score = self.strategy_effectiveness_scores[strategy_id]
            self.strategy_effectiveness_scores[strategy_id] = 0.8 * current_score + 0.2 * effectiveness
        else:
            self.strategy_effectiveness_scores[strategy_id] = effectiveness
    
    async def _update_context_patterns(self, context: ExecutionContext, outcome: Dict[str, Any]):
        """Update context patterns based on outcomes"""
        pattern_key = f"{context.context_type.value}_{context.site_category.value if context.site_category else 'unknown'}"
        
        if pattern_key not in self.context_patterns:
            self.context_patterns[pattern_key] = {
                "success_factors": [],
                "failure_factors": [],
                "optimal_parameters": {}
            }
        
        # Update patterns based on outcome
        if outcome.get("success_rate", 0) > 0.9:
            # Record success factors
            self.context_patterns[pattern_key]["success_factors"].append({
                "complexity": context.complexity_level,
                "user_priority": context.user_priority.value if context.user_priority else None,
                "outcome": outcome
            })
        elif outcome.get("success_rate", 0) < 0.7:
            # Record failure factors
            self.context_patterns[pattern_key]["failure_factors"].append({
                "complexity": context.complexity_level,
                "user_priority": context.user_priority.value if context.user_priority else None,
                "outcome": outcome
            })
    
    async def _adapt_strategies_from_learning(self):
        """Adapt strategies based on accumulated learning"""
        
        for strategy_id, strategy in self.contextual_strategies.items():
            if strategy_id in self.strategy_performance_history:
                performance_history = self.strategy_performance_history[strategy_id]
                
                if len(performance_history) >= 5:  # Enough data for adaptation
                    # Analyze performance trends
                    recent_performance = performance_history[-5:]
                    avg_effectiveness = np.mean([p["effectiveness_score"] for p in recent_performance])
                    
                    # Adapt strategy if performance is below threshold
                    if avg_effectiveness < 0.7:
                        await self._adapt_underperforming_strategy(strategy_id, strategy, recent_performance)
    
    async def _adapt_underperforming_strategy(self, strategy_id: str,
                                            strategy: ContextualStrategy,
                                            performance_history: List[Dict[str, Any]]):
        """Adapt underperforming strategy"""
        
        # Analyze common issues
        common_issues = []
        for record in performance_history:
            outcome = record["outcome"]
            if outcome.get("response_time", 0) > 5.0:
                common_issues.append("slow_response")
            if outcome.get("success_rate", 0) < 0.8:
                common_issues.append("low_success_rate")
            if outcome.get("accuracy", 0) < 0.85:
                common_issues.append("low_accuracy")
        
        # Adapt strategy parameters
        if "slow_response" in common_issues:
            strategy.strategy_parameters["preferred_model"] = "gemini-2.5-flash"
            strategy.strategy_parameters["timeout_multiplier"] = 0.8
        
        if "low_success_rate" in common_issues:
            strategy.strategy_parameters["retry_attempts"] = min(
                strategy.strategy_parameters.get("retry_attempts", 3) + 2, 8
            )
        
        if "low_accuracy" in common_issues:
            strategy.strategy_parameters["verification_level"] = "high"
            strategy.strategy_parameters["preferred_model"] = "gemini-2.5-pro"
        
        logger.info(f"Adapted underperforming strategy: {strategy_id}")
    
    def get_current_context(self) -> Optional[ExecutionContext]:
        """Get current execution context"""
        return self.current_context
    
    def get_contextual_strategies(self) -> Dict[str, ContextualStrategy]:
        """Get all contextual strategies"""
        return self.contextual_strategies.copy()
    
    def get_strategy_effectiveness_scores(self) -> Dict[str, float]:
        """Get strategy effectiveness scores"""
        return self.strategy_effectiveness_scores.copy()