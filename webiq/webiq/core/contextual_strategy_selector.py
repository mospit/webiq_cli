import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

class StrategyType(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    HYBRID = "hybrid"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"

class ExecutionMode(Enum):
    HEADLESS = "headless"
    VISIBLE = "visible"
    HYBRID = "hybrid"

@dataclass
class StrategyTemplate:
    """Template for execution strategies"""
    name: str
    strategy_type: StrategyType
    execution_mode: ExecutionMode
    priority_factors: List[str]
    optimization_targets: List[str]
    risk_tolerance: float  # 0.0 to 1.0
    performance_weight: float  # 0.0 to 1.0
    reliability_weight: float  # 0.0 to 1.0
    speed_weight: float  # 0.0 to 1.0
    resource_requirements: Dict[str, Any]
    constraints: List[str]
    success_criteria: List[str]
    fallback_strategies: List[str]
    estimated_duration: Optional[int] = None  # seconds
    confidence_threshold: float = 0.8
    retry_policy: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.retry_policy is None:
            self.retry_policy = {
                "max_retries": 3,
                "backoff_factor": 2.0,
                "retry_on_errors": ["timeout", "network_error", "element_not_found"]
            }

@dataclass
class ContextFactors:
    """Factors that influence strategy selection"""
    site_complexity: float
    goal_complexity: float
    time_constraints: Optional[int]  # seconds
    resource_availability: Dict[str, float]
    user_preferences: Dict[str, Any]
    historical_performance: Dict[str, float]
    current_load: float
    network_conditions: Dict[str, Any]
    device_capabilities: Dict[str, Any]
    security_requirements: List[str]
    compliance_needs: List[str]
    error_tolerance: float
    success_rate_requirement: float

@dataclass
class StrategyEvaluation:
    """Evaluation results for a strategy"""
    strategy_name: str
    suitability_score: float
    confidence_level: float
    estimated_success_rate: float
    estimated_duration: int
    resource_cost: float
    risk_assessment: Dict[str, float]
    pros: List[str]
    cons: List[str]
    recommendations: List[str]

class ContextualStrategySelector:
    """Select optimal execution strategies based on context"""
    
    def __init__(self):
        self.strategy_templates = self._initialize_strategy_templates()
        self.context_weights = self._initialize_context_weights()
        self.performance_history = {}
        self.strategy_cache = {}
        
    def select_strategy(self, goal_decomposition: Dict[str, Any], 
                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Select the optimal strategy for goal execution"""
        try:
            # Extract context factors
            context_factors = self._extract_context_factors(goal_decomposition, context)
            
            # Evaluate all available strategies
            strategy_evaluations = []
            for template in self.strategy_templates:
                evaluation = self._evaluate_strategy(template, context_factors, goal_decomposition)
                strategy_evaluations.append(evaluation)
            
            # Sort by suitability score
            strategy_evaluations.sort(key=lambda x: x.suitability_score, reverse=True)
            
            # Select the best strategy
            best_strategy = strategy_evaluations[0]
            
            # Customize the selected strategy
            customized_strategy = self._customize_strategy(
                self.strategy_templates[0],  # Get template by name
                context_factors,
                goal_decomposition
            )
            
            return {
                "selected_strategy": customized_strategy,
                "evaluation": asdict(best_strategy),
                "alternatives": [asdict(eval) for eval in strategy_evaluations[1:3]],  # Top 3 alternatives
                "context_analysis": asdict(context_factors),
                "selection_reasoning": self._generate_selection_reasoning(best_strategy, context_factors)
            }
            
        except Exception as e:
            logger.error(f"Strategy selection failed: {e}")
            return self._get_fallback_strategy(goal_decomposition, context)
    
    def _extract_context_factors(self, goal_decomposition: Dict[str, Any], 
                                context: Dict[str, Any]) -> ContextFactors:
        """Extract and analyze context factors"""
        return ContextFactors(
            site_complexity=context.get("site_complexity", 0.5),
            goal_complexity=goal_decomposition.get("complexity_score", 0.5),
            time_constraints=context.get("time_limit_seconds"),
            resource_availability={
                "cpu": context.get("cpu_availability", 0.8),
                "memory": context.get("memory_availability", 0.8),
                "network": context.get("network_quality", 0.8),
                "browser_instances": context.get("max_browser_instances", 1)
            },
            user_preferences=context.get("user_preferences", {}),
            historical_performance=self._get_historical_performance(context.get("site_url", "")),
            current_load=context.get("system_load", 0.3),
            network_conditions={
                "latency": context.get("network_latency", 50),
                "bandwidth": context.get("network_bandwidth", 100),
                "stability": context.get("network_stability", 0.9)
            },
            device_capabilities={
                "is_mobile": context.get("is_mobile", False),
                "screen_size": context.get("screen_size", "desktop"),
                "touch_enabled": context.get("touch_enabled", False)
            },
            security_requirements=context.get("security_requirements", []),
            compliance_needs=context.get("compliance_needs", []),
            error_tolerance=context.get("error_tolerance", 0.1),
            success_rate_requirement=context.get("required_success_rate", 0.95)
        )
    
    def _evaluate_strategy(self, template: StrategyTemplate, 
                          context_factors: ContextFactors,
                          goal_decomposition: Dict[str, Any]) -> StrategyEvaluation:
        """Evaluate a strategy template against context factors"""
        
        # Calculate base suitability score
        suitability_score = 0.0
        
        # Factor 1: Complexity alignment
        complexity_alignment = 1.0 - abs(context_factors.goal_complexity - 
                                        self._get_strategy_complexity_preference(template))
        suitability_score += complexity_alignment * 0.25
        
        # Factor 2: Resource requirements vs availability
        resource_score = self._calculate_resource_compatibility(template, context_factors)
        suitability_score += resource_score * 0.20
        
        # Factor 3: Performance alignment
        performance_score = self._calculate_performance_alignment(template, context_factors)
        suitability_score += performance_score * 0.20
        
        # Factor 4: Risk tolerance alignment
        risk_score = self._calculate_risk_alignment(template, context_factors)
        suitability_score += risk_score * 0.15
        
        # Factor 5: Historical performance
        historical_score = self._calculate_historical_performance_score(template, context_factors)
        suitability_score += historical_score * 0.20
        
        # Calculate confidence level
        confidence_level = min(suitability_score * 1.2, 1.0)
        
        # Estimate success rate
        estimated_success_rate = self._estimate_success_rate(template, context_factors, goal_decomposition)
        
        # Estimate duration
        estimated_duration = self._estimate_execution_duration(template, context_factors, goal_decomposition)
        
        # Calculate resource cost
        resource_cost = self._calculate_resource_cost(template, context_factors)
        
        # Risk assessment
        risk_assessment = self._assess_strategy_risks(template, context_factors)
        
        # Generate pros and cons
        pros, cons = self._generate_pros_cons(template, context_factors)
        
        # Generate recommendations
        recommendations = self._generate_strategy_recommendations(template, context_factors)
        
        return StrategyEvaluation(
            strategy_name=template.name,
            suitability_score=suitability_score,
            confidence_level=confidence_level,
            estimated_success_rate=estimated_success_rate,
            estimated_duration=estimated_duration,
            resource_cost=resource_cost,
            risk_assessment=risk_assessment,
            pros=pros,
            cons=cons,
            recommendations=recommendations
        )
    
    def _customize_strategy(self, template: StrategyTemplate, 
                           context_factors: ContextFactors,
                           goal_decomposition: Dict[str, Any]) -> Dict[str, Any]:
        """Customize strategy template based on specific context"""
        customized = asdict(template)
        
        # Adjust retry policy based on error tolerance
        if context_factors.error_tolerance < 0.05:
            customized["retry_policy"]["max_retries"] = 5
            customized["retry_policy"]["backoff_factor"] = 1.5
        elif context_factors.error_tolerance > 0.2:
            customized["retry_policy"]["max_retries"] = 2
            customized["retry_policy"]["backoff_factor"] = 3.0
        
        # Adjust execution mode based on device capabilities
        if context_factors.device_capabilities["is_mobile"]:
            if customized["execution_mode"] == ExecutionMode.VISIBLE.value:
                customized["execution_mode"] = ExecutionMode.HYBRID.value
        
        # Adjust timeouts based on network conditions
        network_quality = context_factors.network_conditions["stability"]
        if network_quality < 0.7:
            customized["resource_requirements"]["timeout_multiplier"] = 2.0
        elif network_quality > 0.95:
            customized["resource_requirements"]["timeout_multiplier"] = 0.8
        
        # Adjust parallelization based on resource availability
        cpu_availability = context_factors.resource_availability["cpu"]
        if cpu_availability > 0.8 and template.strategy_type == StrategyType.PARALLEL:
            customized["resource_requirements"]["max_parallel_tasks"] = min(
                customized["resource_requirements"].get("max_parallel_tasks", 3) + 1, 5
            )
        elif cpu_availability < 0.5:
            customized["resource_requirements"]["max_parallel_tasks"] = 1
        
        # Add context-specific constraints
        if context_factors.security_requirements:
            customized["constraints"].extend([
                f"security_requirement: {req}" for req in context_factors.security_requirements
            ])
        
        if context_factors.compliance_needs:
            customized["constraints"].extend([
                f"compliance_requirement: {need}" for need in context_factors.compliance_needs
            ])
        
        # Adjust success criteria based on requirements
        if context_factors.success_rate_requirement > 0.98:
            customized["success_criteria"].append("enhanced_verification_required")
        
        # Add time-specific optimizations
        if context_factors.time_constraints and context_factors.time_constraints < 300:  # 5 minutes
            customized["optimization_targets"].append("speed_optimization")
            customized["speed_weight"] = min(customized["speed_weight"] + 0.2, 1.0)
        
        # Add goal-specific customizations
        sub_goals = goal_decomposition.get("sub_goals", [])
        if len(sub_goals) > 10:
            customized["optimization_targets"].append("batch_processing")
        
        # Add execution metadata
        customized["execution_metadata"] = {
            "customization_timestamp": datetime.now().isoformat(),
            "context_hash": hash(str(context_factors)),
            "goal_hash": hash(str(goal_decomposition)),
            "customization_version": "1.0"
        }
        
        return customized
    
    def _initialize_strategy_templates(self) -> List[StrategyTemplate]:
        """Initialize predefined strategy templates"""
        return [
            StrategyTemplate(
                name="conservative_sequential",
                strategy_type=StrategyType.SEQUENTIAL,
                execution_mode=ExecutionMode.VISIBLE,
                priority_factors=["reliability", "accuracy", "error_handling"],
                optimization_targets=["success_rate", "error_recovery"],
                risk_tolerance=0.2,
                performance_weight=0.3,
                reliability_weight=0.8,
                speed_weight=0.4,
                resource_requirements={
                    "cpu_usage": 0.4,
                    "memory_usage": 0.3,
                    "network_usage": 0.3,
                    "timeout_multiplier": 1.5,
                    "max_parallel_tasks": 1
                },
                constraints=["single_threaded", "full_verification"],
                success_criteria=["zero_errors", "complete_verification"],
                fallback_strategies=["manual_intervention", "step_by_step_debug"]
            ),
            StrategyTemplate(
                name="aggressive_parallel",
                strategy_type=StrategyType.PARALLEL,
                execution_mode=ExecutionMode.HEADLESS,
                priority_factors=["speed", "efficiency", "throughput"],
                optimization_targets=["execution_time", "resource_utilization"],
                risk_tolerance=0.7,
                performance_weight=0.8,
                reliability_weight=0.5,
                speed_weight=0.9,
                resource_requirements={
                    "cpu_usage": 0.8,
                    "memory_usage": 0.7,
                    "network_usage": 0.6,
                    "timeout_multiplier": 0.8,
                    "max_parallel_tasks": 5
                },
                constraints=["multi_threaded", "minimal_verification"],
                success_criteria=["fast_completion", "high_throughput"],
                fallback_strategies=["reduce_parallelism", "sequential_fallback"]
            ),
            StrategyTemplate(
                name="adaptive_hybrid",
                strategy_type=StrategyType.ADAPTIVE,
                execution_mode=ExecutionMode.HYBRID,
                priority_factors=["adaptability", "balance", "optimization"],
                optimization_targets=["dynamic_adjustment", "context_awareness"],
                risk_tolerance=0.5,
                performance_weight=0.6,
                reliability_weight=0.7,
                speed_weight=0.6,
                resource_requirements={
                    "cpu_usage": 0.6,
                    "memory_usage": 0.5,
                    "network_usage": 0.5,
                    "timeout_multiplier": 1.0,
                    "max_parallel_tasks": 3
                },
                constraints=["dynamic_adjustment", "context_monitoring"],
                success_criteria=["optimal_balance", "adaptive_performance"],
                fallback_strategies=["conservative_fallback", "aggressive_boost"]
            ),
            StrategyTemplate(
                name="mobile_optimized",
                strategy_type=StrategyType.CONSERVATIVE,
                execution_mode=ExecutionMode.VISIBLE,
                priority_factors=["mobile_compatibility", "touch_optimization", "bandwidth_efficiency"],
                optimization_targets=["mobile_performance", "data_usage"],
                risk_tolerance=0.3,
                performance_weight=0.4,
                reliability_weight=0.8,
                speed_weight=0.5,
                resource_requirements={
                    "cpu_usage": 0.3,
                    "memory_usage": 0.4,
                    "network_usage": 0.2,
                    "timeout_multiplier": 2.0,
                    "max_parallel_tasks": 1
                },
                constraints=["mobile_optimized", "touch_friendly", "low_bandwidth"],
                success_criteria=["mobile_compatibility", "efficient_data_usage"],
                fallback_strategies=["desktop_mode", "simplified_interaction"]
            ),
            StrategyTemplate(
                name="high_security",
                strategy_type=StrategyType.CONSERVATIVE,
                execution_mode=ExecutionMode.VISIBLE,
                priority_factors=["security", "compliance", "audit_trail"],
                optimization_targets=["security_compliance", "audit_logging"],
                risk_tolerance=0.1,
                performance_weight=0.3,
                reliability_weight=0.9,
                speed_weight=0.3,
                resource_requirements={
                    "cpu_usage": 0.5,
                    "memory_usage": 0.6,
                    "network_usage": 0.4,
                    "timeout_multiplier": 2.0,
                    "max_parallel_tasks": 1
                },
                constraints=["security_enhanced", "full_logging", "compliance_verified"],
                success_criteria=["security_compliance", "complete_audit_trail"],
                fallback_strategies=["manual_security_review", "enhanced_verification"]
            )
        ]
    
    def _initialize_context_weights(self) -> Dict[str, float]:
        """Initialize weights for different context factors"""
        return {
            "site_complexity": 0.20,
            "goal_complexity": 0.25,
            "time_constraints": 0.15,
            "resource_availability": 0.15,
            "historical_performance": 0.10,
            "network_conditions": 0.10,
            "security_requirements": 0.05
        }
    
    def _get_historical_performance(self, site_url: str) -> Dict[str, float]:
        """Get historical performance data for a site"""
        return self.performance_history.get(site_url, {
            "success_rate": 0.85,
            "average_duration": 120,
            "error_rate": 0.15,
            "retry_rate": 0.10
        })
    
    def _get_strategy_complexity_preference(self, template: StrategyTemplate) -> float:
        """Get the complexity preference for a strategy template"""
        complexity_map = {
            StrategyType.CONSERVATIVE: 0.3,
            StrategyType.SEQUENTIAL: 0.4,
            StrategyType.ADAPTIVE: 0.6,
            StrategyType.PARALLEL: 0.7,
            StrategyType.AGGRESSIVE: 0.8,
            StrategyType.HYBRID: 0.5
        }
        return complexity_map.get(template.strategy_type, 0.5)
    
    def _calculate_resource_compatibility(self, template: StrategyTemplate, 
                                        context_factors: ContextFactors) -> float:
        """Calculate how well strategy resource requirements match availability"""
        compatibility_score = 0.0
        
        # CPU compatibility
        cpu_required = template.resource_requirements.get("cpu_usage", 0.5)
        cpu_available = context_factors.resource_availability["cpu"]
        cpu_score = 1.0 if cpu_available >= cpu_required else cpu_available / cpu_required
        compatibility_score += cpu_score * 0.4
        
        # Memory compatibility
        memory_required = template.resource_requirements.get("memory_usage", 0.5)
        memory_available = context_factors.resource_availability["memory"]
        memory_score = 1.0 if memory_available >= memory_required else memory_available / memory_required
        compatibility_score += memory_score * 0.3
        
        # Network compatibility
        network_required = template.resource_requirements.get("network_usage", 0.5)
        network_available = context_factors.resource_availability["network"]
        network_score = 1.0 if network_available >= network_required else network_available / network_required
        compatibility_score += network_score * 0.3
        
        return min(compatibility_score, 1.0)
    
    def _calculate_performance_alignment(self, template: StrategyTemplate, 
                                       context_factors: ContextFactors) -> float:
        """Calculate performance alignment score"""
        alignment_score = 0.0
        
        # Time constraint alignment
        if context_factors.time_constraints:
            if context_factors.time_constraints < 300:  # 5 minutes - need speed
                alignment_score += template.speed_weight * 0.5
            else:  # More time available - can prioritize reliability
                alignment_score += template.reliability_weight * 0.5
        else:
            alignment_score += 0.3  # Neutral score when no time constraints
        
        # Success rate requirement alignment
        if context_factors.success_rate_requirement > 0.95:
            alignment_score += template.reliability_weight * 0.3
        else:
            alignment_score += template.performance_weight * 0.3
        
        # Error tolerance alignment
        if context_factors.error_tolerance < 0.05:
            alignment_score += template.reliability_weight * 0.2
        else:
            alignment_score += template.speed_weight * 0.2
        
        return min(alignment_score, 1.0)
    
    def _calculate_risk_alignment(self, template: StrategyTemplate, 
                                context_factors: ContextFactors) -> float:
        """Calculate risk tolerance alignment"""
        # Calculate context risk tolerance based on requirements
        context_risk_tolerance = 0.5  # Default
        
        if context_factors.security_requirements:
            context_risk_tolerance -= 0.2
        
        if context_factors.compliance_needs:
            context_risk_tolerance -= 0.1
        
        if context_factors.error_tolerance < 0.05:
            context_risk_tolerance -= 0.2
        
        if context_factors.success_rate_requirement > 0.98:
            context_risk_tolerance -= 0.1
        
        context_risk_tolerance = max(context_risk_tolerance, 0.0)
        
        # Calculate alignment
        risk_diff = abs(template.risk_tolerance - context_risk_tolerance)
        return 1.0 - risk_diff
    
    def _calculate_historical_performance_score(self, template: StrategyTemplate, 
                                              context_factors: ContextFactors) -> float:
        """Calculate score based on historical performance"""
        historical = context_factors.historical_performance
        
        # Base score from historical success rate
        base_score = historical["success_rate"]
        
        # Adjust based on strategy characteristics
        if template.strategy_type == StrategyType.CONSERVATIVE and historical["error_rate"] > 0.2:
            base_score += 0.1  # Conservative strategies better for error-prone sites
        
        if template.strategy_type == StrategyType.AGGRESSIVE and historical["success_rate"] > 0.9:
            base_score += 0.1  # Aggressive strategies better for reliable sites
        
        return min(base_score, 1.0)
    
    def _estimate_success_rate(self, template: StrategyTemplate, 
                             context_factors: ContextFactors,
                             goal_decomposition: Dict[str, Any]) -> float:
        """Estimate success rate for strategy"""
        base_rate = context_factors.historical_performance["success_rate"]
        
        # Adjust based on strategy reliability weight
        reliability_adjustment = (template.reliability_weight - 0.5) * 0.2
        
        # Adjust based on goal complexity
        complexity_penalty = context_factors.goal_complexity * 0.1
        
        # Adjust based on resource availability
        resource_score = self._calculate_resource_compatibility(template, context_factors)
        resource_adjustment = (resource_score - 0.5) * 0.1
        
        estimated_rate = base_rate + reliability_adjustment - complexity_penalty + resource_adjustment
        return max(min(estimated_rate, 1.0), 0.0)
    
    def _estimate_execution_duration(self, template: StrategyTemplate, 
                                   context_factors: ContextFactors,
                                   goal_decomposition: Dict[str, Any]) -> int:
        """Estimate execution duration in seconds"""
        base_duration = context_factors.historical_performance["average_duration"]
        
        # Adjust based on goal complexity
        complexity_multiplier = 1.0 + (context_factors.goal_complexity * 0.5)
        
        # Adjust based on strategy speed weight
        speed_multiplier = 2.0 - template.speed_weight
        
        # Adjust based on number of sub-goals
        sub_goal_count = len(goal_decomposition.get("sub_goals", []))
        sub_goal_multiplier = 1.0 + (sub_goal_count * 0.1)
        
        # Adjust based on network conditions
        network_multiplier = 2.0 - context_factors.network_conditions["stability"]
        
        estimated_duration = base_duration * complexity_multiplier * speed_multiplier * sub_goal_multiplier * network_multiplier
        
        return int(estimated_duration)
    
    def _calculate_resource_cost(self, template: StrategyTemplate, 
                               context_factors: ContextFactors) -> float:
        """Calculate relative resource cost"""
        cpu_cost = template.resource_requirements.get("cpu_usage", 0.5)
        memory_cost = template.resource_requirements.get("memory_usage", 0.5)
        network_cost = template.resource_requirements.get("network_usage", 0.5)
        
        # Weight the costs
        total_cost = (cpu_cost * 0.4) + (memory_cost * 0.3) + (network_cost * 0.3)
        
        # Adjust for parallelization
        max_parallel = template.resource_requirements.get("max_parallel_tasks", 1)
        if max_parallel > 1:
            total_cost *= (1.0 + (max_parallel - 1) * 0.2)
        
        return min(total_cost, 1.0)
    
    def _assess_strategy_risks(self, template: StrategyTemplate, 
                             context_factors: ContextFactors) -> Dict[str, float]:
        """Assess various risks for the strategy"""
        risks = {
            "execution_failure": 0.0,
            "performance_degradation": 0.0,
            "resource_exhaustion": 0.0,
            "security_breach": 0.0,
            "compliance_violation": 0.0
        }
        
        # Execution failure risk
        risks["execution_failure"] = 1.0 - self._estimate_success_rate(template, context_factors, {})
        
        # Performance degradation risk
        if template.resource_requirements.get("cpu_usage", 0.5) > context_factors.resource_availability["cpu"]:
            risks["performance_degradation"] += 0.3
        
        # Resource exhaustion risk
        total_resource_usage = sum([
            template.resource_requirements.get("cpu_usage", 0.5),
            template.resource_requirements.get("memory_usage", 0.5),
            template.resource_requirements.get("network_usage", 0.5)
        ]) / 3.0
        
        if total_resource_usage > 0.8:
            risks["resource_exhaustion"] = 0.4
        
        # Security breach risk
        if context_factors.security_requirements and template.risk_tolerance > 0.5:
            risks["security_breach"] = 0.2
        
        # Compliance violation risk
        if context_factors.compliance_needs and template.strategy_type == StrategyType.AGGRESSIVE:
            risks["compliance_violation"] = 0.3
        
        return risks
    
    def _generate_pros_cons(self, template: StrategyTemplate, 
                          context_factors: ContextFactors) -> Tuple[List[str], List[str]]:
        """Generate pros and cons for the strategy"""
        pros = []
        cons = []
        
        # Strategy type specific pros/cons
        if template.strategy_type == StrategyType.CONSERVATIVE:
            pros.extend(["High reliability", "Low error rate", "Predictable execution"])
            cons.extend(["Slower execution", "Higher resource usage per task"])
        
        elif template.strategy_type == StrategyType.AGGRESSIVE:
            pros.extend(["Fast execution", "High throughput", "Efficient resource usage"])
            cons.extend(["Higher error risk", "Less predictable", "May overwhelm target site"])
        
        elif template.strategy_type == StrategyType.PARALLEL:
            pros.extend(["Concurrent execution", "Better resource utilization"])
            cons.extend(["Complex coordination", "Higher memory usage"])
        
        elif template.strategy_type == StrategyType.ADAPTIVE:
            pros.extend(["Context-aware", "Balanced approach", "Self-optimizing"])
            cons.extend(["Complex implementation", "Overhead from adaptation"])
        
        # Context-specific pros/cons
        if context_factors.time_constraints and context_factors.time_constraints < 300:
            if template.speed_weight > 0.7:
                pros.append("Well-suited for time constraints")
            else:
                cons.append("May not meet time requirements")
        
        if context_factors.security_requirements:
            if template.risk_tolerance < 0.3:
                pros.append("Security-conscious approach")
            else:
                cons.append("May not meet security requirements")
        
        return pros, cons
    
    def _generate_strategy_recommendations(self, template: StrategyTemplate, 
                                         context_factors: ContextFactors) -> List[str]:
        """Generate recommendations for strategy optimization"""
        recommendations = []
        
        # Resource optimization recommendations
        if template.resource_requirements.get("cpu_usage", 0.5) > context_factors.resource_availability["cpu"]:
            recommendations.append("Consider reducing parallelization to match CPU availability")
        
        # Network optimization recommendations
        if context_factors.network_conditions["stability"] < 0.7:
            recommendations.append("Increase timeout values and retry attempts for unstable network")
        
        # Security recommendations
        if context_factors.security_requirements and template.risk_tolerance > 0.5:
            recommendations.append("Consider using a more conservative strategy for security-sensitive operations")
        
        # Performance recommendations
        if context_factors.time_constraints and template.speed_weight < 0.5:
            recommendations.append("Consider optimizing for speed to meet time constraints")
        
        # Mobile optimization recommendations
        if context_factors.device_capabilities["is_mobile"]:
            recommendations.append("Optimize for mobile interaction patterns and touch interfaces")
        
        return recommendations
    
    def _generate_selection_reasoning(self, best_strategy: StrategyEvaluation, 
                                    context_factors: ContextFactors) -> str:
        """Generate human-readable reasoning for strategy selection"""
        reasoning_parts = [
            f"Selected '{best_strategy.strategy_name}' with suitability score {best_strategy.suitability_score:.2f}"
        ]
        
        # Add key factors
        if context_factors.time_constraints and context_factors.time_constraints < 300:
            reasoning_parts.append("Time constraints favor speed-optimized execution")
        
        if context_factors.security_requirements:
            reasoning_parts.append("Security requirements favor conservative approach")
        
        if context_factors.goal_complexity > 0.7:
            reasoning_parts.append("High goal complexity requires robust error handling")
        
        if context_factors.resource_availability["cpu"] > 0.8:
            reasoning_parts.append("High resource availability enables parallel execution")
        
        return ". ".join(reasoning_parts)
    
    def _get_fallback_strategy(self, goal_decomposition: Dict[str, Any], 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Get a safe fallback strategy when selection fails"""
        fallback_template = self.strategy_templates[0]  # Conservative sequential
        
        return {
            "selected_strategy": asdict(fallback_template),
            "evaluation": {
                "strategy_name": fallback_template.name,
                "suitability_score": 0.5,
                "confidence_level": 0.3,
                "estimated_success_rate": 0.7,
                "estimated_duration": 300,
                "resource_cost": 0.4,
                "risk_assessment": {"execution_failure": 0.3},
                "pros": ["Safe fallback option"],
                "cons": ["May not be optimal"],
                "recommendations": ["Review strategy selection logic"]
            },
            "alternatives": [],
            "context_analysis": {},
            "selection_reasoning": "Fallback strategy selected due to selection failure"
        }
    
    def update_performance_history(self, site_url: str, strategy_name: str, 
                                 execution_result: Dict[str, Any]):
        """Update performance history with execution results"""
        if site_url not in self.performance_history:
            self.performance_history[site_url] = {
                "success_rate": 0.85,
                "average_duration": 120,
                "error_rate": 0.15,
                "retry_rate": 0.10,
                "strategy_performance": {}
            }
        
        site_history = self.performance_history[site_url]
        
        # Update strategy-specific performance
        if strategy_name not in site_history["strategy_performance"]:
            site_history["strategy_performance"][strategy_name] = {
                "executions": 0,
                "successes": 0,
                "total_duration": 0,
                "errors": 0
            }
        
        strategy_perf = site_history["strategy_performance"][strategy_name]
        strategy_perf["executions"] += 1
        
        if execution_result.get("success", False):
            strategy_perf["successes"] += 1
        else:
            strategy_perf["errors"] += 1
        
        strategy_perf["total_duration"] += execution_result.get("duration", 0)
        
        # Update overall site performance
        total_executions = sum(perf["executions"] for perf in site_history["strategy_performance"].values())
        total_successes = sum(perf["successes"] for perf in site_history["strategy_performance"].values())
        total_duration = sum(perf["total_duration"] for perf in site_history["strategy_performance"].values())
        total_errors = sum(perf["errors"] for perf in site_history["strategy_performance"].values())
        
        if total_executions > 0:
            site_history["success_rate"] = total_successes / total_executions
            site_history["error_rate"] = total_errors / total_executions
            site_history["average_duration"] = total_duration / total_executions
        
        logger.info(f"Updated performance history for {site_url} with strategy {strategy_name}")