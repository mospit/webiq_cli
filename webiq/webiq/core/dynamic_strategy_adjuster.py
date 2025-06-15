import asyncio
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
import json

from .real_time_adaptation_system import (
    AdaptationEvent, SystemState, AdaptationSeverity, AdaptationTrigger,
    LiveMonitoringSystem
)

logger = logging.getLogger(__name__)

class DynamicStrategyAdjuster:
    """Real-time strategy modification during execution"""
    
    def __init__(self, monitoring_system: LiveMonitoringSystem):
        self.monitoring_system = monitoring_system
        self.active_adaptations: Dict[str, Dict[str, Any]] = {}
        self.adaptation_history: List[Dict[str, Any]] = []
        self.strategy_modifications: Dict[str, Any] = {}
        
        # Adaptation effectiveness tracking
        self.adaptation_outcomes: Dict[str, List[float]] = defaultdict(list)
        
        # Strategy adjustment rules
        self.adjustment_rules = self._initialize_adjustment_rules()
        
        # Rollback mechanisms
        self.rollback_stack: List[Dict[str, Any]] = []
        
        # Register with monitoring system
        self.monitoring_system.register_adaptation_callback(self.handle_adaptation_event)
    
    def _initialize_adjustment_rules(self) -> Dict[str, Any]:
        """Initialize strategy adjustment rules"""
        return {
            "timeout_adjustment": {
                "min_multiplier": 0.5,
                "max_multiplier": 3.0,
                "step_size": 0.2
            },
            "retry_enhancement": {
                "min_attempts": 1,
                "max_attempts": 8,
                "step_size": 1
            },
            "model_optimization": {
                "available_models": ["gemini-2.5-flash", "gemini-2.5-pro"],
                "performance_thresholds": {
                    "speed_priority": 0.7,
                    "quality_priority": 0.9
                }
            },
            "parallel_execution": {
                "min_workers": 1,
                "max_workers": 10,
                "step_size": 1
            }
        }
    
    async def handle_adaptation_event(self, event: AdaptationEvent):
        """Handle adaptation event from monitoring system"""
        logger.info(f"Handling adaptation event: {event.event_id}")
        
        # Apply adaptation based on event
        result = await self.apply_adaptation(event)
        
        logger.info(f"Adaptation result: {result['status']}")
        return result
    
    async def apply_adaptation(self, adaptation_event: AdaptationEvent) -> Dict[str, Any]:
        """Apply adaptation based on detected event"""
        
        adaptation_id = f"adapt_{int(time.time())}_{adaptation_event.event_id}"
        
        # Validate adaptation applicability
        validation_result = await self._validate_adaptation(adaptation_event)
        if not validation_result["valid"]:
            return {
                "status": "rejected",
                "reason": validation_result["reason"],
                "adaptation_id": adaptation_id
            }
        
        # Select best adaptation from suggestions
        selected_adaptation = await self._select_optimal_adaptation(
            adaptation_event.suggested_adaptations,
            adaptation_event.context
        )
        
        # Create rollback point
        rollback_point = await self._create_rollback_point()
        self.rollback_stack.append(rollback_point)
        
        # Apply the adaptation
        application_result = await self._apply_strategy_modification(
            selected_adaptation,
            adaptation_event.context
        )
        
        if application_result["success"]:
            # Track active adaptation
            self.active_adaptations[adaptation_id] = {
                "adaptation": selected_adaptation,
                "event": adaptation_event,
                "applied_at": datetime.now(),
                "rollback_point": rollback_point,
                "monitoring_metrics": []
            }
            
            # Start monitoring adaptation effectiveness
            await self._start_adaptation_monitoring(adaptation_id)
            
            return {
                "status": "applied",
                "adaptation_id": adaptation_id,
                "modifications": application_result["modifications"],
                "expected_impact": selected_adaptation.get("expected_impact", {})
            }
        else:
            # Rollback on failure
            await self._rollback_to_point(rollback_point)
            return {
                "status": "failed",
                "reason": application_result["error"],
                "adaptation_id": adaptation_id
            }
    
    async def _validate_adaptation(self, event: AdaptationEvent) -> Dict[str, Any]:
        """Validate if adaptation should be applied"""
        
        # Check if similar adaptation is already active
        for adaptation_id, adaptation_info in self.active_adaptations.items():
            if adaptation_info["event"].trigger == event.trigger:
                return {
                    "valid": False,
                    "reason": f"Similar adaptation already active: {adaptation_id}"
                }
        
        # Check confidence threshold
        if event.confidence_score < 0.6:
            return {
                "valid": False,
                "reason": f"Confidence score too low: {event.confidence_score}"
            }
        
        # Check if system is in a state that allows adaptation
        current_health = self.monitoring_system.get_current_health_status()
        if current_health and current_health.value == "failing":
            return {
                "valid": False,
                "reason": "System in failing state, avoiding further changes"
            }
        
        return {"valid": True}
    
    async def _select_optimal_adaptation(self, suggested_adaptations: List[Dict[str, Any]],
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best adaptation from suggestions"""
        
        if not suggested_adaptations:
            return {}
        
        # Score each adaptation
        scored_adaptations = []
        for adaptation in suggested_adaptations:
            score = await self._score_adaptation(adaptation, context)
            scored_adaptations.append((score, adaptation))
        
        # Select highest scoring adaptation
        scored_adaptations.sort(key=lambda x: x[0], reverse=True)
        return scored_adaptations[0][1]
    
    async def _score_adaptation(self, adaptation: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Score an adaptation based on expected effectiveness"""
        base_score = 0.5
        
        adaptation_type = adaptation.get("type", "")
        
        # Historical effectiveness
        if adaptation_type in self.adaptation_outcomes:
            historical_scores = self.adaptation_outcomes[adaptation_type]
            if historical_scores:
                base_score = np.mean(historical_scores)
        
        # Context-specific adjustments
        if adaptation_type == "timeout_adjustment":
            # Prefer timeout adjustments for response time issues
            if "response_time_increase" in context.get("degradation_signals", []):
                base_score += 0.2
        
        elif adaptation_type == "retry_enhancement":
            # Prefer retry enhancements for success rate issues
            if "success_rate_decrease" in context.get("degradation_signals", []):
                base_score += 0.2
        
        elif adaptation_type == "model_optimization":
            # Prefer model optimization for performance issues
            if "response_time_increase" in context.get("degradation_signals", []):
                base_score += 0.15
        
        return min(1.0, max(0.0, base_score))
    
    async def _create_rollback_point(self) -> Dict[str, Any]:
        """Create a rollback point for current strategy state"""
        return {
            "timestamp": datetime.now(),
            "strategy_modifications": self.strategy_modifications.copy(),
            "active_adaptations": list(self.active_adaptations.keys()),
            "system_state_snapshot": {
                "health_status": self.monitoring_system.get_current_health_status(),
                "recent_events": len(self.monitoring_system.get_recent_adaptation_events())
            }
        }
    
    async def _apply_strategy_modification(self, adaptation: Dict[str, Any],
                                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply specific strategy modification"""
        
        try:
            modifications = {}
            adaptation_type = adaptation["type"]
            action = adaptation["action"]
            parameters = adaptation.get("parameters", {})
            
            if adaptation_type == "timeout_adjustment":
                modifications.update(await self._adjust_timeouts(action, parameters))
            
            elif adaptation_type == "retry_enhancement":
                modifications.update(await self._adjust_retry_strategy(action, parameters))
            
            elif adaptation_type == "model_optimization":
                modifications.update(await self._adjust_model_selection(action, parameters))
            
            elif adaptation_type == "strategy_change":
                modifications.update(await self._change_execution_strategy(action, parameters))
            
            elif adaptation_type == "error_handling_enhancement":
                modifications.update(await self._enhance_error_handling(action, parameters))
            
            elif adaptation_type == "resource_optimization":
                modifications.update(await self._optimize_resource_usage(action, parameters))
            
            elif adaptation_type == "parallel_execution_adjustment":
                modifications.update(await self._adjust_parallel_execution(action, parameters))
            
            # Apply modifications to current strategy
            await self._update_current_strategy(modifications)
            
            return {
                "success": True,
                "modifications": modifications
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _adjust_timeouts(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust timeout configurations"""
        modifications = {}
        
        if action == "increase_timeouts":
            multiplier = parameters.get("multiplier", 1.5)
            modifications["timeout_multiplier"] = multiplier
            modifications["page_load_timeout"] = 30 * multiplier
            modifications["element_wait_timeout"] = 10 * multiplier
            modifications["action_timeout"] = 5 * multiplier
            
        elif action == "decrease_timeouts":
            multiplier = parameters.get("multiplier", 0.8)
            modifications["timeout_multiplier"] = multiplier
            modifications["page_load_timeout"] = 30 * multiplier
            modifications["element_wait_timeout"] = 10 * multiplier
            modifications["action_timeout"] = 5 * multiplier
        
        elif action == "adaptive_timeouts":
            # Implement adaptive timeout based on recent performance
            recent_response_times = await self._get_recent_response_times()
            if recent_response_times:
                avg_response = np.mean(recent_response_times)
                adaptive_multiplier = max(1.0, avg_response / 2.0)
                modifications["timeout_multiplier"] = adaptive_multiplier
        
        return modifications
    
    async def _adjust_retry_strategy(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust retry strategy configurations"""
        modifications = {}
        
        if action == "increase_retry_attempts":
            additional = parameters.get("additional_attempts", 2)
            current_attempts = self.strategy_modifications.get("retry_attempts", 3)
            modifications["retry_attempts"] = min(current_attempts + additional, 8)
            
        elif action == "decrease_retry_attempts":
            reduction = parameters.get("reduction", 1)
            current_attempts = self.strategy_modifications.get("retry_attempts", 3)
            modifications["retry_attempts"] = max(current_attempts - reduction, 1)
            
        elif action == "adaptive_retry":
            # Implement intelligent retry based on error types
            recent_errors = await self._get_recent_error_types()
            if recent_errors:
                retry_strategy = await self._calculate_optimal_retry_strategy(recent_errors)
                modifications.update(retry_strategy)
        
        elif action == "exponential_backoff":
            modifications["retry_strategy"] = "exponential_backoff"
            modifications["initial_retry_delay"] = parameters.get("initial_delay", 1.0)
            modifications["max_retry_delay"] = parameters.get("max_delay", 30.0)
            modifications["backoff_multiplier"] = parameters.get("multiplier", 2.0)
        
        return modifications
    
    async def _adjust_model_selection(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust AI model selection strategy"""
        modifications = {}
        
        if action == "prefer_faster_model":
            target_model = parameters.get("model", "gemini-2.5-flash")
            modifications["preferred_model"] = target_model
            modifications["model_selection_strategy"] = "speed_optimized"
            
        elif action == "prefer_quality_model":
            target_model = parameters.get("model", "gemini-2.5-pro")
            modifications["preferred_model"] = target_model
            modifications["model_selection_strategy"] = "quality_optimized"
            
        elif action == "adaptive_model_selection":
            # Implement context-aware model selection
            current_context = await self._get_current_execution_context()
            optimal_model = await self._determine_optimal_model(current_context)
            modifications["preferred_model"] = optimal_model
            modifications["model_selection_strategy"] = "context_adaptive"
        
        return modifications
    
    async def _change_execution_strategy(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Change overall execution strategy"""
        modifications = {}
        
        if action == "switch_to_conservative":
            modifications["execution_strategy"] = "conservative"
            modifications["risk_tolerance"] = "low"
            modifications["verification_level"] = "high"
            
        elif action == "switch_to_aggressive":
            modifications["execution_strategy"] = "aggressive"
            modifications["risk_tolerance"] = "high"
            modifications["verification_level"] = "low"
            
        elif action == "switch_to_balanced":
            modifications["execution_strategy"] = "balanced"
            modifications["risk_tolerance"] = "medium"
            modifications["verification_level"] = "medium"
        
        return modifications
    
    async def _enhance_error_handling(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance error handling mechanisms"""
        modifications = {}
        
        if action == "enable_detailed_error_recovery":
            recovery_level = parameters.get("recovery_level", "comprehensive")
            modifications["error_recovery_level"] = recovery_level
            modifications["detailed_error_logging"] = True
            modifications["error_context_capture"] = True
            
        elif action == "increase_error_tolerance":
            tolerance_level = parameters.get("tolerance_level", "high")
            modifications["error_tolerance"] = tolerance_level
            modifications["continue_on_non_critical_errors"] = True
        
        return modifications
    
    async def _optimize_resource_usage(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize resource usage patterns"""
        modifications = {}
        
        if action == "reduce_memory_usage":
            modifications["memory_optimization"] = True
            modifications["cache_size_limit"] = parameters.get("cache_limit", 100)
            modifications["garbage_collection_frequency"] = "high"
            
        elif action == "reduce_cpu_usage":
            modifications["cpu_optimization"] = True
            modifications["parallel_execution_limit"] = parameters.get("cpu_limit", 2)
            modifications["processing_delay"] = parameters.get("delay", 0.1)
        
        return modifications
    
    async def _adjust_parallel_execution(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust parallel execution settings"""
        modifications = {}
        
        if action == "increase_parallelism":
            additional_workers = parameters.get("additional_workers", 2)
            current_workers = self.strategy_modifications.get("parallel_workers", 3)
            modifications["parallel_workers"] = min(current_workers + additional_workers, 10)
            
        elif action == "decrease_parallelism":
            reduction = parameters.get("reduction", 1)
            current_workers = self.strategy_modifications.get("parallel_workers", 3)
            modifications["parallel_workers"] = max(current_workers - reduction, 1)
        
        return modifications
    
    async def _update_current_strategy(self, modifications: Dict[str, Any]):
        """Update current strategy with modifications"""
        self.strategy_modifications.update(modifications)
        logger.info(f"Strategy updated with modifications: {list(modifications.keys())}")
    
    async def _start_adaptation_monitoring(self, adaptation_id: str):
        """Start monitoring adaptation effectiveness"""
        # This would start a background task to monitor the adaptation
        # For now, just log the start
        logger.info(f"Started monitoring adaptation: {adaptation_id}")
    
    async def _rollback_to_point(self, rollback_point: Dict[str, Any]):
        """Rollback to a previous strategy state"""
        self.strategy_modifications = rollback_point["strategy_modifications"].copy()
        logger.info(f"Rolled back to point: {rollback_point['timestamp']}")
    
    async def _get_recent_response_times(self) -> List[float]:
        """Get recent response times for analysis"""
        # This would get actual response times from the monitoring system
        return [2.1, 2.3, 2.8, 3.1, 2.9]  # Mock data
    
    async def _get_recent_error_types(self) -> List[str]:
        """Get recent error types for analysis"""
        # This would get actual error types from the monitoring system
        return ["timeout", "element_not_found", "network_error"]  # Mock data
    
    async def _calculate_optimal_retry_strategy(self, error_types: List[str]) -> Dict[str, Any]:
        """Calculate optimal retry strategy based on error types"""
        strategy = {}
        
        if "timeout" in error_types:
            strategy["retry_attempts"] = 5
            strategy["retry_delay"] = 2.0
        
        if "network_error" in error_types:
            strategy["retry_attempts"] = 3
            strategy["retry_delay"] = 1.0
        
        return strategy
    
    async def _get_current_execution_context(self) -> Dict[str, Any]:
        """Get current execution context"""
        # This would get actual execution context
        return {
            "complexity": "medium",
            "site_type": "e-commerce",
            "user_priority": "speed"
        }
    
    async def _determine_optimal_model(self, context: Dict[str, Any]) -> str:
        """Determine optimal model based on context"""
        if context.get("user_priority") == "speed":
            return "gemini-2.5-flash"
        elif context.get("complexity") == "high":
            return "gemini-2.5-pro"
        else:
            return "gemini-2.5-flash"
    
    async def monitor_adaptation_effectiveness(self, adaptation_id: str) -> Dict[str, Any]:
        """Monitor the effectiveness of an applied adaptation"""
        
        if adaptation_id not in self.active_adaptations:
            return {"error": "Adaptation not found"}
        
        adaptation_info = self.active_adaptations[adaptation_id]
        
        # Collect performance metrics since adaptation
        metrics_since_adaptation = await self._collect_metrics_since_timestamp(
            adaptation_info["applied_at"]
        )
        
        # Compare with baseline performance
        baseline_metrics = await self._get_baseline_metrics()
        effectiveness_analysis = await self._analyze_adaptation_effectiveness(
            baseline_metrics, metrics_since_adaptation, adaptation_info["adaptation"]
        )
        
        # Update adaptation tracking
        adaptation_info["monitoring_metrics"].append({
            "timestamp": datetime.now(),
            "metrics": metrics_since_adaptation,
            "effectiveness": effectiveness_analysis
        })
        
        # Decide if adaptation should continue, be modified, or rolled back
        decision = await self._make_adaptation_decision(effectiveness_analysis, adaptation_info)
        
        if decision["action"] == "rollback":
            await self._rollback_adaptation(adaptation_id)
            return {
                "status": "rolled_back",
                "reason": decision["reason"],
                "effectiveness": effectiveness_analysis
            }
        
        elif decision["action"] == "modify":
            await self._modify_active_adaptation(adaptation_id, decision["modifications"])
            return {
                "status": "modified",
                "modifications": decision["modifications"],
                "effectiveness": effectiveness_analysis
            }
        
        return {
            "status": "continuing",
            "effectiveness": effectiveness_analysis,
            "recommendation": decision.get("recommendation", "")
        }
    
    async def _collect_metrics_since_timestamp(self, timestamp: datetime) -> Dict[str, Any]:
        """Collect metrics since a specific timestamp"""
        # This would collect actual metrics from the monitoring system
        return {
            "average_response_time": 2.2,
            "success_rate": 0.96,
            "error_rate": 0.04,
            "resource_usage": 0.65
        }
    
    async def _get_baseline_metrics(self) -> Dict[str, Any]:
        """Get baseline performance metrics"""
        # This would get actual baseline metrics
        return {
            "average_response_time": 2.8,
            "success_rate": 0.92,
            "error_rate": 0.08,
            "resource_usage": 0.75
        }
    
    async def _analyze_adaptation_effectiveness(self, baseline: Dict[str, Any],
                                              current: Dict[str, Any],
                                              adaptation: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze adaptation effectiveness"""
        effectiveness = {}
        
        # Response time improvement
        response_time_improvement = (baseline["average_response_time"] - current["average_response_time"]) / baseline["average_response_time"]
        effectiveness["response_time_improvement"] = response_time_improvement
        
        # Success rate improvement
        success_rate_improvement = current["success_rate"] - baseline["success_rate"]
        effectiveness["success_rate_improvement"] = success_rate_improvement
        
        # Error rate improvement
        error_rate_improvement = baseline["error_rate"] - current["error_rate"]
        effectiveness["error_rate_improvement"] = error_rate_improvement
        
        # Overall effectiveness score
        effectiveness["overall_score"] = np.mean([
            max(0, response_time_improvement),
            max(0, success_rate_improvement * 10),  # Scale to similar range
            max(0, error_rate_improvement * 10)     # Scale to similar range
        ])
        
        return effectiveness
    
    async def _make_adaptation_decision(self, effectiveness: Dict[str, Any],
                                      adaptation_info: Dict[str, Any]) -> Dict[str, Any]:
        """Make decision about adaptation continuation"""
        overall_score = effectiveness["overall_score"]
        
        if overall_score < -0.1:  # Significant degradation
            return {
                "action": "rollback",
                "reason": "Adaptation causing performance degradation"
            }
        elif overall_score < 0.05:  # Minimal improvement
            return {
                "action": "modify",
                "modifications": {"intensity": "increase"},
                "reason": "Adaptation showing minimal improvement"
            }
        else:
            return {
                "action": "continue",
                "recommendation": "Adaptation showing positive results"
            }
    
    async def _rollback_adaptation(self, adaptation_id: str):
        """Rollback a specific adaptation"""
        if adaptation_id in self.active_adaptations:
            adaptation_info = self.active_adaptations[adaptation_id]
            rollback_point = adaptation_info["rollback_point"]
            await self._rollback_to_point(rollback_point)
            del self.active_adaptations[adaptation_id]
            logger.info(f"Rolled back adaptation: {adaptation_id}")
    
    async def _modify_active_adaptation(self, adaptation_id: str, modifications: Dict[str, Any]):
        """Modify an active adaptation"""
        if adaptation_id in self.active_adaptations:
            # Apply modifications to the adaptation
            adaptation_info = self.active_adaptations[adaptation_id]
            # Update the adaptation with new parameters
            logger.info(f"Modified adaptation {adaptation_id} with: {modifications}")
    
    def get_active_adaptations(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active adaptations"""
        return self.active_adaptations.copy()
    
    def get_current_strategy_modifications(self) -> Dict[str, Any]:
        """Get current strategy modifications"""
        return self.strategy_modifications.copy()