import asyncio
import time
import psutil
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Core Enums
class SystemHealthStatus(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILING = "failing"

class AdaptationTrigger(Enum):
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ERROR_RATE_SPIKE = "error_rate_spike"
    RESOURCE_CONSTRAINT = "resource_constraint"
    SUCCESS_RATE_DROP = "success_rate_drop"
    ENVIRONMENTAL_CHANGE = "environmental_change"
    COST_ESCALATION = "cost_escalation"
    TIMEOUT_INCREASE = "timeout_increase"
    MEMORY_PRESSURE = "memory_pressure"
    NETWORK_ISSUES = "network_issues"

class AdaptationSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class InterventionType(Enum):
    PREEMPTIVE_RESOURCE_OPTIMIZATION = "preemptive_resource_optimization"
    PROACTIVE_STRATEGY_ADJUSTMENT = "proactive_strategy_adjustment"
    PREVENTIVE_ERROR_HANDLING = "preventive_error_handling"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    COST_CONTROL_INTERVENTION = "cost_control_intervention"

# Core Data Classes
@dataclass
class SystemState:
    """Comprehensive system state snapshot"""
    timestamp: datetime
    health_status: SystemHealthStatus
    resource_metrics: Dict[str, float]
    execution_metrics: Dict[str, Any]
    environment_factors: Dict[str, Any]
    active_goals: List[Dict[str, Any]]
    error_patterns: List[str]
    performance_indicators: Dict[str, float]
    adaptation_history: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)

@dataclass
class AdaptationEvent:
    """Represents a detected adaptation trigger event"""
    event_id: str
    trigger: AdaptationTrigger
    severity: AdaptationSeverity
    timestamp: datetime
    context: Dict[str, Any]
    affected_components: List[str]
    suggested_adaptations: List[Dict[str, Any]]
    confidence_score: float
    urgency_score: float
    impact_assessment: Dict[str, Any]

@dataclass
class InterventionPlan:
    """Detailed intervention execution plan"""
    intervention_id: str
    intervention_type: InterventionType
    steps: List[Dict[str, Any]]
    expected_impact: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    rollback_plan: Dict[str, Any]
    estimated_duration: float
    resource_requirements: Dict[str, Any]

# Advanced Real-time Monitoring System
class LiveMonitoringSystem:
    """Real-time system monitoring and adaptation trigger detection"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # State tracking
        self.current_state: Optional[SystemState] = None
        self.state_history: deque = deque(maxlen=1000)
        self.adaptation_events: List[AdaptationEvent] = []
        
        # Metrics tracking
        self.performance_baselines: Dict[str, float] = {}
        self.error_rate_thresholds: Dict[str, float] = {
            "warning": 0.05,
            "critical": 0.15
        }
        self.performance_thresholds: Dict[str, float] = {
            "response_time_warning": 5.0,
            "response_time_critical": 10.0,
            "success_rate_warning": 0.8,
            "success_rate_critical": 0.6
        }
        
        # Event callbacks
        self.adaptation_callbacks: List[callable] = []
        
        # Initialize monitoring components
        self._initialize_monitoring_components()
    
    def _initialize_monitoring_components(self):
        """Initialize monitoring subsystems"""
        # Performance trend analyzers
        self.trend_analyzers = {
            "response_time": self._analyze_response_time_trend,
            "success_rate": self._analyze_success_rate_trend,
            "error_rate": self._analyze_error_rate_trend,
            "resource_usage": self._analyze_resource_usage_trend
        }
        
        # Pattern detectors
        self.pattern_detectors = {
            "error_patterns": self._detect_error_patterns,
            "performance_patterns": self._detect_performance_patterns,
            "resource_patterns": self._detect_resource_patterns
        }
    
    async def start_monitoring(self):
        """Start real-time monitoring"""
        if self.is_monitoring:
            logger.warning("Monitoring already active")
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Real-time monitoring started")
    
    async def stop_monitoring(self):
        """Stop real-time monitoring"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Real-time monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while self.is_monitoring:
                # Capture current system state
                current_state = await self._capture_system_state()
                
                # Update state history
                self.current_state = current_state
                self.state_history.append(current_state)
                
                # Detect adaptation triggers
                await self._detect_adaptation_triggers()
                
                # Sleep until next monitoring cycle
                await asyncio.sleep(self.monitoring_interval)
                
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Monitoring loop error: {e}")
            self.is_monitoring = False
    
    async def _capture_system_state(self) -> SystemState:
        """Capture comprehensive system state"""
        timestamp = datetime.now()
        
        # Gather metrics
        resource_metrics = await self._gather_resource_metrics()
        execution_metrics = await self._gather_execution_metrics()
        environment_factors = await self._gather_environment_factors()
        
        # Analyze performance indicators
        performance_indicators = await self._analyze_performance_indicators(
            resource_metrics, execution_metrics
        )
        
        # Determine health status
        health_status = await self._determine_health_status(
            resource_metrics, execution_metrics, performance_indicators
        )
        
        # Detect error patterns
        error_patterns = await self._detect_current_error_patterns()
        
        return SystemState(
            timestamp=timestamp,
            health_status=health_status,
            resource_metrics=resource_metrics,
            execution_metrics=execution_metrics,
            environment_factors=environment_factors,
            active_goals=[],  # To be populated by goal processor
            error_patterns=error_patterns,
            performance_indicators=performance_indicators
        )
    
    async def _gather_execution_metrics(self) -> Dict[str, Any]:
        """Gather execution-specific metrics"""
        # This would integrate with the actual execution system
        # For now, return mock data structure
        return {
            "average_response_time": 2.5,
            "success_rate": 0.95,
            "error_rate": 0.05,
            "total_requests": 100,
            "failed_requests": 5,
            "timeout_rate": 0.02,
            "retry_rate": 0.08,
            "cost_per_request": 0.001,
            "concurrent_executions": 3
        }
    
    async def _gather_resource_metrics(self) -> Dict[str, float]:
        """Gather system resource utilization metrics"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_used_mb": psutil.virtual_memory().used / (1024 * 1024),
                "disk_io_read_mb": psutil.disk_io_counters().read_bytes / (1024 * 1024) if psutil.disk_io_counters() else 0,
                "disk_io_write_mb": psutil.disk_io_counters().write_bytes / (1024 * 1024) if psutil.disk_io_counters() else 0,
                "network_sent_mb": psutil.net_io_counters().bytes_sent / (1024 * 1024) if psutil.net_io_counters() else 0,
                "network_recv_mb": psutil.net_io_counters().bytes_recv / (1024 * 1024) if psutil.net_io_counters() else 0,
                "open_connections": len(psutil.net_connections()),
                "process_threads": psutil.Process().num_threads()
            }
        except Exception as e:
            # Return safe defaults if system monitoring fails
            return {
                "cpu_percent": 50.0,
                "memory_percent": 50.0,
                "memory_used_mb": 1024.0,
                "disk_io_read_mb": 0.0,
                "disk_io_write_mb": 0.0,
                "network_sent_mb": 0.0,
                "network_recv_mb": 0.0,
                "open_connections": 0,
                "process_threads": 1
            }
    
    async def _gather_environment_factors(self) -> Dict[str, Any]:
        """Gather environmental context factors"""
        return {
            "time_of_day": datetime.now().hour,
            "day_of_week": datetime.now().weekday(),
            "system_load": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 1.0,
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
            "network_latency_estimate": 50.0,  # Would be measured
            "concurrent_processes": len(psutil.pids())
        }
    
    async def _analyze_performance_indicators(self, resource_metrics: Dict[str, float],
                                            execution_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Analyze key performance indicators"""
        indicators = {}
        
        # Response time performance
        avg_response = execution_metrics.get("average_response_time", 0)
        indicators["response_time_score"] = max(0, 1 - (avg_response / 10.0))
        
        # Success rate performance
        success_rate = execution_metrics.get("success_rate", 1.0)
        indicators["success_rate_score"] = success_rate
        
        # Resource efficiency
        cpu_usage = resource_metrics.get("cpu_percent", 0) / 100.0
        memory_usage = resource_metrics.get("memory_percent", 0) / 100.0
        indicators["resource_efficiency_score"] = 1 - ((cpu_usage + memory_usage) / 2)
        
        # Error rate performance
        error_rate = execution_metrics.get("error_rate", 0)
        indicators["error_rate_score"] = max(0, 1 - (error_rate * 10))
        
        # Overall performance score
        indicators["overall_performance_score"] = np.mean([
            indicators["response_time_score"],
            indicators["success_rate_score"],
            indicators["resource_efficiency_score"],
            indicators["error_rate_score"]
        ])
        
        return indicators
    
    async def _determine_health_status(self, resource_metrics: Dict[str, float],
                                     execution_metrics: Dict[str, Any],
                                     performance_indicators: Dict[str, float]) -> SystemHealthStatus:
        """Determine overall system health status"""
        overall_score = performance_indicators.get("overall_performance_score", 0.5)
        
        if overall_score >= 0.9:
            return SystemHealthStatus.EXCELLENT
        elif overall_score >= 0.7:
            return SystemHealthStatus.GOOD
        elif overall_score >= 0.5:
            return SystemHealthStatus.DEGRADED
        elif overall_score >= 0.3:
            return SystemHealthStatus.CRITICAL
        else:
            return SystemHealthStatus.FAILING
    
    async def _detect_current_error_patterns(self) -> List[str]:
        """Detect current error patterns"""
        # This would analyze recent errors and identify patterns
        # For now, return empty list
        return []
    
    async def _detect_adaptation_triggers(self):
        """Detect conditions that warrant adaptation"""
        
        if len(self.state_history) < 10:  # Need sufficient history
            return
        
        current = self.current_state
        recent_states = list(self.state_history)[-10:]  # Last 10 states
        
        # Performance degradation detection
        await self._detect_performance_degradation(current, recent_states)
        
        # Error pattern detection
        await self._detect_error_patterns(current, recent_states)
        
        # Resource constraint detection
        await self._detect_resource_constraints(current, recent_states)
        
        # Environmental change detection
        await self._detect_environmental_changes(current, recent_states)
        
        # Success probability change detection
        await self._detect_success_probability_changes(current, recent_states)
    
    async def _detect_performance_degradation(self, current: SystemState, 
                                            recent_states: List[SystemState]):
        """Detect performance degradation patterns"""
        
        # Calculate performance trends
        response_times = [s.execution_metrics.get("average_response_time", 0) for s in recent_states]
        success_rates = [s.execution_metrics.get("success_rate", 1.0) for s in recent_states]
        error_rates = [s.execution_metrics.get("error_rate", 0) for s in recent_states]
        
        # Check for degradation trends
        degradation_signals = []
        
        # Response time trending up
        if len(response_times) >= 5:
            recent_avg = np.mean(response_times[-3:])
            baseline_avg = np.mean(response_times[:3])
            if recent_avg > baseline_avg * 1.5:  # 50% increase
                degradation_signals.append("response_time_increase")
        
        # Success rate trending down
        if len(success_rates) >= 5:
            recent_avg = np.mean(success_rates[-3:])
            baseline_avg = np.mean(success_rates[:3])
            if recent_avg < baseline_avg * 0.8:  # 20% decrease
                degradation_signals.append("success_rate_decrease")
        
        # Error rate trending up
        if len(error_rates) >= 5:
            recent_avg = np.mean(error_rates[-3:])
            baseline_avg = np.mean(error_rates[:3])
            if recent_avg > baseline_avg * 2.0:  # 100% increase
                degradation_signals.append("error_rate_increase")
        
        # Generate adaptation events
        if degradation_signals:
            severity = AdaptationSeverity.HIGH if len(degradation_signals) > 1 else AdaptationSeverity.MEDIUM
            
            event = AdaptationEvent(
                event_id=f"perf_deg_{int(time.time())}",
                trigger=AdaptationTrigger.PERFORMANCE_DEGRADATION,
                severity=severity,
                timestamp=datetime.now(),
                context={
                    "degradation_signals": degradation_signals,
                    "performance_metrics": current.execution_metrics,
                    "trend_analysis": {
                        "response_time_trend": self._calculate_trend(response_times),
                        "success_rate_trend": self._calculate_trend(success_rates),
                        "error_rate_trend": self._calculate_trend(error_rates)
                    }
                },
                affected_components=["web_agent", "orchestrator"],
                suggested_adaptations=await self._generate_performance_adaptations(degradation_signals),
                confidence_score=0.8,
                urgency_score=0.7,
                impact_assessment=await self._assess_performance_impact(degradation_signals)
            )
            
            await self._emit_adaptation_event(event)
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values"""
        if len(values) < 2:
            return "stable"
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    async def _generate_performance_adaptations(self, signals: List[str]) -> List[Dict[str, Any]]:
        """Generate specific adaptations for performance issues"""
        adaptations = []
        
        if "response_time_increase" in signals:
            adaptations.extend([
                {
                    "type": "timeout_adjustment",
                    "action": "increase_timeouts",
                    "parameters": {"multiplier": 1.5},
                    "description": "Increase timeouts to accommodate slower responses"
                },
                {
                    "type": "model_optimization",
                    "action": "prefer_faster_model",
                    "parameters": {"model": "gemini-2.5-flash"},
                    "description": "Switch to faster model for simple decisions"
                }
            ])
        
        if "success_rate_decrease" in signals:
            adaptations.extend([
                {
                    "type": "retry_enhancement",
                    "action": "increase_retry_attempts",
                    "parameters": {"additional_attempts": 2},
                    "description": "Increase retry attempts to improve success rate"
                },
                {
                    "type": "strategy_change",
                    "action": "switch_to_conservative",
                    "parameters": {"strategy": "conservative"},
                    "description": "Switch to more conservative execution strategy"
                }
            ])
        
        if "error_rate_increase" in signals:
            adaptations.extend([
                {
                    "type": "error_handling_enhancement",
                    "action": "enable_detailed_error_recovery",
                    "parameters": {"recovery_level": "comprehensive"},
                    "description": "Enable comprehensive error recovery mechanisms"
                },
                {
                    "type": "logging_enhancement",
                    "action": "increase_logging_detail",
                    "parameters": {"level": "debug"},
                    "description": "Increase logging detail for better error analysis"
                }
            ])
        
        return adaptations
    
    async def _assess_performance_impact(self, signals: List[str]) -> Dict[str, Any]:
        """Assess the impact of performance degradation"""
        return {
            "severity_level": "high" if len(signals) > 1 else "medium",
            "affected_metrics": signals,
            "estimated_user_impact": "moderate_to_high",
            "business_impact": "potential_revenue_loss",
            "technical_debt_increase": "low"
        }
    
    async def _detect_error_patterns(self, current: SystemState, recent_states: List[SystemState]):
        """Detect error pattern changes"""
        # Implementation for error pattern detection
        pass
    
    async def _detect_resource_constraints(self, current: SystemState, recent_states: List[SystemState]):
        """Detect resource constraint issues"""
        # Implementation for resource constraint detection
        pass
    
    async def _detect_environmental_changes(self, current: SystemState, recent_states: List[SystemState]):
        """Detect environmental changes"""
        # Implementation for environmental change detection
        pass
    
    async def _detect_success_probability_changes(self, current: SystemState, recent_states: List[SystemState]):
        """Detect changes in success probability"""
        # Implementation for success probability change detection
        pass
    
    async def _emit_adaptation_event(self, event: AdaptationEvent):
        """Emit adaptation event to registered callbacks"""
        self.adaptation_events.append(event)
        
        # Notify all registered callbacks
        for callback in self.adaptation_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Error in adaptation callback: {e}")
    
    def register_adaptation_callback(self, callback: callable):
        """Register callback for adaptation events"""
        self.adaptation_callbacks.append(callback)
    
    def get_current_health_status(self) -> Optional[SystemHealthStatus]:
        """Get current system health status"""
        return self.current_state.health_status if self.current_state else None
    
    def get_recent_adaptation_events(self, limit: int = 10) -> List[AdaptationEvent]:
        """Get recent adaptation events"""
        return self.adaptation_events[-limit:] if self.adaptation_events else []