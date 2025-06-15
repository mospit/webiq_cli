"""Performance Agent - Meta-agent for real-time monitoring, pattern analysis, and predictive optimization"""

import asyncio
import statistics
import psutil
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics"""
    EXECUTION_TIME = "execution_time"
    SUCCESS_RATE = "success_rate"
    COST = "cost"
    RESOURCE_USAGE = "resource_usage"
    ERROR_RATE = "error_rate"
    RESPONSE_TIME = "response_time"


class OptimizationType(Enum):
    """Types of optimization suggestions"""
    MODEL_SELECTION = "model_selection"
    TIMEOUT_ADJUSTMENT = "timeout_adjustment"
    ACTION_CONSOLIDATION = "action_consolidation"
    RESOURCE_ALLOCATION = "resource_allocation"
    CACHING_STRATEGY = "caching_strategy"
    PARALLEL_EXECUTION = "parallel_execution"


@dataclass
class PerformanceMetric:
    """Individual performance measurement"""
    metric_id: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    agent_id: str
    session_id: str
    action_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationSuggestion:
    """Optimization recommendation"""
    suggestion_type: str
    description: str
    impact_score: float  # 0.0 to 1.0
    implementation_complexity: str  # "low", "medium", "high"
    estimated_savings: Dict[str, float]  # {"time": seconds, "cost": dollars}
    applicable_context: List[str]  # Which agents/scenarios this applies to
    priority: str = "medium"  # "low", "medium", "high", "critical"
    implementation_steps: List[str] = field(default_factory=list)


class CostTracker:
    """Detailed cost tracking and analysis"""
    
    def __init__(self):
        self.cost_records: List[Dict[str, Any]] = []
        self.model_pricing = {
            "gemini-2.5-pro": {"input": 0.000002, "output": 0.000006},  # Per token
            "gemini-2.5-flash": {"input": 0.000001, "output": 0.000002},
            "gemini-2.0-flash": {"input": 0.0000005, "output": 0.000001}
        }
        self.steel_pricing = {"session_minute": 0.001}  # Per minute
    
    async def track_model_usage(self, model: str, input_tokens: int, 
                              output_tokens: int, session_id: str) -> float:
        """Track individual model API calls"""
        pricing = self.model_pricing.get(model, {"input": 0, "output": 0})
        cost = (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])
        
        record = {
            "timestamp": datetime.now(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "session_id": session_id
        }
        
        self.cost_records.append(record)
        return cost
    
    async def track_browser_usage(self, session_duration_minutes: float, session_id: str) -> float:
        """Track Steel browser session costs"""
        cost = session_duration_minutes * self.steel_pricing["session_minute"]
        
        record = {
            "timestamp": datetime.now(),
            "service": "steel_browser",
            "duration_minutes": session_duration_minutes,
            "cost": cost,
            "session_id": session_id
        }
        
        self.cost_records.append(record)
        return cost
    
    async def analyze_session_costs(self, session_id: str) -> Dict[str, Any]:
        """Detailed cost breakdown for a session"""
        session_records = [r for r in self.cost_records if r["session_id"] == session_id]
        
        cost_breakdown = {
            "total_cost": sum(r["cost"] for r in session_records),
            "model_costs": {},
            "browser_costs": 0,
            "cost_per_action": 0,
            "efficiency_rating": "A"  # A-F rating
        }
        
        # Model cost breakdown
        for record in session_records:
            if "model" in record:
                model = record["model"]
                if model not in cost_breakdown["model_costs"]:
                    cost_breakdown["model_costs"][model] = 0
                cost_breakdown["model_costs"][model] += record["cost"]
            elif record.get("service") == "steel_browser":
                cost_breakdown["browser_costs"] += record["cost"]
        
        # Calculate efficiency rating
        total_cost = cost_breakdown["total_cost"]
        if total_cost < 0.05:
            cost_breakdown["efficiency_rating"] = "A"
        elif total_cost < 0.10:
            cost_breakdown["efficiency_rating"] = "B"
        elif total_cost < 0.20:
            cost_breakdown["efficiency_rating"] = "C"
        else:
            cost_breakdown["efficiency_rating"] = "D"
        
        return cost_breakdown
    
    async def analyze_cost_patterns(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Analyze cost patterns across metrics"""
        session_ids = list(set(m.session_id for m in metrics))
        
        total_cost = 0
        gemini_pro_calls = 0
        gemini_flash_calls = 0
        
        for session_id in session_ids:
            session_records = [r for r in self.cost_records if r["session_id"] == session_id]
            for record in session_records:
                total_cost += record["cost"]
                if "model" in record:
                    if "pro" in record["model"]:
                        gemini_pro_calls += 1
                    elif "flash" in record["model"]:
                        gemini_flash_calls += 1
        
        return {
            "total_cost": total_cost,
            "gemini_pro_calls": gemini_pro_calls,
            "gemini_flash_calls": gemini_flash_calls,
            "cost_efficiency": gemini_flash_calls / max(gemini_pro_calls + gemini_flash_calls, 1)
        }


class ResourceMonitor:
    """Monitor system and application resource usage"""
    
    def __init__(self):
        self.baseline_resources = None
        self.resource_history: List[Dict[str, Any]] = []
    
    async def get_snapshot(self) -> Dict[str, Any]:
        """Get current resource usage snapshot"""
        try:
            snapshot = {
                "timestamp": datetime.now(),
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_used_mb": psutil.virtual_memory().used / (1024 * 1024),
                "disk_io": psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
                "network_io": psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {},
                "open_files": len(psutil.Process().open_files()),
                "threads": psutil.Process().num_threads()
            }
            
            self.resource_history.append(snapshot)
            return snapshot
            
        except Exception as e:
            # Return minimal snapshot if detailed monitoring fails
            return {
                "timestamp": datetime.now(),
                "cpu_percent": 0,
                "memory_percent": 0,
                "error": str(e)
            }
    
    def _calculate_resource_delta(self, pre: Dict[str, Any], post: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate resource usage delta between snapshots"""
        delta = {}
        
        numeric_fields = ["cpu_percent", "memory_percent", "memory_used_mb", "open_files", "threads"]
        
        for field in numeric_fields:
            if field in pre and field in post:
                delta[f"{field}_delta"] = post[field] - pre[field]
        
        return delta
    
    async def analyze_resource_trends(self, time_window_minutes: int = 30) -> Dict[str, Any]:
        """Analyze resource usage trends over time"""
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        recent_snapshots = [s for s in self.resource_history if s["timestamp"] > cutoff_time]
        
        if not recent_snapshots:
            return {"error": "No recent resource data"}
        
        analysis = {
            "avg_cpu": statistics.mean(s["cpu_percent"] for s in recent_snapshots),
            "max_cpu": max(s["cpu_percent"] for s in recent_snapshots),
            "avg_memory": statistics.mean(s["memory_percent"] for s in recent_snapshots),
            "max_memory": max(s["memory_percent"] for s in recent_snapshots),
            "resource_spikes": self._identify_resource_spikes(recent_snapshots),
            "efficiency_score": self._calculate_resource_efficiency(recent_snapshots)
        }
        
        return analysis
    
    def _identify_resource_spikes(self, snapshots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify resource usage spikes"""
        spikes = []
        
        cpu_threshold = 80  # 80% CPU
        memory_threshold = 85  # 85% Memory
        
        for snapshot in snapshots:
            if snapshot["cpu_percent"] > cpu_threshold:
                spikes.append({
                    "type": "cpu_spike",
                    "timestamp": snapshot["timestamp"],
                    "value": snapshot["cpu_percent"]
                })
            
            if snapshot["memory_percent"] > memory_threshold:
                spikes.append({
                    "type": "memory_spike",
                    "timestamp": snapshot["timestamp"],
                    "value": snapshot["memory_percent"]
                })
        
        return spikes
    
    def _calculate_resource_efficiency(self, snapshots: List[Dict[str, Any]]) -> float:
        """Calculate resource efficiency score (0-100)"""
        if not snapshots:
            return 0.0
        
        avg_cpu = statistics.mean(s["cpu_percent"] for s in snapshots)
        avg_memory = statistics.mean(s["memory_percent"] for s in snapshots)
        
        # Efficiency decreases as resource usage increases
        cpu_efficiency = max(0, 100 - avg_cpu)
        memory_efficiency = max(0, 100 - avg_memory)
        
        return (cpu_efficiency + memory_efficiency) / 2


class RealTimeOptimizer:
    """Provides real-time optimization during execution"""
    
    def __init__(self, performance_agent: 'PerformanceAgent'):
        self.performance_agent = performance_agent
        self.active_optimizations: Dict[str, Any] = {}
        self.optimization_history: List[Dict[str, Any]] = []
    
    async def optimize_next_action(self, current_state: Dict[str, Any], 
                                 upcoming_action: str) -> Dict[str, Any]:
        """Optimize the next action based on current performance data"""
        
        # Analyze current performance trends
        recent_metrics = self.performance_agent.get_recent_metrics(minutes=5)
        performance_trend = self._analyze_performance_trend(recent_metrics)
        
        optimizations = {
            "model_selection": await self._optimize_model_selection(upcoming_action, performance_trend),
            "timeout_adjustment": self._optimize_timeouts(performance_trend),
            "parallel_execution": self._suggest_parallel_execution(upcoming_action, current_state),
            "caching_strategy": self._optimize_caching(upcoming_action),
            "resource_allocation": await self._optimize_resource_allocation(performance_trend)
        }
        
        return {
            "optimizations": optimizations,
            "confidence_score": self._calculate_optimization_confidence(optimizations),
            "estimated_improvement": self._estimate_performance_improvement(optimizations)
        }
    
    async def _optimize_model_selection(self, action: str, trend: Dict[str, Any]) -> Dict[str, Any]:
        """Select optimal model based on action complexity and performance trend"""
        
        # Simple actions can use Flash models
        simple_actions = ["click", "type", "scroll", "wait"]
        complex_actions = ["analyze", "plan", "decide", "evaluate"]
        
        if any(simple in action.lower() for simple in simple_actions):
            recommended_model = "gemini-2.5-flash"
            confidence = 0.9
        elif any(complex in action.lower() for complex in complex_actions):
            recommended_model = "gemini-2.5-pro"
            confidence = 0.8
        else:
            # Use trend analysis to decide
            if trend.get("avg_response_time", 0) > 3.0:
                recommended_model = "gemini-2.5-flash"  # Prioritize speed
                confidence = 0.7
            else:
                recommended_model = "gemini-2.5-pro"  # Prioritize quality
                confidence = 0.6
        
        return {
            "recommended_model": recommended_model,
            "confidence": confidence,
            "reasoning": f"Based on action type and performance trend"
        }
    
    def _optimize_timeouts(self, trend: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize timeout values based on performance trends"""
        
        base_timeout = 30  # seconds
        avg_response_time = trend.get("avg_response_time", 2.0)
        
        # Adjust timeout based on recent performance
        if avg_response_time > 5.0:
            recommended_timeout = base_timeout * 1.5
        elif avg_response_time < 1.0:
            recommended_timeout = base_timeout * 0.7
        else:
            recommended_timeout = base_timeout
        
        return {
            "recommended_timeout": recommended_timeout,
            "adjustment_factor": recommended_timeout / base_timeout,
            "reasoning": f"Based on average response time of {avg_response_time:.2f}s"
        }
    
    def _suggest_parallel_execution(self, action: str, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest parallel execution opportunities"""
        
        parallelizable_actions = ["screenshot", "analyze_multiple", "batch_click"]
        
        if any(parallel in action.lower() for parallel in parallelizable_actions):
            return {
                "can_parallelize": True,
                "suggested_threads": 3,
                "reasoning": "Action can be parallelized for better performance"
            }
        
        return {
            "can_parallelize": False,
            "reasoning": "Action requires sequential execution"
        }
    
    def _optimize_caching(self, action: str) -> Dict[str, Any]:
        """Optimize caching strategy"""
        
        cacheable_actions = ["analyze_page", "get_elements", "screenshot"]
        
        if any(cache in action.lower() for cache in cacheable_actions):
            return {
                "should_cache": True,
                "cache_duration": 300,  # 5 minutes
                "reasoning": "Action results can be cached for performance"
            }
        
        return {
            "should_cache": False,
            "reasoning": "Action results are not suitable for caching"
        }
    
    async def _optimize_resource_allocation(self, trend: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize resource allocation"""
        
        resource_snapshot = await self.performance_agent.resource_monitor.get_snapshot()
        
        if resource_snapshot["cpu_percent"] > 80:
            return {
                "reduce_concurrency": True,
                "suggested_threads": 1,
                "reasoning": "High CPU usage detected, reducing concurrency"
            }
        elif resource_snapshot["memory_percent"] > 85:
            return {
                "enable_memory_optimization": True,
                "clear_cache": True,
                "reasoning": "High memory usage detected, enabling optimizations"
            }
        
        return {
            "status": "optimal",
            "reasoning": "Resource usage is within acceptable limits"
        }
    
    def _analyze_performance_trend(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Analyze performance trends from recent metrics"""
        
        if not metrics:
            return {"avg_response_time": 2.0, "trend": "stable"}
        
        response_times = [m.value for m in metrics if m.metric_type == MetricType.RESPONSE_TIME]
        
        if not response_times:
            return {"avg_response_time": 2.0, "trend": "stable"}
        
        avg_response_time = statistics.mean(response_times)
        
        # Simple trend analysis
        if len(response_times) >= 3:
            recent_avg = statistics.mean(response_times[-3:])
            older_avg = statistics.mean(response_times[:-3]) if len(response_times) > 3 else recent_avg
            
            if recent_avg > older_avg * 1.2:
                trend = "degrading"
            elif recent_avg < older_avg * 0.8:
                trend = "improving"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            "avg_response_time": avg_response_time,
            "trend": trend,
            "sample_size": len(response_times)
        }
    
    def _calculate_optimization_confidence(self, optimizations: Dict[str, Any]) -> float:
        """Calculate confidence score for optimizations"""
        
        confidence_scores = []
        
        for opt_type, opt_data in optimizations.items():
            if isinstance(opt_data, dict) and "confidence" in opt_data:
                confidence_scores.append(opt_data["confidence"])
            else:
                confidence_scores.append(0.5)  # Default confidence
        
        return statistics.mean(confidence_scores) if confidence_scores else 0.5
    
    def _estimate_performance_improvement(self, optimizations: Dict[str, Any]) -> Dict[str, float]:
        """Estimate performance improvement from optimizations"""
        
        estimated_improvements = {
            "time_savings_percent": 0.0,
            "cost_savings_percent": 0.0,
            "reliability_improvement": 0.0
        }
        
        # Simple estimation based on optimization types
        for opt_type, opt_data in optimizations.items():
            if opt_type == "model_selection" and isinstance(opt_data, dict):
                if "flash" in opt_data.get("recommended_model", ""):
                    estimated_improvements["time_savings_percent"] += 15
                    estimated_improvements["cost_savings_percent"] += 25
            
            elif opt_type == "parallel_execution" and isinstance(opt_data, dict):
                if opt_data.get("can_parallelize"):
                    estimated_improvements["time_savings_percent"] += 30
            
            elif opt_type == "caching_strategy" and isinstance(opt_data, dict):
                if opt_data.get("should_cache"):
                    estimated_improvements["time_savings_percent"] += 20
                    estimated_improvements["cost_savings_percent"] += 15
        
        return estimated_improvements


class PerformanceDashboard:
    """Real-time performance monitoring dashboard"""
    
    def __init__(self, performance_agent: 'PerformanceAgent'):
        self.performance_agent = performance_agent
        self.dashboard_data = {}
    
    async def generate_real_time_dashboard(self) -> Dict[str, Any]:
        """Generate real-time dashboard data"""
        
        current_time = datetime.now()
        dashboard = {
            "timestamp": current_time,
            "system_health": await self._get_system_health(),
            "active_sessions": await self._get_active_sessions_status(),
            "performance_metrics": await self._get_current_performance_metrics(),
            "cost_monitoring": await self._get_cost_monitoring_data(),
            "optimization_suggestions": await self._get_current_optimizations(),
            "alerts": await self._get_active_alerts()
        }
        
        return dashboard
    
    async def _get_system_health(self) -> Dict[str, Any]:
        """Get overall system health indicators"""
        resource_snapshot = await self.performance_agent.resource_monitor.get_snapshot()
        
        health_score = 100
        alerts = []
        
        # Check CPU usage
        if resource_snapshot["cpu_percent"] > 80:
            health_score -= 20
            alerts.append("High CPU usage detected")
        
        # Check memory usage
        if resource_snapshot["memory_percent"] > 85:
            health_score -= 25
            alerts.append("High memory usage detected")
        
        return {
            "health_score": max(health_score, 0),
            "status": "healthy" if health_score > 70 else "warning" if health_score > 40 else "critical",
            "alerts": alerts,
            "resource_usage": {
                "cpu": resource_snapshot["cpu_percent"],
                "memory": resource_snapshot["memory_percent"]
            }
        }
    
    async def _get_active_sessions_status(self) -> Dict[str, Any]:
        """Get status of active sessions"""
        # This would integrate with your session management system
        return {
            "active_count": 0,
            "total_today": 0,
            "avg_duration": 0.0
        }
    
    async def _get_current_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics summary"""
        recent_metrics = self.performance_agent.get_recent_metrics(minutes=10)
        
        if not recent_metrics:
            return {"status": "no_data"}
        
        return {
            "avg_execution_time": statistics.mean(m.value for m in recent_metrics),
            "total_operations": len(recent_metrics),
            "success_rate": self._calculate_recent_success_rate(recent_metrics),
            "cost_per_operation": await self._calculate_cost_per_operation(recent_metrics),
            "efficiency_trend": self._calculate_efficiency_trend(recent_metrics)
        }
    
    async def _get_cost_monitoring_data(self) -> Dict[str, Any]:
        """Get cost monitoring data"""
        recent_metrics = self.performance_agent.get_recent_metrics(minutes=60)
        cost_analysis = await self.performance_agent.cost_tracker.analyze_cost_patterns(recent_metrics)
        
        return {
            "hourly_cost": cost_analysis.get("total_cost", 0),
            "cost_efficiency": cost_analysis.get("cost_efficiency", 0),
            "model_distribution": {
                "pro_calls": cost_analysis.get("gemini_pro_calls", 0),
                "flash_calls": cost_analysis.get("gemini_flash_calls", 0)
            }
        }
    
    async def _get_current_optimizations(self) -> List[OptimizationSuggestion]:
        """Get current optimization suggestions"""
        recent_metrics = self.performance_agent.get_recent_metrics(minutes=30)
        return await self.performance_agent.generate_optimization_suggestions(recent_metrics)
    
    async def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active system alerts"""
        alerts = []
        
        # Check for performance issues
        recent_metrics = self.performance_agent.get_recent_metrics(minutes=10)
        if recent_metrics:
            avg_time = statistics.mean(m.value for m in recent_metrics)
            if avg_time > 10.0:  # 10 seconds threshold
                alerts.append({
                    "type": "performance",
                    "severity": "warning",
                    "message": f"Average execution time is {avg_time:.2f}s",
                    "timestamp": datetime.now()
                })
        
        # Check resource usage
        resource_snapshot = await self.performance_agent.resource_monitor.get_snapshot()
        if resource_snapshot["cpu_percent"] > 90:
            alerts.append({
                "type": "resource",
                "severity": "critical",
                "message": f"CPU usage at {resource_snapshot['cpu_percent']:.1f}%",
                "timestamp": datetime.now()
            })
        
        return alerts
    
    def _calculate_recent_success_rate(self, metrics: List[PerformanceMetric]) -> float:
        """Calculate recent success rate"""
        success_metrics = [m for m in metrics if m.metric_type == MetricType.SUCCESS_RATE]
        if not success_metrics:
            return 1.0  # Assume 100% if no data
        
        return statistics.mean(m.value for m in success_metrics)
    
    async def _calculate_cost_per_operation(self, metrics: List[PerformanceMetric]) -> float:
        """Calculate cost per operation"""
        if not metrics:
            return 0.0
        
        # This would integrate with actual cost tracking
        return 0.01  # Placeholder
    
    def _calculate_efficiency_trend(self, metrics: List[PerformanceMetric]) -> str:
        """Calculate efficiency trend"""
        if len(metrics) < 6:
            return "stable"
        
        # Compare first half vs second half
        mid_point = len(metrics) // 2
        first_half_avg = statistics.mean(m.value for m in metrics[:mid_point])
        second_half_avg = statistics.mean(m.value for m in metrics[mid_point:])
        
        if second_half_avg < first_half_avg * 0.9:
            return "improving"
        elif second_half_avg > first_half_avg * 1.1:
            return "degrading"
        else:
            return "stable"


class PerformanceAgent:
    """Meta-agent for comprehensive performance monitoring and optimization"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetric] = []
        self.cost_tracker = CostTracker()
        self.resource_monitor = ResourceMonitor()
        self.optimizer = RealTimeOptimizer(self)
        self.dashboard = PerformanceDashboard(self)
        self.optimization_rules = self._initialize_optimization_rules()
    
    def _initialize_optimization_rules(self) -> Dict[str, Any]:
        """Initialize optimization rules and thresholds"""
        return {
            "slow_execution_threshold": 5.0,  # seconds
            "high_cost_threshold": 0.10,  # dollars
            "repeated_action_threshold": 3,  # count
            "resource_usage_threshold": 80,  # percent
            "error_rate_threshold": 0.05  # 5%
        }
    
    async def monitor_agent_execution(self, agent_id: str, action: str, 
                                    execution_context: Dict[str, Any]) -> PerformanceMetric:
        """Monitor individual agent execution"""
        
        start_time = datetime.now()
        
        # Take resource snapshot before execution
        pre_execution_resources = await self.resource_monitor.get_snapshot()
        
        try:
            # This would be called around actual agent execution
            # For now, we'll simulate execution time
            await asyncio.sleep(0.1)  # Simulate execution
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Take resource snapshot after execution
            post_execution_resources = await self.resource_monitor.get_snapshot()
            
            # Create performance metric
            metric = PerformanceMetric(
                metric_id=f"{agent_id}_{action}_{start_time.timestamp()}",
                metric_type=MetricType.EXECUTION_TIME,
                value=execution_time,
                timestamp=start_time,
                agent_id=agent_id,
                session_id=execution_context.get("session_id", "unknown"),
                action_context={"action": action, **execution_context},
                metadata={
                    "pre_resources": pre_execution_resources,
                    "post_resources": post_execution_resources
                }
            )
            
            self.metrics.append(metric)
            
            # Check for immediate optimization opportunities
            await self._check_immediate_optimizations(metric)
            
            return metric
            
        except Exception as e:
            logger.error(f"Error monitoring agent execution: {e}")
            # Create error metric
            error_metric = PerformanceMetric(
                metric_id=f"{agent_id}_{action}_error_{start_time.timestamp()}",
                metric_type=MetricType.ERROR_RATE,
                value=1.0,  # Error occurred
                timestamp=start_time,
                agent_id=agent_id,
                session_id=execution_context.get("session_id", "unknown"),
                action_context={"action": action, "error": str(e)}
            )
            
            self.metrics.append(error_metric)
            return error_metric
    
    async def _check_immediate_optimizations(self, metric: PerformanceMetric):
        """Check for immediate optimization opportunities"""
        
        # Check for slow execution
        if metric.value > self.optimization_rules["slow_execution_threshold"]:
            logger.warning(f"Slow execution detected: {metric.value:.2f}s for {metric.action_context.get('action')}")
            
            # Generate immediate optimization suggestion
            suggestion = OptimizationSuggestion(
                suggestion_type="immediate_optimization",
                description=f"Slow execution detected for {metric.action_context.get('action')}",
                impact_score=0.8,
                implementation_complexity="medium",
                estimated_savings={"time": metric.value * 0.3},
                applicable_context=[metric.agent_id],
                priority="high"
            )
            
            # This could trigger immediate action or be queued for review
            logger.info(f"Generated optimization suggestion: {suggestion.description}")
    
    async def analyze_session_performance(self, session_id: str) -> Dict[str, Any]:
        """Comprehensive session performance analysis"""
        
        session_metrics = [m for m in self.metrics if m.session_id == session_id]
        
        if not session_metrics:
            return {"error": "No metrics found for session"}
        
        analysis = {
            "session_id": session_id,
            "total_operations": len(session_metrics),
            "total_execution_time": sum(m.value for m in session_metrics),
            "avg_execution_time": statistics.mean(m.value for m in session_metrics),
            "agent_breakdown": self._analyze_agent_performance(session_metrics),
            "cost_analysis": await self.cost_tracker.analyze_session_costs(session_id),
            "resource_impact": await self._analyze_resource_impact(session_metrics),
            "optimization_opportunities": await self.generate_optimization_suggestions(session_metrics)
        }
        
        return analysis
    
    def _analyze_agent_performance(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Analyze performance by agent"""
        
        agent_stats = {}
        
        for metric in metrics:
            agent_id = metric.agent_id
            if agent_id not in agent_stats:
                agent_stats[agent_id] = {
                    "operations": 0,
                    "total_time": 0,
                    "errors": 0
                }
            
            agent_stats[agent_id]["operations"] += 1
            if metric.metric_type == MetricType.EXECUTION_TIME:
                agent_stats[agent_id]["total_time"] += metric.value
            
            if metric.metric_type == MetricType.ERROR_RATE:
                agent_stats[agent_id]["errors"] += 1
        
        # Calculate averages and efficiency scores
        for agent_id, stats in agent_stats.items():
            stats["avg_time"] = stats["total_time"] / stats["operations"] if stats["operations"] > 0 else 0
            stats["error_rate"] = stats["errors"] / stats["operations"] if stats["operations"] > 0 else 0
            stats["efficiency_score"] = max(0, 100 - (stats["avg_time"] * 10) - (stats["error_rate"] * 100))
            stats["success_rate"] = self._calculate_agent_success_rate([m for m in metrics if m.agent_id == agent_id])
        
        return agent_stats
    
    def _calculate_agent_success_rate(self, metrics: List[PerformanceMetric]) -> float:
         """Calculate success rate for specific agent"""
         if not metrics:
             return 0.0
         
         success_count = sum(1 for m in metrics if m.action_context.get("success", True))
         return success_count / len(metrics)
     
     def _analyze_agent_performance(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
         """Analyze performance breakdown by agent"""
         agent_metrics = {}
         
         for metric in metrics:
             agent_id = metric.agent_id
             if agent_id not in agent_metrics:
                 agent_metrics[agent_id] = []
             agent_metrics[agent_id].append(metric)
         
         breakdown = {}
         for agent_id, agent_metric_list in agent_metrics.items():
             execution_metrics = [m for m in agent_metric_list if m.metric_type == MetricType.EXECUTION_TIME]
             
             breakdown[agent_id] = {
                 "total_time": sum(m.value for m in execution_metrics),
                 "avg_time": statistics.mean(m.value for m in execution_metrics) if execution_metrics else 0,
                 "operation_count": len(execution_metrics),
                 "success_rate": self._calculate_agent_success_rate(agent_metric_list)
             }
         
         return breakdown
    
    async def _analyze_resource_impact(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Analyze resource impact of session"""
        
        resource_deltas = []
        
        for metric in metrics:
            if "pre_resources" in metric.metadata and "post_resources" in metric.metadata:
                delta = self.resource_monitor._calculate_resource_delta(
                    metric.metadata["pre_resources"],
                    metric.metadata["post_resources"]
                )
                resource_deltas.append(delta)
        
        if not resource_deltas:
            return {"status": "no_resource_data"}
        
        return {
            "avg_cpu_impact": statistics.mean(d.get("cpu_percent_delta", 0) for d in resource_deltas),
            "avg_memory_impact": statistics.mean(d.get("memory_percent_delta", 0) for d in resource_deltas),
            "total_resource_events": len(resource_deltas)
        }
    
    async def generate_optimization_suggestions(self, metrics: List[PerformanceMetric]) -> List[OptimizationSuggestion]:
        """Generate comprehensive optimization suggestions"""
        
        suggestions = []
        
        # Analyze slow execution patterns
        suggestions.extend(await self._optimize_slow_execution(metrics))
        
        # Analyze cost optimization opportunities
        suggestions.extend(await self._optimize_costs(metrics))
        
        # Analyze repeated action patterns
        suggestions.extend(await self._optimize_repeated_actions(metrics))
        
        # Sort by impact score and priority
        suggestions.sort(key=lambda x: (x.priority == "critical", x.impact_score), reverse=True)
        
        return suggestions
    
    async def _optimize_slow_execution(self, metrics: List[PerformanceMetric]) -> List[OptimizationSuggestion]:
        """Identify and optimize slow execution patterns"""
        
        suggestions = []
        slow_threshold = self.optimization_rules["slow_execution_threshold"]
        
        slow_metrics = [m for m in metrics if m.value > slow_threshold]
        
        if slow_metrics:
            avg_slow_time = statistics.mean(m.value for m in slow_metrics)
            
            suggestions.append(OptimizationSuggestion(
                suggestion_type="execution_optimization",
                description=f"Found {len(slow_metrics)} slow operations (avg: {avg_slow_time:.2f}s)",
                impact_score=0.8,
                implementation_complexity="medium",
                estimated_savings={"time": len(slow_metrics) * (avg_slow_time - slow_threshold)},
                applicable_context=list(set(m.agent_id for m in slow_metrics)),
                priority="high",
                implementation_steps=[
                    "Analyze slow operations for common patterns",
                    "Implement caching for repeated operations",
                    "Consider model optimization for complex tasks",
                    "Add parallel execution where possible"
                ]
            ))
        
        return suggestions
    
    async def _optimize_costs(self, metrics: List[PerformanceMetric]) -> List[OptimizationSuggestion]:
        """Identify cost optimization opportunities"""
        
        suggestions = []
        cost_analysis = await self.cost_tracker.analyze_cost_patterns(metrics)
        
        # Check if too many expensive model calls
        total_calls = cost_analysis["gemini_pro_calls"] + cost_analysis["gemini_flash_calls"]
        if total_calls > 0:
            pro_ratio = cost_analysis["gemini_pro_calls"] / total_calls
            
            if pro_ratio > 0.7:  # More than 70% expensive calls
                suggestions.append(OptimizationSuggestion(
                    suggestion_type="cost_optimization",
                    description=f"High usage of expensive models ({pro_ratio:.1%} Pro calls)",
                    impact_score=0.9,
                    implementation_complexity="low",
                    estimated_savings={"cost": cost_analysis["total_cost"] * 0.3},
                    applicable_context=["orchestrator", "web_agent"],
                    priority="high",
                    implementation_steps=[
                        "Review action complexity requirements",
                        "Use Flash models for simple operations",
                        "Implement intelligent model selection"
                    ]
                ))
        
        return suggestions
    
    async def _optimize_repeated_actions(self, metrics: List[PerformanceMetric]) -> List[OptimizationSuggestion]:
        """Identify and optimize repeated action patterns"""
        
        suggestions = []
        action_patterns = {}
        
        # Track action repetition
        for metric in metrics:
            action = metric.action_context.get("action", "unknown")
            if action not in action_patterns:
                action_patterns[action] = 0
            action_patterns[action] += 1
        
        # Find highly repeated actions
        for action, count in action_patterns.items():
            if count > self.optimization_rules["repeated_action_threshold"]:
                suggestions.append(OptimizationSuggestion(
                    suggestion_type="action_consolidation",
                    description=f"Action '{action}' repeated {count} times. Consider batching or caching.",
                    impact_score=0.7,
                    implementation_complexity="medium",
                    estimated_savings={"time": count * 0.5, "cost": count * 0.01},
                    applicable_context=["web_agent"],
                    priority="medium",
                    implementation_steps=[
                        "Implement result caching for repeated actions",
                        "Consider batching similar operations",
                        "Add intelligent deduplication"
                    ]
                ))
        
        return suggestions
    
    def get_recent_metrics(self, minutes: int = 5) -> List[PerformanceMetric]:
        """Get metrics from the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.metrics if m.timestamp > cutoff_time]
    
    async def analyze_metric_optimization(self, metric: PerformanceMetric) -> List[OptimizationSuggestion]:
        """Analyze metric for real-time optimization opportunities"""
        suggestions = []
        
        # Check for slow execution
        if metric.value > 5.0:  # 5 seconds threshold
            suggestions.append(OptimizationSuggestion(
                suggestion_type="reduce_latency",
                description=f"Agent {metric.agent_id} execution time {metric.value:.2f}s exceeds threshold",
                impact_score=0.8,
                implementation_complexity="medium",
                estimated_savings={"time": metric.value * 0.3},
                applicable_context=[metric.agent_id]
            ))
        
        # Check for resource spikes
        resource_delta = metric.action_context.get("resource_delta", {})
        if resource_delta.get("cpu_percent_delta", 0) > 20:
            suggestions.append(OptimizationSuggestion(
                suggestion_type="reduce_cpu_usage",
                description="High CPU usage spike detected",
                impact_score=0.7,
                implementation_complexity="low",
                estimated_savings={"resource": 0.2},
                applicable_context=[metric.agent_id]
            ))
        
        return suggestions
    
    async def _generate_optimization_suggestions(self, metrics: List[PerformanceMetric]) -> List[OptimizationSuggestion]:
        """Generate comprehensive optimization suggestions"""
        suggestions = []
        
        # Generate suggestions based on metrics analysis
        suggestions.extend(await self._optimize_slow_execution(metrics))
        suggestions.extend(await self._optimize_costs(metrics))
        suggestions.extend(await self._optimize_repeated_actions(metrics))
        
        # Sort by impact score
        suggestions.sort(key=lambda x: x.impact_score, reverse=True)
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def _calculate_efficiency_score(self, metrics: List[PerformanceMetric]) -> float:
        """Calculate overall efficiency score (0-1)"""
        if not metrics:
            return 0.0
        
        # Base score on execution time and success rate
        execution_metrics = [m for m in metrics if m.metric_type == MetricType.EXECUTION_TIME]
        if not execution_metrics:
            return 0.0
            
        avg_time = statistics.mean(m.value for m in execution_metrics)
        success_count = sum(1 for m in metrics if m.action_context.get("success", True))
        success_rate = success_count / len(metrics)
        
        # Normalize time score (lower is better)
        time_score = max(0, 1 - (avg_time / 10))  # 10s = 0 score
        
        # Combine scores
        efficiency = (time_score * 0.6) + (success_rate * 0.4)
        
        return min(1.0, max(0.0, efficiency))
    
    def _analyze_success_patterns(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Analyze patterns in successful vs failed operations"""
        successful = [m for m in metrics if m.action_context.get("success", True)]
        failed = [m for m in metrics if not m.action_context.get("success", True)]
        
        return {
            "total_operations": len(metrics),
            "successful_operations": len(successful),
            "failed_operations": len(failed),
            "success_rate": len(successful) / len(metrics) if metrics else 0,
            "avg_success_time": statistics.mean(m.value for m in successful) if successful else 0,
            "avg_failure_time": statistics.mean(m.value for m in failed) if failed else 0,
            "common_failure_patterns": self._identify_failure_patterns(failed)
        }
    
    def _identify_failure_patterns(self, failed_metrics: List[PerformanceMetric]) -> List[str]:
        """Identify common patterns in failures"""
        patterns = []
        
        if not failed_metrics:
            return patterns
        
        # Check for timeout patterns
        timeout_failures = [m for m in failed_metrics if "timeout" in str(m.action_context.get("error", "")).lower()]
        if len(timeout_failures) > len(failed_metrics) * 0.3:
            patterns.append("frequent_timeouts")
        
        # Check for specific agent failures
        agent_failures = {}
        for metric in failed_metrics:
            agent = metric.agent_id
            agent_failures[agent] = agent_failures.get(agent, 0) + 1
        
        for agent, count in agent_failures.items():
            if count > 2:
                patterns.append(f"repeated_failures_{agent}")
        
        return patterns
    
    async def _track_performance_failure(self, agent_id: str, action: str, error: str):
        """Track performance failures for analysis"""
        failure_metric = PerformanceMetric(
            metric_id=f"{agent_id}_{action}_failure_{datetime.now().timestamp()}",
            metric_type=MetricType.SUCCESS_RATE,
            value=0.0,  # Failure
            timestamp=datetime.now(),
            agent_id=agent_id,
            session_id="failure_tracking",
            action_context={
                "action": action,
                "error": error,
                "success": False
            }
        )
        
        self.metrics.append(failure_metric)
        logger.warning(f"Performance failure tracked: {agent_id} - {action} - {error}")
    
    def _calculate_total_time(self, metrics: List[PerformanceMetric]) -> float:
        """Calculate total execution time from metrics"""
        return sum(m.value for m in metrics if m.metric_type == MetricType.EXECUTION_TIME)
    
    def _identify_bottlenecks(self, metrics: List[PerformanceMetric]) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        # Find slowest operations
        latency_metrics = [m for m in metrics if m.metric_type == MetricType.EXECUTION_TIME]
        if latency_metrics:
            avg_time = statistics.mean(m.value for m in latency_metrics)
            slow_operations = [m for m in latency_metrics if m.value > avg_time * 2]
            
            for op in slow_operations:
                bottlenecks.append({
                    "type": "slow_operation",
                    "agent_id": op.agent_id,
                    "action": op.action_context.get("action", "unknown"),
                    "execution_time": op.value,
                    "severity": "high" if op.value > avg_time * 3 else "medium"
                })
        
        return bottlenecks
     
     async def _optimize_slow_execution(self, metrics: List[PerformanceMetric]) -> List[OptimizationSuggestion]:
         """Generate suggestions for slow execution optimization"""
         suggestions = []
         execution_metrics = [m for m in metrics if m.metric_type == MetricType.EXECUTION_TIME]
         
         if execution_metrics:
             avg_time = statistics.mean(m.value for m in execution_metrics)
             slow_operations = [m for m in execution_metrics if m.value > avg_time * 2]
             
             if slow_operations:
                 suggestions.append(OptimizationSuggestion(
                     type="model_selection",
                     description=f"Detected {len(slow_operations)} slow operations. Consider using faster models.",
                     confidence=0.8,
                     estimated_improvement=0.3
                 ))
         
         return suggestions
     
     async def _optimize_costs(self, metrics: List[PerformanceMetric]) -> List[OptimizationSuggestion]:
         """Generate suggestions for cost optimization"""
         suggestions = []
         cost_metrics = [m for m in metrics if m.metric_type == MetricType.COST]
         
         if cost_metrics:
             total_cost = sum(m.value for m in cost_metrics)
             if total_cost > 1.0:  # High cost threshold
                 suggestions.append(OptimizationSuggestion(
                     type="cost_optimization",
                     description=f"High session cost detected: ${total_cost:.2f}. Consider model optimization.",
                     confidence=0.7,
                     estimated_improvement=0.2
                 ))
         
         return suggestions
     
     async def _optimize_repeated_actions(self, metrics: List[PerformanceMetric]) -> List[OptimizationSuggestion]:
         """Generate suggestions for repeated action optimization"""
         suggestions = []
         action_counts = {}
         
         for metric in metrics:
             action = metric.action_context.get("action", "unknown")
             action_counts[action] = action_counts.get(action, 0) + 1
         
         repeated_actions = {k: v for k, v in action_counts.items() if v > 3}
         
         if repeated_actions:
             suggestions.append(OptimizationSuggestion(
                 type="caching",
                 description=f"Detected repeated actions: {list(repeated_actions.keys())}. Consider caching.",
                 confidence=0.6,
                 estimated_improvement=0.4
             ))
         
         return suggestions
     
     async def analyze_real_time_optimization(self, metric: PerformanceMetric) -> Dict[str, Any]:
        """Analyze real-time optimization opportunities for a single metric"""
        
        current_state = {
            "session_id": metric.session_id,
            "agent_id": metric.agent_id,
            "recent_performance": self.get_recent_metrics(minutes=5)
        }
        
        upcoming_action = metric.action_context.get("action", "unknown")
        
        return await self.optimizer.optimize_next_action(current_state, upcoming_action)
    
    async def get_performance_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {"status": "no_data", "time_window_hours": time_window_hours}
        
        execution_metrics = [m for m in recent_metrics if m.metric_type == MetricType.EXECUTION_TIME]
        
        summary = {
            "time_window_hours": time_window_hours,
            "total_operations": len(recent_metrics),
            "unique_sessions": len(set(m.session_id for m in recent_metrics)),
            "unique_agents": len(set(m.agent_id for m in recent_metrics)),
            "performance_metrics": {
                "avg_execution_time": statistics.mean(m.value for m in execution_metrics) if execution_metrics else 0,
                "median_execution_time": statistics.median(m.value for m in execution_metrics) if execution_metrics else 0,
                "max_execution_time": max(m.value for m in execution_metrics) if execution_metrics else 0,
                "min_execution_time": min(m.value for m in execution_metrics) if execution_metrics else 0
            },
            "agent_performance": self._analyze_agent_performance(recent_metrics),
            "cost_analysis": await self.cost_tracker.analyze_cost_patterns(recent_metrics),
            "resource_trends": await self.resource_monitor.analyze_resource_trends(time_window_hours * 60),
            "optimization_suggestions": await self.generate_optimization_suggestions(recent_metrics),
            "system_health": await self.dashboard._get_system_health(),
            "bottlenecks": self._identify_bottlenecks(recent_metrics),
            "efficiency_score": self._calculate_efficiency_score(recent_metrics)
        }
        
        return summary