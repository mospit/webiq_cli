import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .real_time_adaptation_system import (
    LiveMonitoringSystem, SystemState, AdaptationEvent, 
    AdaptationTrigger, AdaptationSeverity, SystemHealthStatus
)
from .dynamic_strategy_adjuster import DynamicStrategyAdjuster
from .predictive_intervention_engine import (
    PredictiveInterventionEngine, ExecutionTrajectory, InterventionPlan
)
from .intelligent_context_adapter import (
    IntelligentContextAdapter, ExecutionContext, ContextType, 
    SiteCategory, UserPriority
)

logger = logging.getLogger(__name__)

class AdaptationSystemStatus(Enum):
    """Status of the adaptation system"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"

@dataclass
class SystemConfiguration:
    """Configuration for the real-time adaptation system"""
    monitoring_interval: float = 1.0  # seconds
    adaptation_threshold: float = 0.7  # confidence threshold for adaptations
    prediction_horizon: int = 10  # number of steps to predict ahead
    context_analysis_interval: float = 5.0  # seconds
    max_concurrent_adaptations: int = 3
    rollback_timeout: float = 30.0  # seconds
    learning_enabled: bool = True
    proactive_interventions_enabled: bool = True
    context_adaptation_enabled: bool = True

@dataclass
class AdaptationSystemMetrics:
    """Metrics for the adaptation system"""
    total_adaptations_applied: int = 0
    successful_adaptations: int = 0
    failed_adaptations: int = 0
    rollbacks_performed: int = 0
    predictions_made: int = 0
    accurate_predictions: int = 0
    context_changes_detected: int = 0
    proactive_interventions: int = 0
    average_adaptation_time: float = 0.0
    system_uptime: float = 0.0
    last_reset: datetime = field(default_factory=datetime.now)

class RealTimeAdaptationSystem:
    """Unified real-time adaptation system integrating all components"""
    
    def __init__(self, config: Optional[SystemConfiguration] = None):
        self.config = config or SystemConfiguration()
        self.status = AdaptationSystemStatus.INITIALIZING
        self.metrics = AdaptationSystemMetrics()
        
        # Core components
        self.monitoring_system = LiveMonitoringSystem()
        self.strategy_adjuster = DynamicStrategyAdjuster()
        self.predictive_engine = PredictiveInterventionEngine(
            self.monitoring_system, self.strategy_adjuster
        )
        self.context_adapter = IntelligentContextAdapter(
            self.monitoring_system, self.strategy_adjuster, self.predictive_engine
        )
        
        # System state
        self.start_time = datetime.now()
        self.active_adaptations: Dict[str, Dict[str, Any]] = {}
        self.adaptation_queue: List[AdaptationEvent] = []
        
        # Callbacks
        self.adaptation_callbacks: List[Callable] = []
        self.prediction_callbacks: List[Callable] = []
        self.context_callbacks: List[Callable] = []
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._prediction_task: Optional[asyncio.Task] = None
        self._context_analysis_task: Optional[asyncio.Task] = None
        
        # Initialize component integrations
        self._setup_component_integrations()
        
        logger.info("Real-time adaptation system initialized")
    
    def _setup_component_integrations(self):
        """Setup integrations between components"""
        
        # Register monitoring callbacks
        self.monitoring_system.register_adaptation_callback(self._handle_adaptation_event)
        self.monitoring_system.register_context_callback(self._handle_context_change)
        
        # Register strategy adjuster callbacks
        self.strategy_adjuster.register_adaptation_callback(self._handle_adaptation_result)
        
        # Register predictive engine callbacks
        self.predictive_engine.register_prediction_callback(self._handle_prediction_result)
        self.predictive_engine.register_intervention_callback(self._handle_intervention_result)
    
    async def start(self) -> Dict[str, Any]:
        """Start the real-time adaptation system"""
        
        try:
            logger.info("Starting real-time adaptation system...")
            
            # Start core components
            await self.monitoring_system.start_monitoring()
            await self.strategy_adjuster.initialize()
            await self.predictive_engine.initialize()
            
            # Start background tasks
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            if self.config.proactive_interventions_enabled:
                self._prediction_task = asyncio.create_task(self._prediction_loop())
            
            if self.config.context_adaptation_enabled:
                self._context_analysis_task = asyncio.create_task(self._context_analysis_loop())
            
            self.status = AdaptationSystemStatus.ACTIVE
            self.start_time = datetime.now()
            
            logger.info("Real-time adaptation system started successfully")
            
            return {
                "status": "started",
                "components_active": {
                    "monitoring": True,
                    "strategy_adjustment": True,
                    "predictive_intervention": self.config.proactive_interventions_enabled,
                    "context_adaptation": self.config.context_adaptation_enabled
                },
                "start_time": self.start_time
            }
            
        except Exception as e:
            self.status = AdaptationSystemStatus.ERROR
            logger.error(f"Failed to start adaptation system: {e}")
            raise
    
    async def stop(self) -> Dict[str, Any]:
        """Stop the real-time adaptation system"""
        
        logger.info("Stopping real-time adaptation system...")
        
        self.status = AdaptationSystemStatus.SHUTDOWN
        
        # Cancel background tasks
        tasks_to_cancel = []
        if self._monitoring_task:
            tasks_to_cancel.append(self._monitoring_task)
        if self._prediction_task:
            tasks_to_cancel.append(self._prediction_task)
        if self._context_analysis_task:
            tasks_to_cancel.append(self._context_analysis_task)
        
        for task in tasks_to_cancel:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Stop core components
        await self.monitoring_system.stop_monitoring()
        
        # Calculate final metrics
        self.metrics.system_uptime = (datetime.now() - self.start_time).total_seconds()
        
        logger.info("Real-time adaptation system stopped")
        
        return {
            "status": "stopped",
            "final_metrics": self.get_system_metrics(),
            "uptime_seconds": self.metrics.system_uptime
        }
    
    async def pause(self) -> Dict[str, Any]:
        """Pause the adaptation system"""
        
        if self.status != AdaptationSystemStatus.ACTIVE:
            return {"status": "error", "message": "System not active"}
        
        self.status = AdaptationSystemStatus.PAUSED
        logger.info("Real-time adaptation system paused")
        
        return {"status": "paused"}
    
    async def resume(self) -> Dict[str, Any]:
        """Resume the adaptation system"""
        
        if self.status != AdaptationSystemStatus.PAUSED:
            return {"status": "error", "message": "System not paused"}
        
        self.status = AdaptationSystemStatus.ACTIVE
        logger.info("Real-time adaptation system resumed")
        
        return {"status": "resumed"}
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        
        while self.status in [AdaptationSystemStatus.ACTIVE, AdaptationSystemStatus.PAUSED]:
            try:
                if self.status == AdaptationSystemStatus.ACTIVE:
                    # Process adaptation queue
                    await self._process_adaptation_queue()
                    
                    # Update metrics
                    await self._update_system_metrics()
                
                await asyncio.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.config.monitoring_interval)
    
    async def _prediction_loop(self):
        """Predictive intervention loop"""
        
        while self.status in [AdaptationSystemStatus.ACTIVE, AdaptationSystemStatus.PAUSED]:
            try:
                if self.status == AdaptationSystemStatus.ACTIVE:
                    # Analyze execution trajectory
                    current_state = await self.monitoring_system.get_current_state()
                    if current_state:
                        trajectory = await self._build_execution_trajectory(current_state)
                        
                        # Make predictions
                        predictions = await self.predictive_engine.analyze_execution_trajectory(trajectory)
                        
                        # Execute proactive interventions if needed
                        if predictions.get("intervention_needed"):
                            await self._execute_proactive_intervention(predictions)
                
                await asyncio.sleep(self.config.monitoring_interval * 2)  # Less frequent than monitoring
                
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                await asyncio.sleep(self.config.monitoring_interval * 2)
    
    async def _context_analysis_loop(self):
        """Context analysis loop"""
        
        while self.status in [AdaptationSystemStatus.ACTIVE, AdaptationSystemStatus.PAUSED]:
            try:
                if self.status == AdaptationSystemStatus.ACTIVE:
                    # Analyze current context
                    context_data = await self._gather_context_data()
                    if context_data:
                        await self.context_adapter.analyze_context_change(context_data)
                
                await asyncio.sleep(self.config.context_analysis_interval)
                
            except Exception as e:
                logger.error(f"Error in context analysis loop: {e}")
                await asyncio.sleep(self.config.context_analysis_interval)
    
    async def _handle_adaptation_event(self, event: AdaptationEvent):
        """Handle adaptation event from monitoring system"""
        
        if self.status != AdaptationSystemStatus.ACTIVE:
            return
        
        # Check if we can handle more adaptations
        if len(self.active_adaptations) >= self.config.max_concurrent_adaptations:
            self.adaptation_queue.append(event)
            logger.info(f"Queued adaptation event: {event.event_id}")
            return
        
        # Apply adaptation immediately
        await self._apply_adaptation_event(event)
    
    async def _apply_adaptation_event(self, event: AdaptationEvent):
        """Apply adaptation event"""
        
        try:
            # Record start time
            start_time = time.time()
            
            # Apply adaptation through strategy adjuster
            result = await self.strategy_adjuster.apply_adaptation(event)
            
            # Record adaptation
            self.active_adaptations[event.event_id] = {
                "event": event,
                "result": result,
                "start_time": start_time,
                "status": "active"
            }
            
            # Update metrics
            self.metrics.total_adaptations_applied += 1
            if result.get("status") == "success":
                self.metrics.successful_adaptations += 1
            else:
                self.metrics.failed_adaptations += 1
            
            # Calculate adaptation time
            adaptation_time = time.time() - start_time
            self.metrics.average_adaptation_time = (
                (self.metrics.average_adaptation_time * (self.metrics.total_adaptations_applied - 1) + adaptation_time) /
                self.metrics.total_adaptations_applied
            )
            
            # Notify callbacks
            for callback in self.adaptation_callbacks:
                try:
                    await callback(event, result)
                except Exception as e:
                    logger.error(f"Error in adaptation callback: {e}")
            
            logger.info(f"Applied adaptation: {event.event_id} - {result.get('status')}")
            
        except Exception as e:
            logger.error(f"Failed to apply adaptation {event.event_id}: {e}")
            self.metrics.failed_adaptations += 1
    
    async def _handle_adaptation_result(self, adaptation_id: str, result: Dict[str, Any]):
        """Handle adaptation result from strategy adjuster"""
        
        if adaptation_id in self.active_adaptations:
            self.active_adaptations[adaptation_id]["final_result"] = result
            
            # Check if rollback is needed
            if result.get("rollback_needed"):
                await self._handle_rollback(adaptation_id)
            else:
                # Mark as completed
                self.active_adaptations[adaptation_id]["status"] = "completed"
                
                # Learn from outcome if learning is enabled
                if self.config.learning_enabled:
                    await self._learn_from_adaptation_outcome(adaptation_id, result)
    
    async def _handle_rollback(self, adaptation_id: str):
        """Handle adaptation rollback"""
        
        try:
            adaptation_info = self.active_adaptations[adaptation_id]
            event = adaptation_info["event"]
            
            # Perform rollback through strategy adjuster
            rollback_result = await self.strategy_adjuster.rollback_adaptation(adaptation_id)
            
            # Update metrics
            self.metrics.rollbacks_performed += 1
            
            # Mark as rolled back
            self.active_adaptations[adaptation_id]["status"] = "rolled_back"
            self.active_adaptations[adaptation_id]["rollback_result"] = rollback_result
            
            logger.info(f"Rolled back adaptation: {adaptation_id}")
            
        except Exception as e:
            logger.error(f"Failed to rollback adaptation {adaptation_id}: {e}")
    
    async def _handle_context_change(self, context_data: Dict[str, Any]):
        """Handle context change from monitoring system"""
        
        if self.status != AdaptationSystemStatus.ACTIVE:
            return
        
        self.metrics.context_changes_detected += 1
        
        # Notify context callbacks
        for callback in self.context_callbacks:
            try:
                await callback(context_data)
            except Exception as e:
                logger.error(f"Error in context callback: {e}")
    
    async def _handle_prediction_result(self, prediction: Dict[str, Any]):
        """Handle prediction result from predictive engine"""
        
        self.metrics.predictions_made += 1
        
        # Check prediction accuracy (if we have actual outcome)
        if prediction.get("actual_outcome"):
            predicted_outcome = prediction.get("predicted_outcome")
            actual_outcome = prediction["actual_outcome"]
            
            # Simple accuracy check (can be made more sophisticated)
            if abs(predicted_outcome - actual_outcome) < 0.1:
                self.metrics.accurate_predictions += 1
        
        # Notify prediction callbacks
        for callback in self.prediction_callbacks:
            try:
                await callback(prediction)
            except Exception as e:
                logger.error(f"Error in prediction callback: {e}")
    
    async def _handle_intervention_result(self, intervention: Dict[str, Any]):
        """Handle intervention result from predictive engine"""
        
        self.metrics.proactive_interventions += 1
        logger.info(f"Proactive intervention executed: {intervention.get('type')}")
    
    async def _process_adaptation_queue(self):
        """Process queued adaptation events"""
        
        while (self.adaptation_queue and 
               len(self.active_adaptations) < self.config.max_concurrent_adaptations):
            
            event = self.adaptation_queue.pop(0)
            await self._apply_adaptation_event(event)
    
    async def _build_execution_trajectory(self, current_state: SystemState) -> ExecutionTrajectory:
        """Build execution trajectory for predictive analysis"""
        
        # Get recent system states from monitoring system
        recent_states = await self.monitoring_system.get_recent_states(self.config.prediction_horizon)
        
        return ExecutionTrajectory(
            trajectory_id=f"traj_{int(time.time())}",
            states=recent_states,
            current_context=self.context_adapter.get_current_context(),
            prediction_horizon=self.config.prediction_horizon
        )
    
    async def _execute_proactive_intervention(self, predictions: Dict[str, Any]):
        """Execute proactive intervention based on predictions"""
        
        intervention_plan = predictions.get("intervention_plan")
        if not intervention_plan:
            return
        
        # Execute intervention through predictive engine
        result = await self.predictive_engine.execute_proactive_intervention(intervention_plan)
        
        logger.info(f"Executed proactive intervention: {result.get('status')}")
    
    async def _gather_context_data(self) -> Optional[Dict[str, Any]]:
        """Gather current context data"""
        
        try:
            current_state = await self.monitoring_system.get_current_state()
            if not current_state:
                return None
            
            return {
                "context_id": f"ctx_{int(time.time())}",
                "system_state": current_state,
                "active_adaptations": len(self.active_adaptations),
                "recent_performance": await self._get_recent_performance_data(),
                "environmental_factors": await self._get_environmental_factors()
            }
            
        except Exception as e:
            logger.error(f"Failed to gather context data: {e}")
            return None
    
    async def _get_recent_performance_data(self) -> Dict[str, Any]:
        """Get recent performance data"""
        
        # This would analyze recent execution performance
        # For now, return mock data
        return {
            "average_response_time": 2.5,
            "success_rate": 0.92,
            "error_rate": 0.08,
            "resource_utilization": 0.65
        }
    
    async def _get_environmental_factors(self) -> Dict[str, Any]:
        """Get environmental factors"""
        
        # This would gather environmental information
        # For now, return mock data
        return {
            "system_load": 0.7,
            "network_latency": 50,
            "available_memory": 0.8,
            "time_of_day": datetime.now().hour
        }
    
    async def _learn_from_adaptation_outcome(self, adaptation_id: str, result: Dict[str, Any]):
        """Learn from adaptation outcome"""
        
        if not self.config.learning_enabled:
            return
        
        try:
            adaptation_info = self.active_adaptations[adaptation_id]
            event = adaptation_info["event"]
            
            # Extract learning data
            learning_data = {
                "adaptation_type": event.trigger.value,
                "context": self.context_adapter.get_current_context(),
                "outcome": result,
                "effectiveness": result.get("effectiveness_score", 0.5)
            }
            
            # Learn through context adapter
            if learning_data["context"]:
                await self.context_adapter.learn_from_execution_outcome(
                    learning_data["context"],
                    adaptation_info["result"].get("strategy_modifications", {}),
                    learning_data["outcome"]
                )
            
            logger.debug(f"Learned from adaptation outcome: {adaptation_id}")
            
        except Exception as e:
            logger.error(f"Failed to learn from adaptation outcome: {e}")
    
    async def _update_system_metrics(self):
        """Update system metrics"""
        
        self.metrics.system_uptime = (datetime.now() - self.start_time).total_seconds()
        
        # Clean up completed adaptations
        completed_adaptations = [
            aid for aid, info in self.active_adaptations.items()
            if info["status"] in ["completed", "rolled_back"]
        ]
        
        for aid in completed_adaptations:
            del self.active_adaptations[aid]
    
    # Public API methods
    
    def register_adaptation_callback(self, callback: Callable):
        """Register callback for adaptation events"""
        self.adaptation_callbacks.append(callback)
    
    def register_prediction_callback(self, callback: Callable):
        """Register callback for prediction events"""
        self.prediction_callbacks.append(callback)
    
    def register_context_callback(self, callback: Callable):
        """Register callback for context change events"""
        self.context_callbacks.append(callback)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "status": self.status.value,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "active_adaptations": len(self.active_adaptations),
            "queued_adaptations": len(self.adaptation_queue),
            "components_status": {
                "monitoring": self.monitoring_system.is_monitoring(),
                "strategy_adjustment": True,  # Always active when system is running
                "predictive_intervention": self._prediction_task is not None and not self._prediction_task.done(),
                "context_adaptation": self._context_analysis_task is not None and not self._context_analysis_task.done()
            }
        }
    
    def get_system_metrics(self) -> AdaptationSystemMetrics:
        """Get system metrics"""
        return self.metrics
    
    def get_active_adaptations(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active adaptations"""
        return self.active_adaptations.copy()
    
    async def force_adaptation(self, adaptation_event: AdaptationEvent) -> Dict[str, Any]:
        """Force apply an adaptation event"""
        
        if self.status != AdaptationSystemStatus.ACTIVE:
            return {"status": "error", "message": "System not active"}
        
        await self._apply_adaptation_event(adaptation_event)
        
        return {"status": "applied", "event_id": adaptation_event.event_id}
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health information"""
        
        current_state = await self.monitoring_system.get_current_state()
        
        return {
            "system_status": self.get_system_status(),
            "metrics": self.get_system_metrics(),
            "current_state": current_state,
            "current_context": self.context_adapter.get_current_context(),
            "strategy_effectiveness": self.context_adapter.get_strategy_effectiveness_scores(),
            "recent_adaptations": list(self.active_adaptations.keys())[-5:],  # Last 5
            "health_score": await self._calculate_health_score()
        }
    
    async def _calculate_health_score(self) -> float:
        """Calculate overall system health score"""
        
        # Base score
        health_score = 1.0
        
        # Reduce score based on failed adaptations
        if self.metrics.total_adaptations_applied > 0:
            failure_rate = self.metrics.failed_adaptations / self.metrics.total_adaptations_applied
            health_score -= failure_rate * 0.3
        
        # Reduce score based on rollbacks
        if self.metrics.total_adaptations_applied > 0:
            rollback_rate = self.metrics.rollbacks_performed / self.metrics.total_adaptations_applied
            health_score -= rollback_rate * 0.2
        
        # Reduce score if too many active adaptations
        if len(self.active_adaptations) > self.config.max_concurrent_adaptations * 0.8:
            health_score -= 0.1
        
        # Reduce score if prediction accuracy is low
        if self.metrics.predictions_made > 0:
            prediction_accuracy = self.metrics.accurate_predictions / self.metrics.predictions_made
            if prediction_accuracy < 0.7:
                health_score -= (0.7 - prediction_accuracy) * 0.2
        
        return max(0.0, min(1.0, health_score))

# Factory function for easy instantiation
def create_real_time_adaptation_system(config: Optional[SystemConfiguration] = None) -> RealTimeAdaptationSystem:
    """Create and return a configured real-time adaptation system"""
    return RealTimeAdaptationSystem(config)

# Utility function for quick setup
async def setup_and_start_adaptation_system(config: Optional[SystemConfiguration] = None) -> RealTimeAdaptationSystem:
    """Setup and start the real-time adaptation system"""
    system = create_real_time_adaptation_system(config)
    await system.start()
    return system