import asyncio
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
import json
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

from .real_time_adaptation_system import (
    AdaptationEvent, SystemState, AdaptationSeverity, AdaptationTrigger,
    LiveMonitoringSystem, InterventionType, InterventionPlan
)
from .dynamic_strategy_adjuster import DynamicStrategyAdjuster

logger = logging.getLogger(__name__)

@dataclass
class ExecutionTrajectory:
    """Represents execution trajectory data"""
    trajectory_id: str
    start_time: datetime
    current_time: datetime
    execution_steps: List[Dict[str, Any]]
    performance_metrics: List[Dict[str, Any]]
    resource_usage: List[Dict[str, Any]]
    error_events: List[Dict[str, Any]]
    success_indicators: List[Dict[str, Any]]
    predicted_completion_time: Optional[datetime] = None
    risk_factors: List[str] = field(default_factory=list)
    confidence_score: float = 0.0

@dataclass
class PredictionResult:
    """Represents prediction result"""
    prediction_type: str
    predicted_value: float
    confidence: float
    time_horizon: timedelta
    contributing_factors: List[str]
    recommended_actions: List[str]
    risk_level: str
    created_at: datetime = field(default_factory=datetime.now)

class PredictiveInterventionEngine:
    """Proactive intervention based on execution trajectory analysis"""
    
    def __init__(self, monitoring_system: LiveMonitoringSystem, 
                 strategy_adjuster: DynamicStrategyAdjuster):
        self.monitoring_system = monitoring_system
        self.strategy_adjuster = strategy_adjuster
        
        # Predictive models
        self.failure_prediction_model = None
        self.bottleneck_prediction_model = None
        self.success_prediction_model = None
        
        # Model scalers
        self.feature_scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
        # Trajectory tracking
        self.active_trajectories: Dict[str, ExecutionTrajectory] = {}
        self.completed_trajectories: List[ExecutionTrajectory] = []
        
        # Intervention strategies
        self.intervention_strategies = self._initialize_intervention_strategies()
        
        # Prediction history
        self.prediction_history: List[PredictionResult] = []
        
        # Model training data
        self.training_data: Dict[str, List[Any]] = {
            "features": [],
            "failure_labels": [],
            "bottleneck_labels": [],
            "success_labels": []
        }
        
        # Initialize models
        self._initialize_predictive_models()
        
        # Register with monitoring system
        self.monitoring_system.register_trajectory_callback(self.analyze_execution_trajectory)
    
    def _initialize_intervention_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize intervention strategies"""
        return {
            "preemptive_timeout_adjustment": {
                "trigger_conditions": ["predicted_timeout", "response_time_trend_up"],
                "intervention_type": InterventionType.STRATEGY_ADJUSTMENT,
                "parameters": {"timeout_multiplier": 1.5},
                "confidence_threshold": 0.7
            },
            "early_retry_enhancement": {
                "trigger_conditions": ["predicted_failure", "error_rate_increase"],
                "intervention_type": InterventionType.STRATEGY_ADJUSTMENT,
                "parameters": {"additional_retries": 2},
                "confidence_threshold": 0.6
            },
            "proactive_model_switch": {
                "trigger_conditions": ["performance_degradation", "quality_concerns"],
                "intervention_type": InterventionType.STRATEGY_ADJUSTMENT,
                "parameters": {"switch_to_model": "gemini-2.5-pro"},
                "confidence_threshold": 0.8
            },
            "resource_optimization": {
                "trigger_conditions": ["resource_bottleneck", "memory_pressure"],
                "intervention_type": InterventionType.RESOURCE_OPTIMIZATION,
                "parameters": {"reduce_parallel_workers": 1},
                "confidence_threshold": 0.65
            },
            "execution_strategy_change": {
                "trigger_conditions": ["complexity_mismatch", "approach_ineffective"],
                "intervention_type": InterventionType.STRATEGY_ADJUSTMENT,
                "parameters": {"strategy": "conservative"},
                "confidence_threshold": 0.75
            }
        }
    
    def _initialize_predictive_models(self):
        """Initialize predictive models"""
        # Initialize with basic models - these would be trained with historical data
        self.failure_prediction_model = RandomForestRegressor(
            n_estimators=100, random_state=42, max_depth=10
        )
        self.bottleneck_prediction_model = RandomForestRegressor(
            n_estimators=100, random_state=42, max_depth=10
        )
        self.success_prediction_model = RandomForestRegressor(
            n_estimators=100, random_state=42, max_depth=10
        )
        
        # Initialize with dummy data for demonstration
        self._initialize_with_dummy_data()
    
    def _initialize_with_dummy_data(self):
        """Initialize models with dummy data for demonstration"""
        # Generate dummy training data
        np.random.seed(42)
        n_samples = 1000
        
        # Features: [response_time, error_rate, resource_usage, complexity_score, step_count]
        features = np.random.rand(n_samples, 5)
        
        # Labels (dummy)
        failure_labels = (features[:, 1] > 0.7).astype(int)  # High error rate -> failure
        bottleneck_labels = (features[:, 2] > 0.8).astype(int)  # High resource usage -> bottleneck
        success_labels = ((features[:, 0] < 0.3) & (features[:, 1] < 0.2)).astype(int)  # Low response time & error rate -> success
        
        # Train models
        self.failure_prediction_model.fit(features, failure_labels)
        self.bottleneck_prediction_model.fit(features, bottleneck_labels)
        self.success_prediction_model.fit(features, success_labels)
        
        # Fit scaler and anomaly detector
        self.feature_scaler.fit(features)
        self.anomaly_detector.fit(features)
        
        logger.info("Predictive models initialized with dummy data")
    
    async def analyze_execution_trajectory(self, trajectory_data: Dict[str, Any]):
        """Analyze execution trajectory and make predictions"""
        trajectory_id = trajectory_data.get("trajectory_id", f"traj_{int(time.time())}")
        
        # Create or update trajectory
        if trajectory_id not in self.active_trajectories:
            trajectory = ExecutionTrajectory(
                trajectory_id=trajectory_id,
                start_time=datetime.now(),
                current_time=datetime.now(),
                execution_steps=[],
                performance_metrics=[],
                resource_usage=[],
                error_events=[],
                success_indicators=[]
            )
            self.active_trajectories[trajectory_id] = trajectory
        else:
            trajectory = self.active_trajectories[trajectory_id]
        
        # Update trajectory with new data
        await self._update_trajectory(trajectory, trajectory_data)
        
        # Make predictions
        predictions = await self._make_trajectory_predictions(trajectory)
        
        # Analyze for intervention opportunities
        intervention_plan = await self._analyze_intervention_opportunities(trajectory, predictions)
        
        if intervention_plan:
            # Execute proactive intervention
            result = await self.execute_proactive_intervention(intervention_plan)
            logger.info(f"Executed proactive intervention: {result['status']}")
        
        return {
            "trajectory_id": trajectory_id,
            "predictions": predictions,
            "intervention_plan": intervention_plan
        }
    
    async def _update_trajectory(self, trajectory: ExecutionTrajectory, data: Dict[str, Any]):
        """Update trajectory with new data"""
        trajectory.current_time = datetime.now()
        
        # Update execution steps
        if "execution_step" in data:
            trajectory.execution_steps.append({
                "timestamp": datetime.now(),
                "step_data": data["execution_step"]
            })
        
        # Update performance metrics
        if "performance_metrics" in data:
            trajectory.performance_metrics.append({
                "timestamp": datetime.now(),
                "metrics": data["performance_metrics"]
            })
        
        # Update resource usage
        if "resource_usage" in data:
            trajectory.resource_usage.append({
                "timestamp": datetime.now(),
                "usage": data["resource_usage"]
            })
        
        # Update error events
        if "error_event" in data:
            trajectory.error_events.append({
                "timestamp": datetime.now(),
                "error": data["error_event"]
            })
        
        # Update success indicators
        if "success_indicator" in data:
            trajectory.success_indicators.append({
                "timestamp": datetime.now(),
                "indicator": data["success_indicator"]
            })
    
    async def _make_trajectory_predictions(self, trajectory: ExecutionTrajectory) -> List[PredictionResult]:
        """Make predictions based on trajectory data"""
        predictions = []
        
        # Extract features from trajectory
        features = await self._extract_trajectory_features(trajectory)
        
        if features is None:
            return predictions
        
        # Predict failure probability
        failure_prediction = await self._predict_failure_probability(features, trajectory)
        if failure_prediction:
            predictions.append(failure_prediction)
        
        # Predict bottleneck likelihood
        bottleneck_prediction = await self._predict_bottleneck_likelihood(features, trajectory)
        if bottleneck_prediction:
            predictions.append(bottleneck_prediction)
        
        # Predict success evolution
        success_prediction = await self._predict_success_evolution(features, trajectory)
        if success_prediction:
            predictions.append(success_prediction)
        
        # Predict performance degradation
        performance_prediction = await self._predict_performance_degradation(features, trajectory)
        if performance_prediction:
            predictions.append(performance_prediction)
        
        # Predict cost escalation
        cost_prediction = await self._predict_cost_escalation(features, trajectory)
        if cost_prediction:
            predictions.append(cost_prediction)
        
        # Store predictions
        self.prediction_history.extend(predictions)
        
        return predictions
    
    async def _extract_trajectory_features(self, trajectory: ExecutionTrajectory) -> Optional[np.ndarray]:
        """Extract features from trajectory for prediction"""
        if not trajectory.performance_metrics:
            return None
        
        # Calculate aggregate features
        recent_metrics = trajectory.performance_metrics[-5:]  # Last 5 measurements
        
        if not recent_metrics:
            return None
        
        # Extract feature values
        response_times = [m["metrics"].get("response_time", 0) for m in recent_metrics]
        error_rates = [m["metrics"].get("error_rate", 0) for m in recent_metrics]
        resource_usage = [m["metrics"].get("resource_usage", 0) for m in recent_metrics]
        
        # Calculate features
        avg_response_time = np.mean(response_times) if response_times else 0
        avg_error_rate = np.mean(error_rates) if error_rates else 0
        avg_resource_usage = np.mean(resource_usage) if resource_usage else 0
        complexity_score = len(trajectory.execution_steps) / 10.0  # Normalized complexity
        step_count = len(trajectory.execution_steps)
        
        features = np.array([
            avg_response_time,
            avg_error_rate,
            avg_resource_usage,
            complexity_score,
            step_count
        ]).reshape(1, -1)
        
        # Scale features
        try:
            features_scaled = self.feature_scaler.transform(features)
            return features_scaled
        except Exception as e:
            logger.warning(f"Feature scaling failed: {e}")
            return features
    
    async def _predict_failure_probability(self, features: np.ndarray, 
                                         trajectory: ExecutionTrajectory) -> Optional[PredictionResult]:
        """Predict probability of execution failure"""
        try:
            failure_prob = self.failure_prediction_model.predict(features)[0]
            
            if failure_prob > 0.6:  # High failure probability
                contributing_factors = await self._identify_failure_factors(trajectory)
                recommended_actions = await self._recommend_failure_prevention_actions(contributing_factors)
                
                return PredictionResult(
                    prediction_type="failure_probability",
                    predicted_value=failure_prob,
                    confidence=0.8,
                    time_horizon=timedelta(minutes=5),
                    contributing_factors=contributing_factors,
                    recommended_actions=recommended_actions,
                    risk_level="high" if failure_prob > 0.8 else "medium"
                )
        except Exception as e:
            logger.warning(f"Failure prediction failed: {e}")
        
        return None
    
    async def _predict_bottleneck_likelihood(self, features: np.ndarray,
                                           trajectory: ExecutionTrajectory) -> Optional[PredictionResult]:
        """Predict likelihood of performance bottlenecks"""
        try:
            bottleneck_prob = self.bottleneck_prediction_model.predict(features)[0]
            
            if bottleneck_prob > 0.5:  # Moderate bottleneck probability
                contributing_factors = await self._identify_bottleneck_factors(trajectory)
                recommended_actions = await self._recommend_bottleneck_mitigation_actions(contributing_factors)
                
                return PredictionResult(
                    prediction_type="bottleneck_likelihood",
                    predicted_value=bottleneck_prob,
                    confidence=0.75,
                    time_horizon=timedelta(minutes=3),
                    contributing_factors=contributing_factors,
                    recommended_actions=recommended_actions,
                    risk_level="high" if bottleneck_prob > 0.8 else "medium"
                )
        except Exception as e:
            logger.warning(f"Bottleneck prediction failed: {e}")
        
        return None
    
    async def _predict_success_evolution(self, features: np.ndarray,
                                       trajectory: ExecutionTrajectory) -> Optional[PredictionResult]:
        """Predict success probability evolution"""
        try:
            success_prob = self.success_prediction_model.predict(features)[0]
            
            if success_prob < 0.4:  # Low success probability
                contributing_factors = await self._identify_success_inhibitors(trajectory)
                recommended_actions = await self._recommend_success_enhancement_actions(contributing_factors)
                
                return PredictionResult(
                    prediction_type="success_evolution",
                    predicted_value=success_prob,
                    confidence=0.7,
                    time_horizon=timedelta(minutes=10),
                    contributing_factors=contributing_factors,
                    recommended_actions=recommended_actions,
                    risk_level="high" if success_prob < 0.2 else "medium"
                )
        except Exception as e:
            logger.warning(f"Success prediction failed: {e}")
        
        return None
    
    async def _predict_performance_degradation(self, features: np.ndarray,
                                             trajectory: ExecutionTrajectory) -> Optional[PredictionResult]:
        """Predict performance degradation trends"""
        if len(trajectory.performance_metrics) < 3:
            return None
        
        # Analyze performance trends
        recent_metrics = trajectory.performance_metrics[-5:]
        response_times = [m["metrics"].get("response_time", 0) for m in recent_metrics]
        
        if len(response_times) >= 3:
            # Calculate trend
            trend = np.polyfit(range(len(response_times)), response_times, 1)[0]
            
            if trend > 0.1:  # Increasing response time trend
                return PredictionResult(
                    prediction_type="performance_degradation",
                    predicted_value=trend,
                    confidence=0.8,
                    time_horizon=timedelta(minutes=2),
                    contributing_factors=["response_time_increase", "performance_trend"],
                    recommended_actions=["timeout_adjustment", "model_optimization"],
                    risk_level="medium"
                )
        
        return None
    
    async def _predict_cost_escalation(self, features: np.ndarray,
                                     trajectory: ExecutionTrajectory) -> Optional[PredictionResult]:
        """Predict cost escalation based on resource usage trends"""
        if len(trajectory.resource_usage) < 3:
            return None
        
        # Analyze resource usage trends
        recent_usage = trajectory.resource_usage[-5:]
        cpu_usage = [u["usage"].get("cpu_percent", 0) for u in recent_usage]
        
        if len(cpu_usage) >= 3:
            # Calculate trend
            trend = np.polyfit(range(len(cpu_usage)), cpu_usage, 1)[0]
            
            if trend > 5.0:  # Increasing CPU usage trend
                return PredictionResult(
                    prediction_type="cost_escalation",
                    predicted_value=trend,
                    confidence=0.7,
                    time_horizon=timedelta(minutes=5),
                    contributing_factors=["cpu_usage_increase", "resource_trend"],
                    recommended_actions=["resource_optimization", "parallel_reduction"],
                    risk_level="medium"
                )
        
        return None
    
    async def _analyze_intervention_opportunities(self, trajectory: ExecutionTrajectory,
                                                predictions: List[PredictionResult]) -> Optional[InterventionPlan]:
        """Analyze predictions for intervention opportunities"""
        
        if not predictions:
            return None
        
        # Find highest risk predictions
        high_risk_predictions = [p for p in predictions if p.risk_level == "high"]
        medium_risk_predictions = [p for p in predictions if p.risk_level == "medium"]
        
        target_predictions = high_risk_predictions if high_risk_predictions else medium_risk_predictions
        
        if not target_predictions:
            return None
        
        # Select intervention strategy
        intervention_strategy = await self._select_intervention_strategy(target_predictions, trajectory)
        
        if not intervention_strategy:
            return None
        
        # Create intervention plan
        intervention_plan = InterventionPlan(
            plan_id=f"intervention_{int(time.time())}",
            intervention_type=intervention_strategy["intervention_type"],
            target_predictions=target_predictions,
            planned_actions=intervention_strategy["actions"],
            expected_impact=intervention_strategy["expected_impact"],
            confidence_score=intervention_strategy["confidence"],
            execution_priority="high" if high_risk_predictions else "medium",
            estimated_execution_time=timedelta(seconds=30)
        )
        
        return intervention_plan
    
    async def _select_intervention_strategy(self, predictions: List[PredictionResult],
                                          trajectory: ExecutionTrajectory) -> Optional[Dict[str, Any]]:
        """Select appropriate intervention strategy"""
        
        # Analyze prediction types and select strategy
        prediction_types = [p.prediction_type for p in predictions]
        
        if "failure_probability" in prediction_types:
            return {
                "intervention_type": InterventionType.STRATEGY_ADJUSTMENT,
                "actions": ["increase_retry_attempts", "enhance_error_handling"],
                "expected_impact": {"failure_reduction": 0.3},
                "confidence": 0.8
            }
        
        elif "bottleneck_likelihood" in prediction_types:
            return {
                "intervention_type": InterventionType.RESOURCE_OPTIMIZATION,
                "actions": ["optimize_resource_usage", "adjust_parallel_execution"],
                "expected_impact": {"performance_improvement": 0.25},
                "confidence": 0.75
            }
        
        elif "performance_degradation" in prediction_types:
            return {
                "intervention_type": InterventionType.STRATEGY_ADJUSTMENT,
                "actions": ["timeout_adjustment", "model_optimization"],
                "expected_impact": {"response_time_improvement": 0.2},
                "confidence": 0.7
            }
        
        return None
    
    async def execute_proactive_intervention(self, intervention_plan: InterventionPlan) -> Dict[str, Any]:
        """Execute proactive intervention based on plan"""
        
        logger.info(f"Executing proactive intervention: {intervention_plan.plan_id}")
        
        try:
            # Validate intervention plan
            validation_result = await self._validate_intervention_plan(intervention_plan)
            if not validation_result["valid"]:
                return {
                    "status": "rejected",
                    "reason": validation_result["reason"],
                    "plan_id": intervention_plan.plan_id
                }
            
            # Plan intervention execution
            execution_plan = await self._plan_intervention_execution(intervention_plan)
            
            # Execute intervention actions
            execution_results = []
            for action in execution_plan["actions"]:
                result = await self._execute_intervention_action(action, intervention_plan)
                execution_results.append(result)
            
            # Monitor intervention effectiveness
            monitoring_task = asyncio.create_task(
                self._monitor_intervention_effectiveness(intervention_plan, execution_results)
            )
            
            return {
                "status": "executed",
                "plan_id": intervention_plan.plan_id,
                "actions_executed": len(execution_results),
                "monitoring_task": monitoring_task,
                "expected_impact": intervention_plan.expected_impact
            }
            
        except Exception as e:
            logger.error(f"Intervention execution failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "plan_id": intervention_plan.plan_id
            }
    
    async def _validate_intervention_plan(self, plan: InterventionPlan) -> Dict[str, Any]:
        """Validate intervention plan before execution"""
        
        # Check confidence threshold
        if plan.confidence_score < 0.6:
            return {
                "valid": False,
                "reason": f"Confidence score too low: {plan.confidence_score}"
            }
        
        # Check if similar intervention is already active
        active_adaptations = self.strategy_adjuster.get_active_adaptations()
        for adaptation_id, adaptation_info in active_adaptations.items():
            if adaptation_info["adaptation"]["type"] == plan.intervention_type.value:
                return {
                    "valid": False,
                    "reason": f"Similar intervention already active: {adaptation_id}"
                }
        
        return {"valid": True}
    
    async def _plan_intervention_execution(self, plan: InterventionPlan) -> Dict[str, Any]:
        """Plan intervention execution steps"""
        
        execution_actions = []
        
        for action in plan.planned_actions:
            if action == "increase_retry_attempts":
                execution_actions.append({
                    "type": "strategy_modification",
                    "adaptation_type": "retry_enhancement",
                    "action": "increase_retry_attempts",
                    "parameters": {"additional_attempts": 2}
                })
            
            elif action == "timeout_adjustment":
                execution_actions.append({
                    "type": "strategy_modification",
                    "adaptation_type": "timeout_adjustment",
                    "action": "increase_timeouts",
                    "parameters": {"multiplier": 1.5}
                })
            
            elif action == "resource_optimization":
                execution_actions.append({
                    "type": "strategy_modification",
                    "adaptation_type": "resource_optimization",
                    "action": "reduce_cpu_usage",
                    "parameters": {"cpu_limit": 2}
                })
            
            elif action == "model_optimization":
                execution_actions.append({
                    "type": "strategy_modification",
                    "adaptation_type": "model_optimization",
                    "action": "prefer_faster_model",
                    "parameters": {"model": "gemini-2.5-flash"}
                })
        
        return {
            "actions": execution_actions,
            "estimated_duration": plan.estimated_execution_time
        }
    
    async def _execute_intervention_action(self, action: Dict[str, Any],
                                         plan: InterventionPlan) -> Dict[str, Any]:
        """Execute individual intervention action"""
        
        if action["type"] == "strategy_modification":
            # Create adaptation event for strategy adjuster
            adaptation_event = AdaptationEvent(
                event_id=f"intervention_{plan.plan_id}_{int(time.time())}",
                trigger=AdaptationTrigger.PREDICTIVE_INTERVENTION,
                severity=AdaptationSeverity.MEDIUM,
                confidence_score=plan.confidence_score,
                suggested_adaptations=[{
                    "type": action["adaptation_type"],
                    "action": action["action"],
                    "parameters": action["parameters"]
                }],
                context={"intervention_plan_id": plan.plan_id}
            )
            
            # Apply through strategy adjuster
            result = await self.strategy_adjuster.apply_adaptation(adaptation_event)
            return {
                "action": action,
                "result": result,
                "status": "completed"
            }
        
        return {
            "action": action,
            "result": {"status": "unsupported"},
            "status": "skipped"
        }
    
    async def _monitor_intervention_effectiveness(self, plan: InterventionPlan,
                                                execution_results: List[Dict[str, Any]]):
        """Monitor intervention effectiveness over time"""
        
        monitoring_duration = timedelta(minutes=5)
        start_time = datetime.now()
        
        while datetime.now() - start_time < monitoring_duration:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            # Collect current metrics
            current_metrics = await self._collect_current_metrics()
            
            # Analyze intervention impact
            impact_analysis = await self._analyze_intervention_impact(
                plan, execution_results, current_metrics
            )
            
            logger.info(f"Intervention {plan.plan_id} impact: {impact_analysis['effectiveness']}")
            
            # If intervention is not effective, consider rollback
            if impact_analysis["effectiveness"] < 0.2:
                logger.warning(f"Intervention {plan.plan_id} showing poor effectiveness, considering rollback")
                break
    
    async def _collect_current_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        # This would collect actual metrics from the monitoring system
        return {
            "response_time": 2.1,
            "success_rate": 0.95,
            "error_rate": 0.05,
            "resource_usage": 0.6
        }
    
    async def _analyze_intervention_impact(self, plan: InterventionPlan,
                                         execution_results: List[Dict[str, Any]],
                                         current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze intervention impact"""
        # This would perform actual impact analysis
        # For now, return mock analysis
        return {
            "effectiveness": 0.7,
            "metrics_improvement": {
                "response_time": 0.15,
                "success_rate": 0.03
            },
            "recommendation": "continue"
        }
    
    # Helper methods for factor identification and action recommendation
    async def _identify_failure_factors(self, trajectory: ExecutionTrajectory) -> List[str]:
        """Identify factors contributing to failure probability"""
        factors = []
        
        if trajectory.error_events:
            factors.append("high_error_rate")
        
        if len(trajectory.execution_steps) > 20:
            factors.append("complex_execution")
        
        return factors
    
    async def _recommend_failure_prevention_actions(self, factors: List[str]) -> List[str]:
        """Recommend actions to prevent failure"""
        actions = []
        
        if "high_error_rate" in factors:
            actions.extend(["increase_retry_attempts", "enhance_error_handling"])
        
        if "complex_execution" in factors:
            actions.extend(["simplify_strategy", "increase_timeouts"])
        
        return actions
    
    async def _identify_bottleneck_factors(self, trajectory: ExecutionTrajectory) -> List[str]:
        """Identify factors contributing to bottlenecks"""
        factors = []
        
        if trajectory.resource_usage:
            recent_usage = trajectory.resource_usage[-3:]
            avg_cpu = np.mean([u["usage"].get("cpu_percent", 0) for u in recent_usage])
            if avg_cpu > 80:
                factors.append("high_cpu_usage")
        
        return factors
    
    async def _recommend_bottleneck_mitigation_actions(self, factors: List[str]) -> List[str]:
        """Recommend actions to mitigate bottlenecks"""
        actions = []
        
        if "high_cpu_usage" in factors:
            actions.extend(["reduce_parallel_workers", "optimize_processing"])
        
        return actions
    
    async def _identify_success_inhibitors(self, trajectory: ExecutionTrajectory) -> List[str]:
        """Identify factors inhibiting success"""
        factors = []
        
        if not trajectory.success_indicators:
            factors.append("no_success_indicators")
        
        return factors
    
    async def _recommend_success_enhancement_actions(self, factors: List[str]) -> List[str]:
        """Recommend actions to enhance success probability"""
        actions = []
        
        if "no_success_indicators" in factors:
            actions.extend(["strategy_adjustment", "approach_modification"])
        
        return actions
    
    def get_active_trajectories(self) -> Dict[str, ExecutionTrajectory]:
        """Get currently active trajectories"""
        return self.active_trajectories.copy()
    
    def get_prediction_history(self) -> List[PredictionResult]:
        """Get prediction history"""
        return self.prediction_history.copy()