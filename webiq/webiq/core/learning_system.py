import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
from pathlib import Path

from .pattern_recognition_engine import PatternRecognitionEngine
from .adaptive_strategy_engine import AdaptiveStrategyEngine, AdaptiveStrategy
from .predictive_analytics_engine import PredictiveAnalyticsEngine
from .knowledge_base_manager import KnowledgeBaseManager, AutomationPattern, SiteKnowledge, GoalTemplate

logger = logging.getLogger(__name__)

@dataclass
class LearningConfig:
    """Configuration for the learning system"""
    
    # Pattern recognition settings
    min_pattern_confidence: float = 0.6
    pattern_similarity_threshold: float = 0.8
    max_patterns_per_goal: int = 50
    
    # Adaptive strategy settings
    strategy_adaptation_threshold: float = 0.7
    min_strategy_usage: int = 3
    strategy_learning_rate: float = 0.1
    
    # Predictive analytics settings
    min_training_samples: int = 10
    model_retrain_interval: timedelta = timedelta(days=7)
    prediction_cache_ttl: timedelta = timedelta(hours=1)
    
    # Knowledge base settings
    knowledge_cleanup_interval: timedelta = timedelta(days=30)
    pattern_retention_days: int = 90
    min_pattern_usage: int = 2
    
    # Learning system settings
    learning_enabled: bool = True
    auto_optimization: bool = True
    feedback_learning: bool = True
    real_time_adaptation: bool = True

@dataclass
class LearningMetrics:
    """Metrics for learning system performance"""
    
    patterns_learned: int = 0
    strategies_adapted: int = 0
    predictions_made: int = 0
    successful_optimizations: int = 0
    
    learning_accuracy: float = 0.0
    prediction_accuracy: float = 0.0
    optimization_impact: float = 0.0
    
    last_learning_session: Optional[datetime] = None
    last_model_training: Optional[datetime] = None
    last_optimization: Optional[datetime] = None
    
    total_sessions_processed: int = 0
    total_knowledge_items: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "patterns_learned": self.patterns_learned,
            "strategies_adapted": self.strategies_adapted,
            "predictions_made": self.predictions_made,
            "successful_optimizations": self.successful_optimizations,
            "learning_accuracy": self.learning_accuracy,
            "prediction_accuracy": self.prediction_accuracy,
            "optimization_impact": self.optimization_impact,
            "last_learning_session": self.last_learning_session.isoformat() if self.last_learning_session else None,
            "last_model_training": self.last_model_training.isoformat() if self.last_model_training else None,
            "last_optimization": self.last_optimization.isoformat() if self.last_optimization else None,
            "total_sessions_processed": self.total_sessions_processed,
            "total_knowledge_items": self.total_knowledge_items
        }

class LearningSystem:
    """Unified learning and adaptation system for WebIQ automation"""
    
    def __init__(self, config: Optional[LearningConfig] = None, 
                 knowledge_db_path: str = "webiq_knowledge.db",
                 models_dir: str = "webiq_models"):
        
        self.config = config or LearningConfig()
        self.metrics = LearningMetrics()
        
        # Initialize core components
        self.knowledge_manager = KnowledgeBaseManager(knowledge_db_path)
        self.pattern_engine = PatternRecognitionEngine(self.knowledge_manager.knowledge_base)
        self.strategy_engine = AdaptiveStrategyEngine(self.knowledge_manager.knowledge_base)
        self.analytics_engine = PredictiveAnalyticsEngine(models_dir)
        
        # Learning state
        self.learning_active = False
        self.last_cleanup = datetime.now()
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        
        logger.info("Learning system initialized")
    
    async def start(self):
        """Start the learning system and background tasks"""
        if not self.config.learning_enabled:
            logger.info("Learning system disabled by configuration")
            return
        
        self.learning_active = True
        
        # Start background tasks
        if self.config.auto_optimization:
            self._background_tasks.append(
                asyncio.create_task(self._periodic_optimization())
            )
        
        self._background_tasks.append(
            asyncio.create_task(self._periodic_cleanup())
        )
        
        self._background_tasks.append(
            asyncio.create_task(self._periodic_model_training())
        )
        
        logger.info("Learning system started")
    
    async def stop(self):
        """Stop the learning system and cleanup"""
        self.learning_active = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
        
        logger.info("Learning system stopped")
    
    async def learn_from_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from a completed automation session"""
        
        if not self.config.learning_enabled:
            return {"status": "learning_disabled"}
        
        learning_results = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_data.get("session_id", "unknown"),
            "learning_summary": {},
            "insights": [],
            "recommendations": [],
            "errors": []
        }
        
        try:
            # Extract session information
            goal = session_data.get("goal", "")
            url = session_data.get("url", "")
            success = session_data.get("success", False)
            duration = session_data.get("duration", 0.0)
            cost = session_data.get("cost", 0.0)
            action_history = session_data.get("action_history", [])
            errors = session_data.get("errors", [])
            
            # Learn patterns
            pattern_results = await self.pattern_engine.learn_from_session(session_data)
            learning_results["learning_summary"]["patterns"] = pattern_results
            
            # Adapt strategies if feedback learning is enabled
            if self.config.feedback_learning:
                strategy_results = await self.strategy_engine.adapt_strategy_from_feedback(
                    goal, url, success, duration, action_history, errors
                )
                learning_results["learning_summary"]["strategies"] = strategy_results
                self.metrics.strategies_adapted += strategy_results.get("adaptations_made", 0)
            
            # Update predictive models
            if len(action_history) > 0:
                await self.analytics_engine.add_training_data({
                    "goal": goal,
                    "url": url,
                    "success": success,
                    "duration": duration,
                    "cost": cost,
                    "action_count": len(action_history),
                    "complexity": self._calculate_session_complexity(action_history)
                })
            
            # Generate insights and recommendations
            insights = await self._generate_learning_insights(session_data, pattern_results)
            learning_results["insights"] = insights
            
            recommendations = await self._generate_learning_recommendations(session_data)
            learning_results["recommendations"] = recommendations
            
            # Update metrics
            self.metrics.patterns_learned += pattern_results.get("patterns_created", 0)
            self.metrics.total_sessions_processed += 1
            self.metrics.last_learning_session = datetime.now()
            
            # Calculate learning accuracy
            if success:
                self.metrics.learning_accuracy = (
                    (self.metrics.learning_accuracy * (self.metrics.total_sessions_processed - 1) + 1.0) /
                    self.metrics.total_sessions_processed
                )
            else:
                self.metrics.learning_accuracy = (
                    (self.metrics.learning_accuracy * (self.metrics.total_sessions_processed - 1) + 0.0) /
                    self.metrics.total_sessions_processed
                )
            
            logger.info(f"Learning completed for session {session_data.get('session_id', 'unknown')}")
            
        except Exception as e:
            error_msg = f"Learning failed: {str(e)}"
            logger.error(error_msg)
            learning_results["errors"].append(error_msg)
        
        return learning_results
    
    async def get_automation_recommendations(self, goal: str, url: str, 
                                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get comprehensive automation recommendations"""
        
        recommendations = {
            "timestamp": datetime.now().isoformat(),
            "goal": goal,
            "url": url,
            "strategy": {},
            "predictions": {},
            "patterns": [],
            "optimizations": [],
            "confidence_score": 0.0
        }
        
        try:
            # Get optimal strategy
            strategy = await self.strategy_engine.select_optimal_strategy(goal, url, context or {})
            if strategy:
                recommendations["strategy"] = {
                    "name": strategy.name,
                    "parameters": strategy.parameters,
                    "confidence": strategy.confidence,
                    "expected_success_rate": strategy.expected_success_rate,
                    "estimated_duration": strategy.estimated_duration
                }
            
            # Get predictions
            predictions = await self.analytics_engine.predict_automation_success(
                goal, url, context or {}
            )
            recommendations["predictions"] = predictions
            
            # Get relevant patterns
            knowledge_recommendations = await self.knowledge_manager.get_recommendations(
                goal, url, context or {}
            )
            recommendations["patterns"] = knowledge_recommendations.get("patterns", [])
            
            # Get optimization suggestions
            optimizations = await self.pattern_engine.suggest_optimizations(
                goal, url, context or {}
            )
            recommendations["optimizations"] = optimizations
            
            # Calculate overall confidence
            confidence_factors = [
                strategy.confidence if strategy else 0.5,
                predictions.get("confidence", 0.5),
                min(1.0, len(knowledge_recommendations.get("patterns", [])) / 3.0),
                min(1.0, len(optimizations) / 5.0)
            ]
            recommendations["confidence_score"] = sum(confidence_factors) / len(confidence_factors)
            
            self.metrics.predictions_made += 1
            
        except Exception as e:
            logger.error(f"Failed to get automation recommendations: {e}")
            recommendations["error"] = str(e)
        
        return recommendations
    
    async def optimize_automation_strategy(self, goal: str, url: str, 
                                         current_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize automation strategy based on current performance"""
        
        optimization_results = {
            "timestamp": datetime.now().isoformat(),
            "goal": goal,
            "url": url,
            "optimizations_applied": [],
            "expected_improvements": {},
            "confidence": 0.0
        }
        
        try:
            # Get current strategy performance
            current_success_rate = current_performance.get("success_rate", 0.0)
            current_duration = current_performance.get("avg_duration", 0.0)
            current_cost = current_performance.get("avg_cost", 0.0)
            
            # Find optimization opportunities
            optimizations = await self.pattern_engine.suggest_optimizations(
                goal, url, current_performance
            )
            
            applied_optimizations = []
            
            for optimization in optimizations:
                if optimization.get("confidence", 0.0) > self.config.strategy_adaptation_threshold:
                    # Apply optimization
                    optimization_type = optimization.get("type", "unknown")
                    optimization_params = optimization.get("parameters", {})
                    
                    # Update strategy with optimization
                    await self.strategy_engine.apply_optimization(
                        goal, url, optimization_type, optimization_params
                    )
                    
                    applied_optimizations.append({
                        "type": optimization_type,
                        "parameters": optimization_params,
                        "expected_impact": optimization.get("expected_impact", {})
                    })
            
            optimization_results["optimizations_applied"] = applied_optimizations
            
            # Predict improvements
            if applied_optimizations:
                improved_predictions = await self.analytics_engine.predict_automation_success(
                    goal, url, {"optimizations": applied_optimizations}
                )
                
                optimization_results["expected_improvements"] = {
                    "success_rate_improvement": improved_predictions.get("success_probability", 0.0) - current_success_rate,
                    "duration_improvement": current_duration - improved_predictions.get("estimated_duration", current_duration),
                    "cost_improvement": current_cost - improved_predictions.get("estimated_cost", current_cost)
                }
                
                optimization_results["confidence"] = improved_predictions.get("confidence", 0.0)
                
                self.metrics.successful_optimizations += len(applied_optimizations)
                self.metrics.last_optimization = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to optimize automation strategy: {e}")
            optimization_results["error"] = str(e)
        
        return optimization_results
    
    async def get_learning_insights(self, timeframe_days: int = 30) -> Dict[str, Any]:
        """Get insights about learning system performance"""
        
        insights = {
            "timestamp": datetime.now().isoformat(),
            "timeframe_days": timeframe_days,
            "metrics": self.metrics.to_dict(),
            "knowledge_stats": {},
            "performance_trends": {},
            "recommendations": []
        }
        
        try:
            # Get knowledge base statistics
            knowledge_stats = await self.knowledge_manager.knowledge_base.get_knowledge_stats()
            insights["knowledge_stats"] = knowledge_stats
            
            # Get performance trends
            performance_trends = await self._analyze_performance_trends(timeframe_days)
            insights["performance_trends"] = performance_trends
            
            # Generate system recommendations
            system_recommendations = await self._generate_system_recommendations()
            insights["recommendations"] = system_recommendations
            
        except Exception as e:
            logger.error(f"Failed to get learning insights: {e}")
            insights["error"] = str(e)
        
        return insights
    
    async def retrain_models(self, force: bool = False) -> Dict[str, Any]:
        """Retrain predictive models"""
        
        retrain_results = {
            "timestamp": datetime.now().isoformat(),
            "forced": force,
            "models_retrained": [],
            "performance_metrics": {},
            "success": False
        }
        
        try:
            # Check if retraining is needed
            if not force:
                last_training = self.metrics.last_model_training
                if (last_training and 
                    datetime.now() - last_training < self.config.model_retrain_interval):
                    retrain_results["message"] = "Retraining not needed yet"
                    return retrain_results
            
            # Retrain models
            training_results = await self.analytics_engine.train_models()
            
            retrain_results["models_retrained"] = training_results.get("models_trained", [])
            retrain_results["performance_metrics"] = training_results.get("performance", {})
            retrain_results["success"] = training_results.get("success", False)
            
            if retrain_results["success"]:
                self.metrics.last_model_training = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to retrain models: {e}")
            retrain_results["error"] = str(e)
        
        return retrain_results
    
    async def _periodic_optimization(self):
        """Background task for periodic optimization"""
        
        while self.learning_active:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Get recent performance data and optimize
                # This would integrate with the performance monitoring system
                logger.debug("Running periodic optimization")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic optimization failed: {e}")
    
    async def _periodic_cleanup(self):
        """Background task for periodic cleanup"""
        
        while self.learning_active:
            try:
                await asyncio.sleep(86400)  # Run daily
                
                if datetime.now() - self.last_cleanup > self.config.knowledge_cleanup_interval:
                    # Cleanup old patterns
                    cleaned = await self.knowledge_manager.knowledge_base.cleanup_old_patterns(
                        self.config.pattern_retention_days,
                        self.config.min_pattern_usage
                    )
                    
                    logger.info(f"Cleaned up {cleaned} old patterns")
                    self.last_cleanup = datetime.now()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic cleanup failed: {e}")
    
    async def _periodic_model_training(self):
        """Background task for periodic model training"""
        
        while self.learning_active:
            try:
                await asyncio.sleep(86400)  # Check daily
                
                # Check if retraining is needed
                last_training = self.metrics.last_model_training
                if (not last_training or 
                    datetime.now() - last_training > self.config.model_retrain_interval):
                    
                    await self.retrain_models()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic model training failed: {e}")
    
    def _calculate_session_complexity(self, action_history: List[Dict[str, Any]]) -> float:
        """Calculate complexity score for a session"""
        
        if not action_history:
            return 0.0
        
        # Factors that contribute to complexity
        action_count = len(action_history)
        unique_action_types = len(set(action.get("action_type", "") for action in action_history))
        total_wait_time = sum(action.get("wait_time", 0.0) for action in action_history)
        
        # Normalize and combine factors
        complexity = (
            min(1.0, action_count / 20.0) * 0.4 +  # Action count factor
            min(1.0, unique_action_types / 10.0) * 0.3 +  # Diversity factor
            min(1.0, total_wait_time / 30.0) * 0.3  # Timing complexity factor
        )
        
        return complexity * 10.0  # Scale to 0-10
    
    async def _generate_learning_insights(self, session_data: Dict[str, Any], 
                                        pattern_results: Dict[str, Any]) -> List[str]:
        """Generate insights from learning session"""
        
        insights = []
        
        try:
            success = session_data.get("success", False)
            duration = session_data.get("duration", 0.0)
            action_count = len(session_data.get("action_history", []))
            
            # Performance insights
            if success and duration < 30:
                insights.append("Fast successful automation - pattern worth reinforcing")
            elif success and duration > 120:
                insights.append("Slow but successful automation - optimization opportunity")
            elif not success:
                insights.append("Failed automation - failure pattern identified for avoidance")
            
            # Pattern insights
            patterns_created = pattern_results.get("patterns_created", 0)
            if patterns_created > 0:
                insights.append(f"Learned {patterns_created} new automation patterns")
            
            patterns_updated = pattern_results.get("patterns_updated", 0)
            if patterns_updated > 0:
                insights.append(f"Updated {patterns_updated} existing patterns with new data")
            
            # Complexity insights
            if action_count > 15:
                insights.append("Complex automation detected - consider breaking into smaller steps")
            elif action_count < 3:
                insights.append("Simple automation - good candidate for optimization")
            
        except Exception as e:
            logger.error(f"Failed to generate learning insights: {e}")
            insights.append(f"Insight generation error: {str(e)}")
        
        return insights
    
    async def _generate_learning_recommendations(self, session_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on learning"""
        
        recommendations = []
        
        try:
            goal = session_data.get("goal", "")
            url = session_data.get("url", "")
            success = session_data.get("success", False)
            errors = session_data.get("errors", [])
            
            # Get knowledge-based recommendations
            knowledge_recs = await self.knowledge_manager.get_recommendations(goal, url, session_data)
            
            if knowledge_recs.get("patterns"):
                best_pattern = max(knowledge_recs["patterns"], key=lambda p: p["success_rate"])
                recommendations.append(
                    f"Use pattern {best_pattern['pattern_id']} for similar tasks (success rate: {best_pattern['success_rate']:.1%})"
                )
            
            # Site-specific recommendations
            site_insights = knowledge_recs.get("site_insights", {})
            if site_insights.get("reliability_score", 1.0) < 0.7:
                recommendations.append("Site has low reliability - use conservative timeouts and retry logic")
            
            # Error-based recommendations
            if errors:
                for error in errors:
                    if "timeout" in error.lower():
                        recommendations.append("Increase timeout values for this site")
                    elif "element not found" in error.lower():
                        recommendations.append("Update element selectors or add wait conditions")
            
            # Success-based recommendations
            if success:
                recommendations.append("Successful pattern - consider using as template for similar goals")
            
        except Exception as e:
            logger.error(f"Failed to generate learning recommendations: {e}")
            recommendations.append(f"Recommendation generation error: {str(e)}")
        
        return recommendations
    
    async def _analyze_performance_trends(self, timeframe_days: int) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        
        trends = {
            "success_rate_trend": "stable",
            "duration_trend": "stable",
            "learning_efficiency": "good",
            "pattern_quality": "good"
        }
        
        try:
            # This would analyze historical data to identify trends
            # For now, return basic trend analysis
            
            if self.metrics.learning_accuracy > 0.8:
                trends["learning_efficiency"] = "excellent"
            elif self.metrics.learning_accuracy > 0.6:
                trends["learning_efficiency"] = "good"
            else:
                trends["learning_efficiency"] = "needs_improvement"
            
            if self.metrics.patterns_learned > self.metrics.total_sessions_processed * 0.5:
                trends["pattern_quality"] = "excellent"
            elif self.metrics.patterns_learned > self.metrics.total_sessions_processed * 0.2:
                trends["pattern_quality"] = "good"
            else:
                trends["pattern_quality"] = "needs_improvement"
            
        except Exception as e:
            logger.error(f"Failed to analyze performance trends: {e}")
            trends["error"] = str(e)
        
        return trends
    
    async def _generate_system_recommendations(self) -> List[str]:
        """Generate system-level recommendations"""
        
        recommendations = []
        
        try:
            # Learning efficiency recommendations
            if self.metrics.learning_accuracy < 0.6:
                recommendations.append("Learning accuracy is low - consider adjusting pattern recognition thresholds")
            
            if self.metrics.patterns_learned < 10:
                recommendations.append("Few patterns learned - increase automation diversity to improve learning")
            
            # Model training recommendations
            if not self.metrics.last_model_training:
                recommendations.append("Models have not been trained - run initial training")
            elif (datetime.now() - self.metrics.last_model_training > 
                  self.config.model_retrain_interval * 2):
                recommendations.append("Models are outdated - schedule retraining")
            
            # Optimization recommendations
            if self.metrics.successful_optimizations == 0:
                recommendations.append("No optimizations applied - review optimization thresholds")
            
            # Knowledge base recommendations
            if self.metrics.total_knowledge_items > 10000:
                recommendations.append("Large knowledge base - consider cleanup of old patterns")
            
        except Exception as e:
            logger.error(f"Failed to generate system recommendations: {e}")
            recommendations.append(f"System recommendation error: {str(e)}")
        
        return recommendations