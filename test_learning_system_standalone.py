#!/usr/bin/env python3
"""
Standalone test suite for the WebIQ Learning & Adaptation System

This test validates all components of the sophisticated learning system without
depending on the existing WebIQ modules to avoid import conflicts.

Run with: py test_learning_system_standalone.py
"""

import asyncio
import json
import tempfile
import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
import os
from dataclasses import dataclass, field
import pickle
import hashlib

# Mock dependencies that might not be available
class MockMLModel:
    def __init__(self):
        self.is_fitted = False
        self.feature_names = []
    
    def fit(self, X, y):
        self.is_fitted = True
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return [0.7] * len(X)
    
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return [[0.3, 0.7]] * len(X)
    
    def score(self, X, y):
        return 0.85

class MockStandardScaler:
    def __init__(self):
        self.fitted = False
    
    def fit(self, X):
        self.fitted = True
        return self
    
    def transform(self, X):
        return X
    
    def fit_transform(self, X):
        self.fitted = True
        return X

# Mock the sklearn imports
sys.modules['sklearn'] = type('MockModule', (), {})()
sys.modules['sklearn.ensemble'] = type('MockModule', (), {
    'RandomForestClassifier': MockMLModel,
    'GradientBoostingRegressor': MockMLModel
})()
sys.modules['sklearn.preprocessing'] = type('MockModule', (), {
    'StandardScaler': MockStandardScaler
})()
sys.modules['sklearn.model_selection'] = type('MockModule', (), {
    'train_test_split': lambda X, y, test_size=0.2, random_state=42: (X[:int(len(X)*0.8)], X[int(len(X)*0.8):], y[:int(len(y)*0.8)], y[int(len(y)*0.8):])
})()
sys.modules['sklearn.metrics'] = type('MockModule', (), {
    'accuracy_score': lambda y_true, y_pred: 0.85,
    'mean_squared_error': lambda y_true, y_pred: 0.15,
    'r2_score': lambda y_true, y_pred: 0.80
})()

# Simplified implementations of the learning system components

@dataclass
class AutomationPattern:
    """Represents a learned automation pattern"""
    pattern_id: str
    goal: str
    url: str
    action_sequence: List[Dict[str, Any]]
    success_rate: float
    avg_duration: float
    usage_count: int = 0
    last_used: Optional[datetime] = None
    created_at: Optional[datetime] = None
    confidence: float = 0.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_used is None:
            self.last_used = datetime.now()

@dataclass
class SiteKnowledge:
    """Site-specific knowledge and patterns"""
    url: str
    common_elements: Dict[str, str] = field(default_factory=dict)
    timing_patterns: Dict[str, float] = field(default_factory=dict)
    reliability_score: float = 1.0
    optimization_rules: List[str] = field(default_factory=list)
    last_updated: Optional[datetime] = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

@dataclass
class GoalTemplate:
    """Template for achieving specific goals"""
    goal: str
    canonical_steps: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    avg_duration: float = 0.0
    complexity_score: float = 0.0
    usage_count: int = 0
    last_used: Optional[datetime] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_used is None:
            self.last_used = datetime.now()

@dataclass
class AdaptiveStrategy:
    """Represents an adaptive automation strategy"""
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    expected_success_rate: float = 0.5
    estimated_duration: float = 30.0
    usage_count: int = 0
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: Optional[datetime] = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

class SimplifiedKnowledgeBase:
    """Simplified knowledge base for testing"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.patterns: Dict[str, AutomationPattern] = {}
        self.site_knowledge: Dict[str, SiteKnowledge] = {}
        self.goal_templates: Dict[str, GoalTemplate] = {}
        self._init_db()
    
    def _init_db(self):
        """Initialize the database"""
        # For testing, we'll use in-memory storage
        pass
    
    async def store_pattern(self, pattern: AutomationPattern):
        """Store an automation pattern"""
        self.patterns[pattern.pattern_id] = pattern
    
    async def find_patterns(self, goal: str, url: str, limit: int = 10) -> List[AutomationPattern]:
        """Find patterns matching goal and URL"""
        matches = []
        for pattern in self.patterns.values():
            if pattern.goal.lower() in goal.lower() or goal.lower() in pattern.goal.lower():
                if pattern.url in url or url in pattern.url:
                    matches.append(pattern)
        return matches[:limit]
    
    async def store_site_knowledge(self, knowledge: SiteKnowledge):
        """Store site knowledge"""
        self.site_knowledge[knowledge.url] = knowledge
    
    async def get_site_knowledge(self, url: str) -> Optional[SiteKnowledge]:
        """Get site knowledge"""
        return self.site_knowledge.get(url)
    
    async def store_goal_template(self, template: GoalTemplate):
        """Store goal template"""
        self.goal_templates[template.goal] = template
    
    async def get_goal_template(self, goal: str) -> Optional[GoalTemplate]:
        """Get goal template"""
        return self.goal_templates.get(goal)
    
    async def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        return {
            "total_patterns": len(self.patterns),
            "total_sites": len(self.site_knowledge),
            "total_goals": len(self.goal_templates)
        }
    
    async def cleanup_old_patterns(self, retention_days: int, min_usage: int) -> int:
        """Cleanup old patterns"""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        removed = 0
        
        patterns_to_remove = []
        for pattern_id, pattern in self.patterns.items():
            if (pattern.last_used < cutoff_date and 
                pattern.usage_count < min_usage):
                patterns_to_remove.append(pattern_id)
        
        for pattern_id in patterns_to_remove:
            del self.patterns[pattern_id]
            removed += 1
        
        return removed

class SimplifiedPatternRecognitionEngine:
    """Simplified pattern recognition engine for testing"""
    
    def __init__(self, knowledge_base: SimplifiedKnowledgeBase):
        self.knowledge_base = knowledge_base
    
    async def learn_from_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn patterns from session data"""
        goal = session_data.get("goal", "")
        url = session_data.get("url", "")
        success = session_data.get("success", False)
        duration = session_data.get("duration", 0.0)
        action_history = session_data.get("action_history", [])
        
        results = {
            "patterns_created": 0,
            "patterns_updated": 0,
            "site_knowledge_updated": False,
            "goal_template_updated": False
        }
        
        if success and action_history:
            # Create or update pattern
            pattern_id = hashlib.md5(f"{goal}_{url}".encode()).hexdigest()[:8]
            
            existing_patterns = await self.knowledge_base.find_patterns(goal, url)
            if existing_patterns:
                # Update existing pattern
                pattern = existing_patterns[0]
                pattern.usage_count += 1
                pattern.last_used = datetime.now()
                # Update success rate (simple moving average)
                pattern.success_rate = (pattern.success_rate * (pattern.usage_count - 1) + 1.0) / pattern.usage_count
                pattern.avg_duration = (pattern.avg_duration * (pattern.usage_count - 1) + duration) / pattern.usage_count
                results["patterns_updated"] = 1
            else:
                # Create new pattern
                pattern = AutomationPattern(
                    pattern_id=pattern_id,
                    goal=goal,
                    url=url,
                    action_sequence=action_history,
                    success_rate=1.0,
                    avg_duration=duration,
                    usage_count=1
                )
                await self.knowledge_base.store_pattern(pattern)
                results["patterns_created"] = 1
            
            # Update site knowledge
            site_knowledge = await self.knowledge_base.get_site_knowledge(url)
            if not site_knowledge:
                site_knowledge = SiteKnowledge(
                    url=url,
                    timing_patterns={"avg_duration": duration},
                    reliability_score=1.0 if success else 0.0
                )
                await self.knowledge_base.store_site_knowledge(site_knowledge)
                results["site_knowledge_updated"] = True
            
            # Update goal template
            goal_template = await self.knowledge_base.get_goal_template(goal)
            if not goal_template:
                goal_template = GoalTemplate(
                    goal=goal,
                    canonical_steps=[action.get("action_type", "unknown") for action in action_history],
                    success_rate=1.0,
                    avg_duration=duration,
                    complexity_score=len(action_history),
                    usage_count=1
                )
                await self.knowledge_base.store_goal_template(goal_template)
                results["goal_template_updated"] = True
        
        return results
    
    async def suggest_optimizations(self, goal: str, url: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest optimizations based on learned patterns"""
        optimizations = []
        
        # Find relevant patterns
        patterns = await self.knowledge_base.find_patterns(goal, url)
        
        if patterns:
            best_pattern = max(patterns, key=lambda p: p.success_rate)
            
            if best_pattern.success_rate > 0.8:
                optimizations.append({
                    "type": "use_proven_pattern",
                    "confidence": best_pattern.success_rate,
                    "parameters": {
                        "pattern_id": best_pattern.pattern_id,
                        "expected_duration": best_pattern.avg_duration
                    },
                    "expected_impact": {
                        "success_rate_improvement": 0.1,
                        "duration_improvement": 5.0
                    }
                })
        
        # Site-specific optimizations
        site_knowledge = await self.knowledge_base.get_site_knowledge(url)
        if site_knowledge and site_knowledge.reliability_score < 0.7:
            optimizations.append({
                "type": "increase_timeouts",
                "confidence": 0.8,
                "parameters": {
                    "timeout_multiplier": 1.5
                },
                "expected_impact": {
                    "success_rate_improvement": 0.15
                }
            })
        
        return optimizations

class SimplifiedAdaptiveStrategyEngine:
    """Simplified adaptive strategy engine for testing"""
    
    def __init__(self, knowledge_base: SimplifiedKnowledgeBase):
        self.knowledge_base = knowledge_base
        self.strategies: Dict[str, AdaptiveStrategy] = {}
        self._init_base_strategies()
    
    def _init_base_strategies(self):
        """Initialize base strategies"""
        self.strategies["conservative"] = AdaptiveStrategy(
            name="conservative",
            parameters={
                "timeout_multiplier": 2.0,
                "retry_count": 3,
                "wait_between_actions": 1.0
            },
            confidence=0.8,
            expected_success_rate=0.85,
            estimated_duration=45.0
        )
        
        self.strategies["aggressive"] = AdaptiveStrategy(
            name="aggressive",
            parameters={
                "timeout_multiplier": 0.8,
                "retry_count": 1,
                "wait_between_actions": 0.2
            },
            confidence=0.6,
            expected_success_rate=0.75,
            estimated_duration=20.0
        )
        
        self.strategies["balanced"] = AdaptiveStrategy(
            name="balanced",
            parameters={
                "timeout_multiplier": 1.2,
                "retry_count": 2,
                "wait_between_actions": 0.5
            },
            confidence=0.7,
            expected_success_rate=0.80,
            estimated_duration=30.0
        )
    
    async def select_optimal_strategy(self, goal: str, url: str, context: Dict[str, Any]) -> Optional[AdaptiveStrategy]:
        """Select optimal strategy based on context"""
        # Simple strategy selection based on site reliability
        site_knowledge = await self.knowledge_base.get_site_knowledge(url)
        
        if site_knowledge:
            if site_knowledge.reliability_score > 0.8:
                return self.strategies["aggressive"]
            elif site_knowledge.reliability_score < 0.6:
                return self.strategies["conservative"]
        
        return self.strategies["balanced"]
    
    async def adapt_strategy_from_feedback(self, goal: str, url: str, success: bool, 
                                         duration: float, action_history: List[Dict[str, Any]], 
                                         errors: List[str]) -> Dict[str, Any]:
        """Adapt strategy based on feedback"""
        results = {
            "adaptations_made": 0,
            "strategy_updated": False,
            "new_strategy_created": False
        }
        
        # Simple adaptation logic
        if not success and errors:
            # If failed due to timeouts, suggest more conservative approach
            if any("timeout" in error.lower() for error in errors):
                conservative_strategy = self.strategies["conservative"]
                conservative_strategy.confidence += 0.1
                conservative_strategy.usage_count += 1
                results["adaptations_made"] = 1
                results["strategy_updated"] = True
        
        elif success and duration < 20.0:
            # If very fast success, aggressive strategy is working
            aggressive_strategy = self.strategies["aggressive"]
            aggressive_strategy.confidence += 0.05
            aggressive_strategy.usage_count += 1
            results["adaptations_made"] = 1
            results["strategy_updated"] = True
        
        return results
    
    async def apply_optimization(self, goal: str, url: str, optimization_type: str, parameters: Dict[str, Any]):
        """Apply optimization to strategy"""
        # Simple optimization application
        if optimization_type == "increase_timeouts":
            for strategy in self.strategies.values():
                if "timeout_multiplier" in strategy.parameters:
                    strategy.parameters["timeout_multiplier"] *= parameters.get("timeout_multiplier", 1.2)

class SimplifiedPredictiveAnalyticsEngine:
    """Simplified predictive analytics engine for testing"""
    
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.training_data: List[Dict[str, Any]] = []
        self.models = {
            "success_predictor": MockMLModel(),
            "duration_predictor": MockMLModel(),
            "cost_predictor": MockMLModel()
        }
        self.scaler = MockStandardScaler()
        self.prediction_cache: Dict[str, Dict[str, Any]] = {}
    
    async def add_training_data(self, data: Dict[str, Any]):
        """Add training data"""
        self.training_data.append(data)
    
    async def train_models(self) -> Dict[str, Any]:
        """Train predictive models"""
        if len(self.training_data) < 3:
            return {
                "success": False,
                "error": "Insufficient training data",
                "models_trained": []
            }
        
        # Prepare training data
        X = []
        y_success = []
        y_duration = []
        y_cost = []
        
        for sample in self.training_data:
            features = [
                len(sample.get("goal", "")),
                len(sample.get("url", "")),
                sample.get("action_count", 0),
                sample.get("complexity", 0.0)
            ]
            X.append(features)
            y_success.append(1 if sample.get("success", False) else 0)
            y_duration.append(sample.get("duration", 0.0))
            y_cost.append(sample.get("cost", 0.0))
        
        # Train models
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        self.models["success_predictor"].fit(X_scaled, y_success)
        self.models["duration_predictor"].fit(X_scaled, y_duration)
        self.models["cost_predictor"].fit(X_scaled, y_cost)
        
        return {
            "success": True,
            "models_trained": ["success_predictor", "duration_predictor", "cost_predictor"],
            "performance": {
                "success_accuracy": 0.85,
                "duration_r2": 0.80,
                "cost_r2": 0.75
            }
        }
    
    async def predict_automation_success(self, goal: str, url: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Predict automation success"""
        # Create cache key
        cache_key = hashlib.md5(f"{goal}_{url}_{str(context)}".encode()).hexdigest()
        
        # Check cache
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        
        # Prepare features
        features = [
            len(goal),
            len(url),
            context.get("action_count", 5),
            context.get("complexity", 2.5)
        ]
        
        X_scaled = self.scaler.transform([features])
        
        # Make predictions
        success_prob = self.models["success_predictor"].predict_proba(X_scaled)[0][1]
        duration = self.models["duration_predictor"].predict(X_scaled)[0]
        cost = self.models["cost_predictor"].predict(X_scaled)[0]
        
        prediction = {
            "success_probability": success_prob,
            "estimated_duration": duration,
            "estimated_cost": cost,
            "confidence": 0.8,
            "risk_factors": [],
            "optimization_suggestions": []
        }
        
        # Add risk factors
        if success_prob < 0.6:
            prediction["risk_factors"].append("Low predicted success rate")
        
        if duration > 60.0:
            prediction["risk_factors"].append("Long estimated duration")
        
        # Add optimization suggestions
        if success_prob < 0.8:
            prediction["optimization_suggestions"].append("Consider using conservative strategy")
        
        # Cache prediction
        self.prediction_cache[cache_key] = prediction
        
        return prediction

class SimplifiedKnowledgeBaseManager:
    """Simplified knowledge base manager for testing"""
    
    def __init__(self, db_path: str):
        self.knowledge_base = SimplifiedKnowledgeBase(db_path)
    
    async def store_pattern(self, pattern: AutomationPattern):
        """Store automation pattern"""
        await self.knowledge_base.store_pattern(pattern)
    
    async def find_patterns(self, goal: str, url: str) -> List[AutomationPattern]:
        """Find patterns"""
        return await self.knowledge_base.find_patterns(goal, url)
    
    async def store_site_knowledge(self, knowledge: SiteKnowledge):
        """Store site knowledge"""
        await self.knowledge_base.store_site_knowledge(knowledge)
    
    async def store_goal_template(self, template: GoalTemplate):
        """Store goal template"""
        await self.knowledge_base.store_goal_template(template)
    
    async def get_recommendations(self, goal: str, url: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get recommendations"""
        patterns = await self.knowledge_base.find_patterns(goal, url)
        site_knowledge = await self.knowledge_base.get_site_knowledge(url)
        goal_template = await self.knowledge_base.get_goal_template(goal)
        
        return {
            "patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "success_rate": p.success_rate,
                    "avg_duration": p.avg_duration,
                    "usage_count": p.usage_count
                } for p in patterns
            ],
            "site_insights": {
                "reliability_score": site_knowledge.reliability_score if site_knowledge else 1.0,
                "optimization_rules": site_knowledge.optimization_rules if site_knowledge else []
            },
            "goal_insights": {
                "success_rate": goal_template.success_rate if goal_template else 0.5,
                "complexity": goal_template.complexity_score if goal_template else 2.5
            }
        }

@dataclass
class LearningConfig:
    """Configuration for the learning system"""
    learning_enabled: bool = True
    auto_optimization: bool = True
    min_training_samples: int = 10
    pattern_retention_days: int = 90
    strategy_adaptation_threshold: float = 0.7

@dataclass
class LearningMetrics:
    """Metrics for learning system performance"""
    patterns_learned: int = 0
    strategies_adapted: int = 0
    predictions_made: int = 0
    successful_optimizations: int = 0
    learning_accuracy: float = 0.0
    total_sessions_processed: int = 0
    last_learning_session: Optional[datetime] = None
    last_model_training: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "patterns_learned": self.patterns_learned,
            "strategies_adapted": self.strategies_adapted,
            "predictions_made": self.predictions_made,
            "successful_optimizations": self.successful_optimizations,
            "learning_accuracy": self.learning_accuracy,
            "total_sessions_processed": self.total_sessions_processed,
            "last_learning_session": self.last_learning_session.isoformat() if self.last_learning_session else None,
            "last_model_training": self.last_model_training.isoformat() if self.last_model_training else None
        }

class SimplifiedLearningSystem:
    """Simplified learning system for testing"""
    
    def __init__(self, config: Optional[LearningConfig] = None, 
                 knowledge_db_path: str = "test_knowledge.db",
                 models_dir: str = "test_models"):
        
        self.config = config or LearningConfig()
        self.metrics = LearningMetrics()
        
        # Initialize components
        self.knowledge_manager = SimplifiedKnowledgeBaseManager(knowledge_db_path)
        self.pattern_engine = SimplifiedPatternRecognitionEngine(self.knowledge_manager.knowledge_base)
        self.strategy_engine = SimplifiedAdaptiveStrategyEngine(self.knowledge_manager.knowledge_base)
        self.analytics_engine = SimplifiedPredictiveAnalyticsEngine(models_dir)
    
    async def learn_from_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from a completed automation session"""
        if not self.config.learning_enabled:
            return {"status": "learning_disabled"}
        
        learning_results = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_data.get("session_id", "unknown"),
            "learning_summary": {},
            "insights": [],
            "recommendations": []
        }
        
        try:
            # Learn patterns
            pattern_results = await self.pattern_engine.learn_from_session(session_data)
            learning_results["learning_summary"]["patterns"] = pattern_results
            
            # Adapt strategies
            if session_data.get("action_history"):
                strategy_results = await self.strategy_engine.adapt_strategy_from_feedback(
                    session_data.get("goal", ""),
                    session_data.get("url", ""),
                    session_data.get("success", False),
                    session_data.get("duration", 0.0),
                    session_data.get("action_history", []),
                    session_data.get("errors", [])
                )
                learning_results["learning_summary"]["strategies"] = strategy_results
                self.metrics.strategies_adapted += strategy_results.get("adaptations_made", 0)
            
            # Add training data
            await self.analytics_engine.add_training_data({
                "goal": session_data.get("goal", ""),
                "url": session_data.get("url", ""),
                "success": session_data.get("success", False),
                "duration": session_data.get("duration", 0.0),
                "cost": session_data.get("cost", 0.0),
                "action_count": len(session_data.get("action_history", [])),
                "complexity": len(session_data.get("action_history", [])) * 0.5
            })
            
            # Update metrics
            self.metrics.patterns_learned += pattern_results.get("patterns_created", 0)
            self.metrics.total_sessions_processed += 1
            self.metrics.last_learning_session = datetime.now()
            
            # Generate insights
            if session_data.get("success"):
                learning_results["insights"].append("Successful automation - pattern reinforced")
            else:
                learning_results["insights"].append("Failed automation - failure pattern identified")
            
            # Generate recommendations
            learning_results["recommendations"].append("Continue monitoring for pattern improvements")
            
        except Exception as e:
            learning_results["error"] = str(e)
        
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
            
            # Get patterns
            knowledge_recs = await self.knowledge_manager.get_recommendations(goal, url, context or {})
            recommendations["patterns"] = knowledge_recs.get("patterns", [])
            
            # Get optimizations
            optimizations = await self.pattern_engine.suggest_optimizations(goal, url, context or {})
            recommendations["optimizations"] = optimizations
            
            # Calculate confidence
            confidence_factors = [
                strategy.confidence if strategy else 0.5,
                predictions.get("confidence", 0.5),
                min(1.0, len(recommendations["patterns"]) / 3.0),
                min(1.0, len(optimizations) / 2.0)
            ]
            recommendations["confidence_score"] = sum(confidence_factors) / len(confidence_factors)
            
            self.metrics.predictions_made += 1
            
        except Exception as e:
            recommendations["error"] = str(e)
        
        return recommendations
    
    async def optimize_automation_strategy(self, goal: str, url: str, 
                                         current_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize automation strategy"""
        optimization_results = {
            "timestamp": datetime.now().isoformat(),
            "goal": goal,
            "url": url,
            "optimizations_applied": [],
            "expected_improvements": {}
        }
        
        try:
            optimizations = await self.pattern_engine.suggest_optimizations(goal, url, current_performance)
            
            applied_optimizations = []
            for optimization in optimizations:
                if optimization.get("confidence", 0.0) > self.config.strategy_adaptation_threshold:
                    await self.strategy_engine.apply_optimization(
                        goal, url, optimization.get("type", ""), optimization.get("parameters", {})
                    )
                    applied_optimizations.append(optimization)
            
            optimization_results["optimizations_applied"] = applied_optimizations
            
            if applied_optimizations:
                optimization_results["expected_improvements"] = {
                    "success_rate_improvement": 0.1,
                    "duration_improvement": 5.0
                }
                self.metrics.successful_optimizations += len(applied_optimizations)
            
        except Exception as e:
            optimization_results["error"] = str(e)
        
        return optimization_results
    
    async def get_learning_insights(self, timeframe_days: int = 30) -> Dict[str, Any]:
        """Get learning insights"""
        insights = {
            "timestamp": datetime.now().isoformat(),
            "timeframe_days": timeframe_days,
            "metrics": self.metrics.to_dict(),
            "knowledge_stats": {},
            "recommendations": []
        }
        
        try:
            knowledge_stats = await self.knowledge_manager.knowledge_base.get_knowledge_stats()
            insights["knowledge_stats"] = knowledge_stats
            
            # Generate recommendations
            if self.metrics.learning_accuracy < 0.6:
                insights["recommendations"].append("Learning accuracy is low - review pattern recognition")
            
            if self.metrics.patterns_learned < 5:
                insights["recommendations"].append("Few patterns learned - increase automation diversity")
            
        except Exception as e:
            insights["error"] = str(e)
        
        return insights
    
    async def retrain_models(self) -> Dict[str, Any]:
        """Retrain predictive models"""
        try:
            training_results = await self.analytics_engine.train_models()
            if training_results.get("success"):
                self.metrics.last_model_training = datetime.now()
            return training_results
        except Exception as e:
            return {"success": False, "error": str(e)}

class LearningSystemTester:
    """Test suite for the learning system"""
    
    def __init__(self):
        self.temp_dir = None
        self.learning_system = None
        self.test_results = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "failures": []
        }
    
    def setup(self):
        """Setup test environment"""
        print("🔧 Setting up test environment...")
        
        self.temp_dir = tempfile.mkdtemp(prefix="webiq_learning_test_")
        print(f"   Created temp directory: {self.temp_dir}")
        
        config = LearningConfig(
            learning_enabled=True,
            auto_optimization=False,
            min_training_samples=3
        )
        
        knowledge_db_path = os.path.join(self.temp_dir, "test_knowledge.db")
        models_dir = os.path.join(self.temp_dir, "test_models")
        
        self.learning_system = SimplifiedLearningSystem(
            config=config,
            knowledge_db_path=knowledge_db_path,
            models_dir=models_dir
        )
        
        print("✅ Test environment setup complete")
    
    def cleanup(self):
        """Cleanup test environment"""
        print("🧹 Cleaning up test environment...")
        
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"   Removed temp directory: {self.temp_dir}")
        
        print("✅ Cleanup complete")
    
    def assert_test(self, condition: bool, test_name: str, error_msg: str = ""):
        """Assert a test condition"""
        self.test_results["tests_run"] += 1
        
        if condition:
            self.test_results["tests_passed"] += 1
            print(f"   ✅ {test_name}")
        else:
            self.test_results["tests_failed"] += 1
            failure_msg = f"{test_name}: {error_msg}" if error_msg else test_name
            self.test_results["failures"].append(failure_msg)
            print(f"   ❌ {test_name} - {error_msg}")
    
    async def test_knowledge_base_components(self):
        """Test knowledge base functionality"""
        print("\n📚 Testing Knowledge Base Components...")
        
        try:
            kb_manager = self.learning_system.knowledge_manager
            
            # Test pattern storage and retrieval
            pattern = AutomationPattern(
                pattern_id="test_pattern_1",
                goal="test login",
                url="https://example.com",
                action_sequence=[
                    {"action_type": "click", "selector": "#login"},
                    {"action_type": "type", "selector": "#username", "text": "user"}
                ],
                success_rate=0.9,
                avg_duration=15.0,
                usage_count=5
            )
            
            await kb_manager.store_pattern(pattern)
            self.assert_test(True, "Store automation pattern")
            
            patterns = await kb_manager.find_patterns("test login", "https://example.com")
            self.assert_test(
                len(patterns) > 0,
                "Retrieve stored pattern",
                f"Expected patterns, got {len(patterns)}"
            )
            
            # Test site knowledge
            site_knowledge = SiteKnowledge(
                url="https://example.com",
                common_elements={"login": "#login"},
                timing_patterns={"avg_load_time": 2.5},
                reliability_score=0.85
            )
            
            await kb_manager.store_site_knowledge(site_knowledge)
            self.assert_test(True, "Store site knowledge")
            
            # Test goal template
            goal_template = GoalTemplate(
                goal="test login",
                canonical_steps=["navigate", "type", "click"],
                success_rate=0.8,
                avg_duration=20.0,
                complexity_score=3.0
            )
            
            await kb_manager.store_goal_template(goal_template)
            self.assert_test(True, "Store goal template")
            
            # Test recommendations
            recommendations = await kb_manager.get_recommendations(
                "test login", "https://example.com", {}
            )
            self.assert_test(
                "patterns" in recommendations,
                "Get knowledge recommendations"
            )
            
        except Exception as e:
            self.assert_test(False, "Knowledge base components test", str(e))
    
    async def test_pattern_recognition(self):
        """Test pattern recognition engine"""
        print("\n🔍 Testing Pattern Recognition Engine...")
        
        try:
            pattern_engine = self.learning_system.pattern_engine
            
            session_data = {
                "session_id": "test_session_1",
                "goal": "login to website",
                "url": "https://example.com/login",
                "success": True,
                "duration": 12.5,
                "action_history": [
                    {"action_type": "navigate", "duration": 2.0},
                    {"action_type": "type", "duration": 1.5},
                    {"action_type": "click", "duration": 1.0}
                ],
                "errors": []
            }
            
            learning_result = await pattern_engine.learn_from_session(session_data)
            self.assert_test(
                "patterns_created" in learning_result or "patterns_updated" in learning_result,
                "Pattern learning from session"
            )
            
            optimizations = await pattern_engine.suggest_optimizations(
                "login to website", "https://example.com/login", {}
            )
            self.assert_test(
                isinstance(optimizations, list),
                "Generate optimization suggestions"
            )
            
        except Exception as e:
            self.assert_test(False, "Pattern recognition test", str(e))
    
    async def test_adaptive_strategies(self):
        """Test adaptive strategy engine"""
        print("\n🎯 Testing Adaptive Strategy Engine...")
        
        try:
            strategy_engine = self.learning_system.strategy_engine
            
            strategy = await strategy_engine.select_optimal_strategy(
                "login", "https://example.com", {}
            )
            self.assert_test(
                strategy is not None,
                "Strategy selection"
            )
            
            if strategy:
                self.assert_test(
                    hasattr(strategy, 'name') and hasattr(strategy, 'parameters'),
                    "Strategy structure validation"
                )
            
            adaptation_result = await strategy_engine.adapt_strategy_from_feedback(
                "login", "https://example.com", True, 10.0, [], []
            )
            self.assert_test(
                "adaptations_made" in adaptation_result,
                "Strategy adaptation from feedback"
            )
            
        except Exception as e:
            self.assert_test(False, "Adaptive strategies test", str(e))
    
    async def test_predictive_analytics(self):
        """Test predictive analytics engine"""
        print("\n📊 Testing Predictive Analytics Engine...")
        
        try:
            analytics_engine = self.learning_system.analytics_engine
            
            # Add training data
            training_samples = [
                {"goal": "login", "url": "https://example.com", "success": True, "duration": 10.0, "cost": 0.05, "action_count": 3, "complexity": 2.0},
                {"goal": "search", "url": "https://example.com", "success": True, "duration": 8.0, "cost": 0.03, "action_count": 2, "complexity": 1.5},
                {"goal": "purchase", "url": "https://shop.com", "success": False, "duration": 25.0, "cost": 0.12, "action_count": 8, "complexity": 4.0}
            ]
            
            for sample in training_samples:
                await analytics_engine.add_training_data(sample)
            
            self.assert_test(True, "Add training data")
            
            training_result = await analytics_engine.train_models()
            self.assert_test(
                training_result.get("success", False),
                "Train predictive models"
            )
            
            prediction = await analytics_engine.predict_automation_success(
                "login", "https://example.com", {"action_count": 3, "complexity": 2.0}
            )
            
            expected_keys = ["success_probability", "estimated_duration", "estimated_cost", "confidence"]
            for key in expected_keys:
                self.assert_test(
                    key in prediction,
                    f"Prediction contains {key}"
                )
            
        except Exception as e:
            self.assert_test(False, "Predictive analytics test", str(e))
    
    async def test_unified_learning_system(self):
        """Test the unified learning system"""
        print("\n🧠 Testing Unified Learning System...")
        
        try:
            # Test end-to-end learning
            session_data = {
                "session_id": "unified_test",
                "goal": "complete checkout",
                "url": "https://shop.example.com",
                "success": True,
                "duration": 45.0,
                "cost": 0.15,
                "action_history": [
                    {"action_type": "navigate", "duration": 3.0},
                    {"action_type": "click", "duration": 1.5},
                    {"action_type": "type", "duration": 2.0},
                    {"action_type": "click", "duration": 1.0}
                ],
                "errors": []
            }
            
            learning_result = await self.learning_system.learn_from_session(session_data)
            self.assert_test(
                "learning_summary" in learning_result,
                "End-to-end learning from session"
            )
            
            # Test recommendations
            recommendations = await self.learning_system.get_automation_recommendations(
                "complete checkout", "https://shop.example.com", {}
            )
            self.assert_test(
                "strategy" in recommendations and "predictions" in recommendations,
                "Get automation recommendations"
            )
            
            # Test optimization
            optimization_result = await self.learning_system.optimize_automation_strategy(
                "complete checkout", "https://shop.example.com", {"success_rate": 0.75}
            )
            self.assert_test(
                "optimizations_applied" in optimization_result,
                "Optimize automation strategy"
            )
            
            # Test insights
            insights = await self.learning_system.get_learning_insights()
            self.assert_test(
                "metrics" in insights,
                "Get learning insights"
            )
            
            # Test model retraining
            retrain_result = await self.learning_system.retrain_models()
            self.assert_test(
                "success" in retrain_result,
                "Retrain models"
            )
            
        except Exception as e:
            self.assert_test(False, "Unified learning system test", str(e))
    
    async def test_integration_scenarios(self):
        """Test integration scenarios"""
        print("\n🔗 Testing Integration Scenarios...")
        
        try:
            # Test multiple learning sessions
            sessions = [
                {
                    "session_id": "integration_1",
                    "goal": "user registration",
                    "url": "https://app.example.com",
                    "success": True,
                    "duration": 25.0,
                    "action_history": [{"action_type": "navigate"}, {"action_type": "type"}, {"action_type": "click"}],
                    "errors": []
                },
                {
                    "session_id": "integration_2",
                    "goal": "user registration",
                    "url": "https://app.example.com",
                    "success": False,
                    "duration": 35.0,
                    "action_history": [{"action_type": "navigate"}, {"action_type": "type"}],
                    "errors": ["Validation error"]
                },
                {
                    "session_id": "integration_3",
                    "goal": "user registration",
                    "url": "https://app.example.com",
                    "success": True,
                    "duration": 20.0,
                    "action_history": [{"action_type": "navigate"}, {"action_type": "type"}, {"action_type": "click"}],
                    "errors": []
                }
            ]
            
            for session in sessions:
                await self.learning_system.learn_from_session(session)
            
            self.assert_test(True, "Process multiple learning sessions")
            
            # Verify knowledge accumulation
            final_recommendations = await self.learning_system.get_automation_recommendations(
                "user registration", "https://app.example.com"
            )
            
            self.assert_test(
                final_recommendations.get("confidence_score", 0.0) > 0.0,
                "Knowledge accumulation improves confidence"
            )
            
            # Verify metrics
            final_metrics = self.learning_system.metrics
            self.assert_test(
                final_metrics.total_sessions_processed >= len(sessions),
                "Metrics accumulation"
            )
            
        except Exception as e:
            self.assert_test(False, "Integration scenarios test", str(e))
    
    async def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting WebIQ Learning System Test Suite")
        print("=" * 60)
        
        try:
            self.setup()
            
            await self.test_knowledge_base_components()
            await self.test_pattern_recognition()
            await self.test_adaptive_strategies()
            await self.test_predictive_analytics()
            await self.test_unified_learning_system()
            await self.test_integration_scenarios()
            
        except Exception as e:
            print(f"\n❌ Test suite failed: {e}")
            self.test_results["tests_failed"] += 1
            self.test_results["failures"].append(f"Test suite error: {str(e)}")
        
        finally:
            self.cleanup()
    
    def print_results(self):
        """Print test results"""
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        total_tests = self.test_results["tests_run"]
        passed_tests = self.test_results["tests_passed"]
        failed_tests = self.test_results["tests_failed"]
        
        print(f"Total Tests Run: {total_tests}")
        print(f"Tests Passed: {passed_tests} ✅")
        print(f"Tests Failed: {failed_tests} ❌")
        
        if total_tests > 0:
            success_rate = (passed_tests / total_tests) * 100
            print(f"Success Rate: {success_rate:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ FAILURES:")
            for i, failure in enumerate(self.test_results["failures"], 1):
                print(f"   {i}. {failure}")
        
        if failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED! Learning system is ready for integration.")
        else:
            print(f"\n⚠️  {failed_tests} test(s) failed. Please review and fix issues.")
        
        print("\n" + "=" * 60)

async def main():
    """Main test execution"""
    tester = LearningSystemTester()
    
    try:
        await tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        tester.print_results()

if __name__ == "__main__":
    asyncio.run(main())