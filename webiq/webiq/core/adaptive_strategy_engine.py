import asyncio
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import logging
import statistics
import hashlib

logger = logging.getLogger(__name__)

class StrategyType(Enum):
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"
    SITE_SPECIFIC = "site_specific"

class AdaptationTrigger(Enum):
    FAILURE_RATE = "failure_rate"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    NEW_SITE_PATTERN = "new_site_pattern"
    SUCCESS_RATE_IMPROVEMENT = "success_rate_improvement"
    COST_OPTIMIZATION = "cost_optimization"

@dataclass
class AdaptiveStrategy:
    """Represents an adaptive automation strategy"""
    strategy_id: str
    strategy_type: StrategyType
    name: str
    description: str
    parameters: Dict[str, Any]
    success_metrics: Dict[str, float]
    failure_patterns: List[str]
    applicable_contexts: List[str]
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    adaptation_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_adapted: datetime = field(default_factory=datetime.now)
    confidence_score: float = 0.5
    usage_count: int = 0

class AdaptiveStrategyEngine:
    """Engine for selecting and adapting automation strategies"""
    
    def __init__(self, adaptation_threshold: float = 0.3):
        self.strategies: Dict[str, AdaptiveStrategy] = {}
        self.strategy_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.context_strategy_mapping: Dict[str, List[str]] = defaultdict(list)
        self.adaptation_threshold = adaptation_threshold
        
        # ML Components for strategy selection
        self.strategy_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_scaler = StandardScaler()
        self.is_trained = False
        
        # Performance tracking
        self.execution_history: List[Dict[str, Any]] = []
        self.adaptation_history: List[Dict[str, Any]] = []
        
        # Initialize base strategies
        self._initialize_base_strategies()
        
        # Load existing knowledge
        self.knowledge_file = "strategy_knowledge.pkl"
        self.load_existing_knowledge()
    
    def _initialize_base_strategies(self):
        """Initialize base automation strategies"""
        
        # Conservative Strategy
        conservative = AdaptiveStrategy(
            strategy_id="conservative_base",
            strategy_type=StrategyType.CONSERVATIVE,
            name="Conservative Automation",
            description="Safe, slow approach with extensive error handling",
            parameters={
                "timeout_multiplier": 2.0,
                "retry_attempts": 5,
                "wait_between_actions": 2.0,
                "error_recovery_enabled": True,
                "screenshot_on_error": True,
                "detailed_logging": True,
                "element_wait_timeout": 30.0,
                "page_load_timeout": 60.0,
                "interaction_delay": 1.0
            },
            success_metrics={"reliability": 0.95, "speed": 0.3},
            failure_patterns=["timeout", "element_not_found"],
            applicable_contexts=["complex_sites", "unreliable_networks", "critical_tasks"]
        )
        
        # Aggressive Strategy
        aggressive = AdaptiveStrategy(
            strategy_id="aggressive_base",
            strategy_type=StrategyType.AGGRESSIVE,
            name="Aggressive Automation",
            description="Fast approach with minimal delays",
            parameters={
                "timeout_multiplier": 0.5,
                "retry_attempts": 2,
                "wait_between_actions": 0.1,
                "error_recovery_enabled": False,
                "screenshot_on_error": False,
                "detailed_logging": False,
                "element_wait_timeout": 5.0,
                "page_load_timeout": 15.0,
                "interaction_delay": 0.0
            },
            success_metrics={"reliability": 0.7, "speed": 0.9},
            failure_patterns=["rate_limiting", "premature_interaction"],
            applicable_contexts=["simple_sites", "fast_networks", "batch_operations"]
        )
        
        # Balanced Strategy
        balanced = AdaptiveStrategy(
            strategy_id="balanced_base",
            strategy_type=StrategyType.BALANCED,
            name="Balanced Automation",
            description="Moderate approach balancing speed and reliability",
            parameters={
                "timeout_multiplier": 1.0,
                "retry_attempts": 3,
                "wait_between_actions": 0.5,
                "error_recovery_enabled": True,
                "screenshot_on_error": True,
                "detailed_logging": True,
                "element_wait_timeout": 15.0,
                "page_load_timeout": 30.0,
                "interaction_delay": 0.3
            },
            success_metrics={"reliability": 0.85, "speed": 0.6},
            failure_patterns=[],
            applicable_contexts=["general_automation", "mixed_complexity"]
        )
        
        self.strategies[conservative.strategy_id] = conservative
        self.strategies[aggressive.strategy_id] = aggressive
        self.strategies[balanced.strategy_id] = balanced
    
    async def select_optimal_strategy(self, context: Dict[str, Any]) -> AdaptiveStrategy:
        """Select the optimal strategy based on context and learned patterns"""
        
        goal = context.get("goal", "")
        url = context.get("url", "")
        site_complexity = context.get("site_complexity", 5)
        network_quality = context.get("network_quality", "good")
        task_priority = context.get("task_priority", "normal")
        
        # 1. Check for learned context-specific strategies
        context_key = self._generate_context_key(context)
        if context_key in self.context_strategy_mapping:
            candidate_strategies = self.context_strategy_mapping[context_key]
            if candidate_strategies:
                # Select best performing strategy for this context
                best_strategy_id = self._select_best_performing_strategy(candidate_strategies, context)
                if best_strategy_id:
                    strategy = self.strategies[best_strategy_id]
                    strategy.usage_count += 1
                    return strategy
        
        # 2. Use ML-based selection if trained
        if self.is_trained:
            predicted_strategy_id = await self._predict_optimal_strategy(context)
            if predicted_strategy_id and predicted_strategy_id in self.strategies:
                strategy = self.strategies[predicted_strategy_id]
                strategy.usage_count += 1
                return strategy
        
        # 3. Fallback to rule-based selection
        strategy = self._rule_based_strategy_selection(context)
        strategy.usage_count += 1
        return strategy
    
    async def adapt_strategy_from_feedback(self, strategy_id: str, execution_result: Dict[str, Any]) -> Optional[AdaptiveStrategy]:
        """Adapt strategy based on execution feedback"""
        
        if strategy_id not in self.strategies:
            logger.warning(f"Strategy {strategy_id} not found for adaptation")
            return None
        
        strategy = self.strategies[strategy_id]
        
        # Record performance
        performance_record = {
            "timestamp": datetime.now(),
            "success": execution_result.get("success", False),
            "duration": execution_result.get("duration", 0),
            "error_count": len(execution_result.get("errors", [])),
            "context": execution_result.get("context", {})
        }
        
        strategy.performance_history.append(performance_record)
        self.strategy_performance[strategy_id].append(performance_record)
        
        # Analyze if adaptation is needed
        adaptation_needed, adaptation_reasons = await self._analyze_needed_adaptations(strategy, execution_result)
        
        if adaptation_needed:
            adapted_strategy = await self._create_adapted_strategy(strategy, adaptation_reasons, execution_result)
            if adapted_strategy:
                self.strategies[adapted_strategy.strategy_id] = adapted_strategy
                
                # Record adaptation
                self.adaptation_history.append({
                    "timestamp": datetime.now(),
                    "original_strategy": strategy_id,
                    "adapted_strategy": adapted_strategy.strategy_id,
                    "reasons": adaptation_reasons,
                    "trigger_result": execution_result
                })
                
                logger.info(f"Adapted strategy {strategy_id} -> {adapted_strategy.strategy_id}")
                return adapted_strategy
        
        # Update strategy confidence based on performance
        await self._update_strategy_confidence(strategy, execution_result)
        
        return strategy
    
    async def _analyze_needed_adaptations(self, strategy: AdaptiveStrategy, execution_result: Dict[str, Any]) -> Tuple[bool, List[AdaptationTrigger]]:
        """Analyze if strategy adaptation is needed"""
        
        adaptation_triggers = []
        
        # Check failure rate
        recent_performance = strategy.performance_history[-10:]  # Last 10 executions
        if len(recent_performance) >= 5:
            failure_rate = sum(1 for p in recent_performance if not p["success"]) / len(recent_performance)
            if failure_rate > self.adaptation_threshold:
                adaptation_triggers.append(AdaptationTrigger.FAILURE_RATE)
        
        # Check performance degradation
        if len(strategy.performance_history) >= 20:
            recent_durations = [p["duration"] for p in strategy.performance_history[-10:] if p["success"]]
            older_durations = [p["duration"] for p in strategy.performance_history[-20:-10] if p["success"]]
            
            if recent_durations and older_durations:
                recent_avg = statistics.mean(recent_durations)
                older_avg = statistics.mean(older_durations)
                
                if recent_avg > older_avg * 1.5:  # 50% performance degradation
                    adaptation_triggers.append(AdaptationTrigger.PERFORMANCE_DEGRADATION)
        
        # Check for new error patterns
        current_errors = execution_result.get("errors", [])
        for error in current_errors:
            error_type = error.get("type", "unknown")
            if error_type not in strategy.failure_patterns:
                adaptation_triggers.append(AdaptationTrigger.NEW_SITE_PATTERN)
                break
        
        # Check for cost optimization opportunities
        if execution_result.get("cost", 0) > strategy.success_metrics.get("expected_cost", float('inf')):
            adaptation_triggers.append(AdaptationTrigger.COST_OPTIMIZATION)
        
        return len(adaptation_triggers) > 0, adaptation_triggers
    
    async def _create_adapted_strategy(self, base_strategy: AdaptiveStrategy, triggers: List[AdaptationTrigger], execution_result: Dict[str, Any]) -> Optional[AdaptiveStrategy]:
        """Create an adapted version of the strategy"""
        
        # Create new strategy ID
        adaptation_id = hashlib.md5(f"{base_strategy.strategy_id}_{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        new_strategy_id = f"{base_strategy.strategy_id}_adapted_{adaptation_id}"
        
        # Copy base parameters
        new_parameters = base_strategy.parameters.copy()
        adaptation_description = []
        
        # Apply adaptations based on triggers
        for trigger in triggers:
            if trigger == AdaptationTrigger.FAILURE_RATE:
                # Increase reliability parameters
                new_parameters["timeout_multiplier"] *= 1.5
                new_parameters["retry_attempts"] = min(new_parameters["retry_attempts"] + 2, 10)
                new_parameters["wait_between_actions"] *= 1.3
                new_parameters["error_recovery_enabled"] = True
                adaptation_description.append("Increased reliability due to high failure rate")
            
            elif trigger == AdaptationTrigger.PERFORMANCE_DEGRADATION:
                # Optimize for speed while maintaining some reliability
                new_parameters["element_wait_timeout"] *= 0.8
                new_parameters["page_load_timeout"] *= 0.9
                new_parameters["interaction_delay"] *= 0.7
                adaptation_description.append("Optimized timing due to performance degradation")
            
            elif trigger == AdaptationTrigger.NEW_SITE_PATTERN:
                # Add defensive measures
                new_parameters["screenshot_on_error"] = True
                new_parameters["detailed_logging"] = True
                new_parameters["wait_between_actions"] *= 1.2
                adaptation_description.append("Added defensive measures for new error patterns")
            
            elif trigger == AdaptationTrigger.COST_OPTIMIZATION:
                # Reduce resource usage
                new_parameters["screenshot_on_error"] = False
                new_parameters["detailed_logging"] = False
                new_parameters["retry_attempts"] = max(new_parameters["retry_attempts"] - 1, 1)
                adaptation_description.append("Optimized for cost reduction")
        
        # Create adapted strategy
        adapted_strategy = AdaptiveStrategy(
            strategy_id=new_strategy_id,
            strategy_type=StrategyType.ADAPTIVE,
            name=f"{base_strategy.name} (Adapted)",
            description=f"{base_strategy.description}. Adaptations: {'; '.join(adaptation_description)}",
            parameters=new_parameters,
            success_metrics=base_strategy.success_metrics.copy(),
            failure_patterns=base_strategy.failure_patterns.copy(),
            applicable_contexts=base_strategy.applicable_contexts.copy(),
            adaptation_count=base_strategy.adaptation_count + 1,
            confidence_score=0.6  # Start with moderate confidence
        )
        
        # Update failure patterns with new errors
        current_errors = execution_result.get("errors", [])
        for error in current_errors:
            error_type = error.get("type", "unknown")
            if error_type not in adapted_strategy.failure_patterns:
                adapted_strategy.failure_patterns.append(error_type)
        
        return adapted_strategy
    
    def _generate_context_key(self, context: Dict[str, Any]) -> str:
        """Generate a key for context-based strategy mapping"""
        
        # Extract key context elements
        url = context.get("url", "")
        goal_type = self._classify_goal_type(context.get("goal", ""))
        site_complexity = context.get("site_complexity", 5)
        network_quality = context.get("network_quality", "good")
        
        # Create normalized key
        domain = self._extract_domain(url)
        complexity_tier = "simple" if site_complexity <= 3 else "medium" if site_complexity <= 7 else "complex"
        
        return f"{domain}_{goal_type}_{complexity_tier}_{network_quality}"
    
    def _select_best_performing_strategy(self, strategy_ids: List[str], context: Dict[str, Any]) -> Optional[str]:
        """Select the best performing strategy from candidates"""
        
        best_strategy_id = None
        best_score = -1
        
        for strategy_id in strategy_ids:
            if strategy_id not in self.strategies:
                continue
            
            strategy = self.strategies[strategy_id]
            
            # Calculate performance score
            recent_performance = strategy.performance_history[-10:]
            if not recent_performance:
                continue
            
            success_rate = sum(1 for p in recent_performance if p["success"]) / len(recent_performance)
            avg_duration = statistics.mean([p["duration"] for p in recent_performance if p["success"]])
            
            # Normalize duration (lower is better)
            duration_score = max(0, 1 - (avg_duration / 60))  # Assume 60s is poor performance
            
            # Combined score
            performance_score = (success_rate * 0.7) + (duration_score * 0.3)
            
            if performance_score > best_score:
                best_score = performance_score
                best_strategy_id = strategy_id
        
        return best_strategy_id
    
    async def _predict_optimal_strategy(self, context: Dict[str, Any]) -> Optional[str]:
        """Use ML to predict optimal strategy"""
        
        try:
            # Extract features from context
            features = self._extract_context_features(context)
            features_scaled = self.feature_scaler.transform([features])
            
            # Predict strategy
            prediction = self.strategy_classifier.predict(features_scaled)[0]
            
            return prediction
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return None
    
    def _rule_based_strategy_selection(self, context: Dict[str, Any]) -> AdaptiveStrategy:
        """Fallback rule-based strategy selection"""
        
        site_complexity = context.get("site_complexity", 5)
        network_quality = context.get("network_quality", "good")
        task_priority = context.get("task_priority", "normal")
        
        # Simple rule-based logic
        if task_priority == "critical" or site_complexity > 7 or network_quality == "poor":
            return self.strategies["conservative_base"]
        elif site_complexity <= 3 and network_quality == "excellent":
            return self.strategies["aggressive_base"]
        else:
            return self.strategies["balanced_base"]
    
    async def _update_strategy_confidence(self, strategy: AdaptiveStrategy, execution_result: Dict[str, Any]):
        """Update strategy confidence based on execution result"""
        
        success = execution_result.get("success", False)
        
        # Simple confidence update
        if success:
            strategy.confidence_score = min(1.0, strategy.confidence_score + 0.05)
        else:
            strategy.confidence_score = max(0.0, strategy.confidence_score - 0.1)
    
    def _extract_context_features(self, context: Dict[str, Any]) -> List[float]:
        """Extract numerical features from context for ML"""
        
        features = [
            context.get("site_complexity", 5) / 10.0,  # Normalized complexity
            1.0 if context.get("network_quality") == "excellent" else 0.5 if context.get("network_quality") == "good" else 0.0,
            1.0 if context.get("task_priority") == "critical" else 0.5 if context.get("task_priority") == "high" else 0.0,
            len(context.get("goal", "")) / 100.0,  # Goal complexity proxy
            1.0 if "login" in context.get("goal", "").lower() else 0.0,  # Goal type indicators
            1.0 if "search" in context.get("goal", "").lower() else 0.0,
            1.0 if "form" in context.get("goal", "").lower() else 0.0,
            1.0 if "buy" in context.get("goal", "").lower() else 0.0
        ]
        
        return features
    
    def _classify_goal_type(self, goal: str) -> str:
        """Classify goal into a type category"""
        goal_lower = goal.lower()
        
        if any(word in goal_lower for word in ["login", "sign in", "authenticate"]):
            return "authentication"
        elif any(word in goal_lower for word in ["search", "find", "look for"]):
            return "search"
        elif any(word in goal_lower for word in ["buy", "purchase", "order", "checkout"]):
            return "ecommerce"
        elif any(word in goal_lower for word in ["form", "submit", "fill", "register"]):
            return "form_submission"
        elif any(word in goal_lower for word in ["navigate", "go to", "visit"]):
            return "navigation"
        else:
            return "general"
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return "unknown"
    
    async def train_strategy_selector(self, training_data: List[Dict[str, Any]]):
        """Train ML model for strategy selection"""
        
        if len(training_data) < 50:  # Need sufficient data
            logger.warning("Insufficient training data for strategy selector")
            return
        
        try:
            # Prepare training data
            X = []
            y = []
            
            for record in training_data:
                features = self._extract_context_features(record["context"])
                strategy_id = record["strategy_used"]
                
                X.append(features)
                y.append(strategy_id)
            
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Train classifier
            self.strategy_classifier.fit(X_scaled, y)
            self.is_trained = True
            
            logger.info(f"Trained strategy selector on {len(training_data)} samples")
            
        except Exception as e:
            logger.error(f"Failed to train strategy selector: {e}")
    
    async def get_strategy_recommendations(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get strategy recommendations with explanations"""
        
        recommendations = []
        
        # Get top 3 strategies
        context_key = self._generate_context_key(context)
        candidate_strategies = self.context_strategy_mapping.get(context_key, [])
        
        if not candidate_strategies:
            candidate_strategies = list(self.strategies.keys())[:3]
        
        for strategy_id in candidate_strategies[:3]:
            if strategy_id not in self.strategies:
                continue
            
            strategy = self.strategies[strategy_id]
            
            # Calculate recommendation score
            recent_performance = strategy.performance_history[-10:]
            if recent_performance:
                success_rate = sum(1 for p in recent_performance if p["success"]) / len(recent_performance)
                avg_duration = statistics.mean([p["duration"] for p in recent_performance if p["success"]])
            else:
                success_rate = 0.5
                avg_duration = 30.0
            
            recommendation = {
                "strategy_id": strategy_id,
                "name": strategy.name,
                "description": strategy.description,
                "confidence": strategy.confidence_score,
                "expected_success_rate": success_rate,
                "expected_duration": avg_duration,
                "usage_count": strategy.usage_count,
                "parameters": strategy.parameters,
                "recommendation_score": (success_rate * 0.4) + (strategy.confidence_score * 0.3) + (min(strategy.usage_count / 10, 1.0) * 0.3)
            }
            
            recommendations.append(recommendation)
        
        # Sort by recommendation score
        recommendations.sort(key=lambda x: x["recommendation_score"], reverse=True)
        
        return recommendations
    
    async def persist_knowledge(self):
        """Persist strategy knowledge to disk"""
        try:
            knowledge_data = {
                "strategies": {k: self._serialize_strategy(v) for k, v in self.strategies.items()},
                "context_strategy_mapping": dict(self.context_strategy_mapping),
                "adaptation_history": self.adaptation_history,
                "execution_history": self.execution_history[-1000:],  # Keep last 1000
                "last_updated": datetime.now()
            }
            
            with open(self.knowledge_file, 'wb') as f:
                pickle.dump(knowledge_data, f)
                
            logger.info(f"Persisted strategy knowledge: {len(self.strategies)} strategies")
        except Exception as e:
            logger.error(f"Failed to persist strategy knowledge: {e}")
    
    def load_existing_knowledge(self):
        """Load existing strategy knowledge from disk"""
        try:
            with open(self.knowledge_file, 'rb') as f:
                knowledge_data = pickle.load(f)
            
            # Load strategies (merge with base strategies)
            loaded_strategies = {k: self._deserialize_strategy(v) for k, v in knowledge_data.get("strategies", {}).items()}
            self.strategies.update(loaded_strategies)
            
            self.context_strategy_mapping = defaultdict(list, knowledge_data.get("context_strategy_mapping", {}))
            self.adaptation_history = knowledge_data.get("adaptation_history", [])
            self.execution_history = knowledge_data.get("execution_history", [])
            
            logger.info(f"Loaded strategy knowledge: {len(self.strategies)} strategies")
        except FileNotFoundError:
            logger.info("No existing strategy knowledge file found - starting fresh")
        except Exception as e:
            logger.error(f"Failed to load strategy knowledge: {e}")
    
    def _serialize_strategy(self, strategy: AdaptiveStrategy) -> Dict[str, Any]:
        """Serialize strategy for persistence"""
        return {
            "strategy_id": strategy.strategy_id,
            "strategy_type": strategy.strategy_type.value,
            "name": strategy.name,
            "description": strategy.description,
            "parameters": strategy.parameters,
            "success_metrics": strategy.success_metrics,
            "failure_patterns": strategy.failure_patterns,
            "applicable_contexts": strategy.applicable_contexts,
            "performance_history": strategy.performance_history,
            "adaptation_count": strategy.adaptation_count,
            "created_at": strategy.created_at.isoformat(),
            "last_adapted": strategy.last_adapted.isoformat(),
            "confidence_score": strategy.confidence_score,
            "usage_count": strategy.usage_count
        }
    
    def _deserialize_strategy(self, data: Dict[str, Any]) -> AdaptiveStrategy:
        """Deserialize strategy from persistence"""
        return AdaptiveStrategy(
            strategy_id=data["strategy_id"],
            strategy_type=StrategyType(data["strategy_type"]),
            name=data["name"],
            description=data["description"],
            parameters=data["parameters"],
            success_metrics=data["success_metrics"],
            failure_patterns=data["failure_patterns"],
            applicable_contexts=data["applicable_contexts"],
            performance_history=data["performance_history"],
            adaptation_count=data["adaptation_count"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_adapted=datetime.fromisoformat(data["last_adapted"]),
            confidence_score=data["confidence_score"],
            usage_count=data["usage_count"]
        )