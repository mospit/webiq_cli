# 🧠 **Learning & Adaptation System - Deep Dive**

Let's build a sophisticated learning system that makes your multi-agent automation smarter over time by learning from successes, failures, and patterns across all executions.

## 🎯 **Core Architecture Overview**

The Learning & Adaptation System operates on four interconnected levels:

1. **Pattern Recognition Engine** - Identifies successful automation patterns
2. **Knowledge Base Management** - Stores and retrieves learned insights
3. **Adaptive Strategy Engine** - Dynamically adjusts execution strategies
4. **Predictive Analytics** - Anticipates and prevents common failure modes

## 📊 **Detailed Implementation**

### 1. **Advanced Pattern Recognition Engine**

```python
import asyncio
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

class PatternType(Enum):
    SUCCESS_SEQUENCE = "success_sequence"
    FAILURE_PATTERN = "failure_pattern"
    SITE_SPECIFIC = "site_specific"
    GOAL_TEMPLATE = "goal_template"
    OPTIMIZATION_OPPORTUNITY = "optimization_opportunity"

@dataclass
class AutomationPattern:
    """Represents a learned automation pattern"""
    pattern_id: str
    pattern_type: PatternType
    confidence_score: float
    frequency: int
    success_rate: float
    context: Dict[str, Any]
    actions: List[Dict[str, Any]]
    conditions: List[str]
    outcomes: Dict[str, Any]
    created_at: datetime
    last_seen: datetime
    sites_applicable: Set[str] = field(default_factory=set)
    goals_applicable: Set[str] = field(default_factory=set)

class PatternRecognitionEngine:
    """Advanced pattern recognition and learning system"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.patterns: Dict[str, AutomationPattern] = {}
        self.session_history: List[Dict[str, Any]] = []
        self.site_knowledge: Dict[str, Dict[str, Any]] = {}
        self.goal_templates: Dict[str, Dict[str, Any]] = {}
        self.similarity_threshold = similarity_threshold
        
        # ML Components
        self.action_vectorizer = TfidfVectorizer(max_features=1000)
        self.pattern_clusters = {}
        self.success_predictors = {}
        
        # Knowledge persistence
        self.knowledge_file = "automation_knowledge.pkl"
        self.load_existing_knowledge()
    
    async def learn_from_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive learnings from a completed session"""
        
        session_id = session_data["session_id"]
        goal = session_data["goal"]
        url = session_data["url"]
        success = session_data.get("success", False)
        
        # Store session for historical analysis
        self.session_history.append({
            **session_data,
            "analyzed_at": datetime.now()
        })
        
        learning_results = {
            "patterns_discovered": [],
            "knowledge_updated": [],
            "predictions_refined": [],
            "optimizations_identified": []
        }
        
        # 1. Extract action sequences
        action_patterns = await self._extract_action_patterns(session_data)
        learning_results["patterns_discovered"].extend(action_patterns)
        
        # 2. Update site-specific knowledge
        site_updates = await self._update_site_knowledge(url, session_data)
        learning_results["knowledge_updated"].extend(site_updates)
        
        # 3. Refine goal templates
        goal_updates = await self._refine_goal_templates(goal, session_data)
        learning_results["knowledge_updated"].extend(goal_updates)
        
        # 4. Learn from failures
        if not success:
            failure_patterns = await self._learn_from_failures(session_data)
            learning_results["patterns_discovered"].extend(failure_patterns)
        
        # 5. Identify optimization opportunities
        optimizations = await self._identify_optimization_patterns(session_data)
        learning_results["optimizations_identified"].extend(optimizations)
        
        # 6. Update predictive models
        await self._update_predictive_models(session_data)
        
        # 7. Persist learned knowledge
        await self._persist_knowledge()
        
        return learning_results
    
    async def _extract_action_patterns(self, session_data: Dict[str, Any]) -> List[AutomationPattern]:
        """Extract successful action sequence patterns"""
        patterns = []
        
        action_history = session_data.get("action_history", [])
        if len(action_history) < 2:
            return patterns
        
        # Extract sequences of different lengths
        for seq_length in range(2, min(len(action_history) + 1, 8)):
            for i in range(len(action_history) - seq_length + 1):
                sequence = action_history[i:i + seq_length]
                
                # Create pattern signature
                pattern_signature = self._create_pattern_signature(sequence)
                
                # Check if this pattern already exists
                existing_pattern = self._find_similar_pattern(pattern_signature, PatternType.SUCCESS_SEQUENCE)
                
                if existing_pattern:
                    # Update existing pattern
                    existing_pattern.frequency += 1
                    existing_pattern.last_seen = datetime.now()
                    existing_pattern.sites_applicable.add(self._extract_domain(session_data["url"]))
                    existing_pattern.goals_applicable.add(session_data["goal"])
                else:
                    # Create new pattern
                    new_pattern = AutomationPattern(
                        pattern_id=pattern_signature,
                        pattern_type=PatternType.SUCCESS_SEQUENCE,
                        confidence_score=0.7,  # Initial confidence
                        frequency=1,
                        success_rate=1.0 if session_data.get("success", False) else 0.0,
                        context={
                            "sequence_length": seq_length,
                            "position_in_session": i,
                            "session_context": {
                                "goal_type": self._classify_goal_type(session_data["goal"]),
                                "site_type": self._classify_site_type(session_data["url"])
                            }
                        },
                        actions=sequence,
                        conditions=self._extract_conditions(sequence),
                        outcomes=self._extract_outcomes(sequence),
                        created_at=datetime.now(),
                        last_seen=datetime.now(),
                        sites_applicable={self._extract_domain(session_data["url"])},
                        goals_applicable={session_data["goal"]}
                    )
                    
                    self.patterns[pattern_signature] = new_pattern
                    patterns.append(new_pattern)
        
        return patterns
    
    async def _update_site_knowledge(self, url: str, session_data: Dict[str, Any]) -> List[str]:
        """Update site-specific knowledge and optimization rules"""
        domain = self._extract_domain(url)
        updates = []
        
        if domain not in self.site_knowledge:
            self.site_knowledge[domain] = {
                "first_seen": datetime.now(),
                "successful_sessions": 0,
                "failed_sessions": 0,
                "common_elements": {},
                "timing_patterns": {},
                "success_indicators": set(),
                "failure_patterns": [],
                "optimization_rules": []
            }
        
        site_data = self.site_knowledge[domain]
        
        # Update session counts
        if session_data.get("success", False):
            site_data["successful_sessions"] += 1
        else:
            site_data["failed_sessions"] += 1
        
        # Learn common elements and selectors
        dom_snapshots = session_data.get("dom_snapshots", [])
        for snapshot in dom_snapshots:
            elements = self._extract_elements_from_dom(snapshot)
            for element in elements:
                if element not in site_data["common_elements"]:
                    site_data["common_elements"][element] = 0
                site_data["common_elements"][element] += 1
        
        # Learn timing patterns
        action_history = session_data.get("action_history", [])
        timing_data = self._extract_timing_patterns(action_history)
        for timing_key, timing_value in timing_data.items():
            if timing_key not in site_data["timing_patterns"]:
                site_data["timing_patterns"][timing_key] = []
            site_data["timing_patterns"][timing_key].append(timing_value)
        
        # Update success indicators
        if session_data.get("success", False):
            success_indicators = self._extract_success_indicators(session_data)
            site_data["success_indicators"].update(success_indicators)
            updates.append(f"Updated success indicators for {domain}")
        
        # Generate optimization rules
        if site_data["successful_sessions"] >= 3:
            new_rules = self._generate_site_optimization_rules(domain, site_data)
            site_data["optimization_rules"].extend(new_rules)
            if new_rules:
                updates.append(f"Generated {len(new_rules)} optimization rules for {domain}")
        
        site_data["last_updated"] = datetime.now()
        return updates
    
    async def _refine_goal_templates(self, goal: str, session_data: Dict[str, Any]) -> List[str]:
        """Refine and improve goal-specific templates"""
        goal_type = self._classify_goal_type(goal)
        updates = []
        
        if goal_type not in self.goal_templates:
            self.goal_templates[goal_type] = {
                "canonical_steps": [],
                "common_variations": [],
                "success_patterns": [],
                "failure_modes": [],
                "optimization_strategies": [],
                "estimated_complexity": 5,
                "average_duration": 30.0,
                "success_rate": 0.0,
                "session_count": 0
            }
        
        template = self.goal_templates[goal_type]
        template["session_count"] += 1
        
        # Update success rate
        if session_data.get("success", False):
            current_successes = template["success_rate"] * (template["session_count"] - 1)
            template["success_rate"] = (current_successes + 1) / template["session_count"]
        else:
            current_successes = template["success_rate"] * (template["session_count"] - 1)
            template["success_rate"] = current_successes / template["session_count"]
        
        # Extract and refine canonical steps
        action_sequence = self._extract_canonical_sequence(session_data["action_history"])
        template["canonical_steps"] = self._merge_action_sequences(
            template["canonical_steps"], 
            action_sequence
        )
        
        # Update complexity estimation
        complexity = self._estimate_goal_complexity(session_data)
        template["estimated_complexity"] = (
            template["estimated_complexity"] * 0.8 + complexity * 0.2
        )
        
        # Update duration estimation
        duration = session_data.get("total_duration", 30.0)
        template["average_duration"] = (
            template["average_duration"] * 0.8 + duration * 0.2
        )
        
        updates.append(f"Refined template for goal type: {goal_type}")
        return updates
    
    async def _learn_from_failures(self, session_data: Dict[str, Any]) -> List[AutomationPattern]:
        """Extract learnings from failed sessions"""
        failure_patterns = []
        
        errors = session_data.get("errors", [])
        if not errors:
            return failure_patterns
        
        # Analyze error patterns
        for error in errors:
            error_context = self._analyze_error_context(error, session_data)
            
            # Create failure pattern
            pattern_signature = hashlib.md5(
                f"{error['type']}_{error_context['action_type']}_{error_context['element_type']}".encode()
            ).hexdigest()
            
            existing_pattern = self.patterns.get(pattern_signature)
            
            if existing_pattern:
                existing_pattern.frequency += 1
                existing_pattern.last_seen = datetime.now()
            else:
                failure_pattern = AutomationPattern(
                    pattern_id=pattern_signature,
                    pattern_type=PatternType.FAILURE_PATTERN,
                    confidence_score=0.8,
                    frequency=1,
                    success_rate=0.0,
                    context=error_context,
                    actions=[error.get("action", {})],
                    conditions=error_context.get("conditions", []),
                    outcomes={"error_type": error["type"], "error_message": error.get("message", "")},
                    created_at=datetime.now(),
                    last_seen=datetime.now(),
                    sites_applicable={self._extract_domain(session_data["url"])},
                    goals_applicable={session_data["goal"]}
                )
                
                self.patterns[pattern_signature] = failure_pattern
                failure_patterns.append(failure_pattern)
        
        return failure_patterns
    
    async def suggest_optimizations(self, goal: str, url: str, current_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Provide optimization suggestions based on learned patterns"""
        domain = self._extract_domain(url)
        goal_type = self._classify_goal_type(goal)
        
        suggestions = []
        
        # 1. Site-specific optimizations
        if domain in self.site_knowledge:
            site_data = self.site_knowledge[domain]
            
            # Suggest optimal timing
            if "timing_patterns" in site_data:
                timing_suggestions = self._generate_timing_suggestions(site_data["timing_patterns"])
                suggestions.extend(timing_suggestions)
            
            # Suggest reliable selectors
            if "common_elements" in site_data:
                selector_suggestions = self._generate_selector_suggestions(site_data["common_elements"])
                suggestions.extend(selector_suggestions)
            
            # Apply site-specific optimization rules
            rule_suggestions = self._apply_optimization_rules(site_data.get("optimization_rules", []))
            suggestions.extend(rule_suggestions)
        
        # 2. Goal-specific optimizations
        if goal_type in self.goal_templates:
            template = self.goal_templates[goal_type]
            
            # Suggest optimal action sequence
            if template["canonical_steps"]:
                sequence_suggestion = {
                    "type": "action_sequence",
                    "description": "Use proven action sequence for this goal type",
                    "sequence": template["canonical_steps"],
                    "confidence": template["success_rate"],
                    "estimated_duration": template["average_duration"]
                }
                suggestions.append(sequence_suggestion)
            
            # Suggest complexity-based strategy
            complexity = template["estimated_complexity"]
            if complexity > 7:
                suggestions.append({
                    "type": "strategy_adjustment",
                    "description": "High complexity goal detected - use conservative approach",
                    "recommendations": [
                        "Enable enhanced error recovery",
                        "Use longer timeouts",
                        "Enable detailed logging"
                    ]
                })
        
        # 3. Pattern-based optimizations
        relevant_patterns = self._find_relevant_patterns(goal, url, current_context)
        for pattern in relevant_patterns:
            if pattern.pattern_type == PatternType.SUCCESS_SEQUENCE and pattern.confidence_score > 0.8:
                suggestions.append({
                    "type": "proven_pattern",
                    "description": f"Use proven action pattern (success rate: {pattern.success_rate:.2f})",
                    "pattern": pattern.actions,
                    "confidence": pattern.confidence_score
                })
        
        # 4. Failure prevention suggestions
        failure_patterns = self._find_failure_patterns(goal, url)
        for failure_pattern in failure_patterns:
            suggestions.append({
                "type": "failure_prevention",
                "description": f"Avoid known failure pattern: {failure_pattern.outcomes.get('error_type', 'Unknown')}",
                "avoidance_strategy": self._generate_avoidance_strategy(failure_pattern),
                "risk_level": "high" if failure_pattern.frequency > 3 else "medium"
            })
        
        return suggestions
```

### 2. **Adaptive Strategy Engine**

```python
class AdaptiveStrategy:
    """Represents an adaptive execution strategy"""
    
    def __init__(self, strategy_id: str, base_strategy: Dict[str, Any]):
        self.strategy_id = strategy_id
        self.base_strategy = base_strategy
        self.adaptations: List[Dict[str, Any]] = []
        self.performance_history: List[float] = []
        self.success_count = 0
        self.failure_count = 0
        self.last_updated = datetime.now()
    
    def add_adaptation(self, adaptation: Dict[str, Any]):
        """Add a new adaptation to the strategy"""
        self.adaptations.append({
            **adaptation,
            "added_at": datetime.now()
        })
        self.last_updated = datetime.now()
    
    def get_current_strategy(self) -> Dict[str, Any]:
        """Get the current strategy with all adaptations applied"""
        current = self.base_strategy.copy()
        
        for adaptation in self.adaptations:
            if adaptation["type"] == "parameter_adjustment":
                current[adaptation["parameter"]] = adaptation["value"]
            elif adaptation["type"] == "behavior_modification":
                current["behaviors"] = current.get("behaviors", {})
                current["behaviors"][adaptation["behavior"]] = adaptation["enabled"]
            elif adaptation["type"] == "rule_addition":
                current["rules"] = current.get("rules", [])
                current["rules"].append(adaptation["rule"])
        
        return current
    
    def calculate_effectiveness(self) -> float:
        """Calculate the effectiveness of this strategy"""
        total_sessions = self.success_count + self.failure_count
        if total_sessions == 0:
            return 0.5  # Neutral effectiveness for new strategies
        
        success_rate = self.success_count / total_sessions
        
        # Factor in performance history
        if self.performance_history:
            avg_performance = sum(self.performance_history) / len(self.performance_history)
            # Combine success rate and performance (weighted average)
            return (success_rate * 0.7) + (avg_performance * 0.3)
        
        return success_rate

class AdaptiveStrategyEngine:
    """Manages and evolves execution strategies based on learned patterns"""
    
    def __init__(self, pattern_engine: PatternRecognitionEngine):
        self.pattern_engine = pattern_engine
        self.strategies: Dict[str, AdaptiveStrategy] = {}
        self.strategy_performance: Dict[str, List[float]] = {}
        self.context_strategies: Dict[str, str] = {}  # Context -> Strategy mapping
        
        # Initialize base strategies
        self._initialize_base_strategies()
    
    def _initialize_base_strategies(self):
        """Initialize base execution strategies"""
        
        # Conservative strategy for complex goals
        conservative_strategy = AdaptiveStrategy(
            "conservative",
            {
                "timeout_multiplier": 2.0,
                "retry_attempts": 5,
                "wait_between_actions": 1.5,
                "error_recovery_enabled": True,
                "detailed_logging": True,
                "fail_fast": False
            }
        )
        
        # Aggressive strategy for simple goals
        aggressive_strategy = AdaptiveStrategy(
            "aggressive",
            {
                "timeout_multiplier": 0.7,
                "retry_attempts": 2,
                "wait_between_actions": 0.3,
                "error_recovery_enabled": False,
                "detailed_logging": False,
                "fail_fast": True
            }
        )
        
        # Balanced strategy for general use
        balanced_strategy = AdaptiveStrategy(
            "balanced",
            {
                "timeout_multiplier": 1.0,
                "retry_attempts": 3,
                "wait_between_actions": 0.8,
                "error_recovery_enabled": True,
                "detailed_logging": True,
                "fail_fast": False
            }
        )
        
        self.strategies = {
            "conservative": conservative_strategy,
            "aggressive": aggressive_strategy,
            "balanced": balanced_strategy
        }
    
    async def select_optimal_strategy(self, goal: str, url: str, 
                                    current_context: Dict[str, Any]) -> Dict[str, Any]:
        """Select the optimal strategy for the current context"""
        
        context_key = self._generate_context_key(goal, url, current_context)
        
        # Check if we have a learned strategy for this context
        if context_key in self.context_strategies:
            strategy_id = self.context_strategies[context_key]
            strategy = self.strategies[strategy_id]
            return {
                "strategy_id": strategy_id,
                "strategy": strategy.get_current_strategy(),
                "confidence": strategy.calculate_effectiveness(),
                "source": "learned"
            }
        
        # Use pattern-based selection
        goal_type = self.pattern_engine._classify_goal_type(goal)
        domain = self.pattern_engine._extract_domain(url)
        
        # Analyze historical performance for similar contexts
        similar_contexts = self._find_similar_contexts(goal_type, domain)
        
        if similar_contexts:
            best_strategy_id = self._select_best_performing_strategy(similar_contexts)
            strategy = self.strategies[best_strategy_id]
            
            return {
                "strategy_id": best_strategy_id,
                "strategy": strategy.get_current_strategy(),
                "confidence": strategy.calculate_effectiveness(),
                "source": "pattern_based"
            }
        
        # Fallback to complexity-based selection
        complexity = self._estimate_context_complexity(goal, url, current_context)
        
        if complexity > 7:
            strategy_id = "conservative"
        elif complexity < 4:
            strategy_id = "aggressive"
        else:
            strategy_id = "balanced"
        
        strategy = self.strategies[strategy_id]
        
        return {
            "strategy_id": strategy_id,
            "strategy": strategy.get_current_strategy(),
            "confidence": 0.6,  # Lower confidence for fallback
            "source": "complexity_based"
        }
    
    async def adapt_strategy_from_feedback(self, strategy_id: str, 
                                         execution_result: Dict[str, Any],
                                         context: Dict[str, Any]):
        """Adapt strategy based on execution feedback"""
        
        if strategy_id not in self.strategies:
            return
        
        strategy = self.strategies[strategy_id]
        success = execution_result.get("success", False)
        performance_score = execution_result.get("performance_score", 0.5)
        
        # Update strategy performance
        strategy.performance_history.append(performance_score)
        if success:
            strategy.success_count += 1
        else:
            strategy.failure_count += 1
        
        # Keep only recent performance history
        if len(strategy.performance_history) > 100:
            strategy.performance_history = strategy.performance_history[-50:]
        
        # Analyze what adaptations might be needed
        adaptations = await self._analyze_needed_adaptations(
            strategy, execution_result, context
        )
        
        for adaptation in adaptations:
            strategy.add_adaptation(adaptation)
        
        # Update context-strategy mapping if this was successful
        if success and performance_score > 0.7:
            context_key = self._generate_context_key(
                context["goal"], context["url"], context
            )
            self.context_strategies[context_key] = strategy_id
    
    async def _analyze_needed_adaptations(self, strategy: AdaptiveStrategy,
                                        execution_result: Dict[str, Any],
                                        context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze what adaptations are needed based on execution results"""
        adaptations = []
        
        # Analyze timeout issues
        if "timeout" in execution_result.get("error_types", []):
            current_multiplier = strategy.get_current_strategy().get("timeout_multiplier", 1.0)
            if current_multiplier < 3.0:  # Don't make timeouts too long
                adaptations.append({
                    "type": "parameter_adjustment",
                    "parameter": "timeout_multiplier",
                    "value": current_multiplier * 1.3,
                    "reason": "Addressing timeout issues"
                })
        
        # Analyze rate limiting issues
        if "rate_limiting" in execution_result.get("error_types", []):
            current_wait = strategy.get_current_strategy().get("wait_between_actions", 0.8)
            adaptations.append({
                "type": "parameter_adjustment",
                "parameter": "wait_between_actions",
                "value": current_wait * 1.5,
                "reason": "Addressing rate limiting"
            })
        
        # Analyze element detection issues
        if "element_not_found" in execution_result.get("error_types", []):
            adaptations.append({
                "type": "behavior_modification",
                "behavior": "enhanced_element_detection",
                "enabled": True,
                "reason": "Improving element detection reliability"
            })
        
        # Analyze performance issues
        performance_score = execution_result.get("performance_score", 0.5)
        if performance_score < 0.4:
            # Poor performance - make strategy more conservative
            adaptations.append({
                "type": "parameter_adjustment",
                "parameter": "retry_attempts",
                "value": min(strategy.get_current_strategy().get("retry_attempts", 3) + 1, 7),
                "reason": "Improving reliability due to poor performance"
            })
        
        return adaptations
```

### 3. **Predictive Analytics Engine**

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

class PredictiveAnalyticsEngine:
    """Predicts automation success and optimal strategies using machine learning"""
    
    def __init__(self, pattern_engine: PatternRecognitionEngine):
        self.pattern_engine = pattern_engine
        self.success_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        self.duration_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.cost_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        self.feature_scaler = StandardScaler()
        self.models_trained = False
        self.feature_names = []
        
        # Prediction cache
        self.prediction_cache = {}
        self.cache_ttl = timedelta(hours=1)
    
    async def train_predictive_models(self) -> Dict[str, Any]:
        """Train ML models on historical automation data"""
        
        if len(self.pattern_engine.session_history) < 50:
            return {"status": "insufficient_data", "sessions_needed": 50 - len(self.pattern_engine.session_history)}
        
        # Prepare training data
        features, labels = await self._prepare_training_data()
        
        if len(features) < 30:
            return {"status": "insufficient_valid_data"}
        
        # Split data
        X_train, X_test, y_success_train, y_success_test = train_test_split(
            features, labels["success"], test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # Train success predictor
        self.success_predictor.fit(X_train_scaled, y_success_train)
        success_accuracy = self.success_predictor.score(X_test_scaled, y_success_test)
        
        # Train duration predictor
        duration_data = [(f, d) for f, d in zip(features, labels["duration"]) if d > 0]
        if len(duration_data) > 10:
            dur_features, dur_labels = zip(*duration_data)
            dur_features_scaled = self.feature_scaler.transform(dur_features)
            self.duration_predictor.fit(dur_features_scaled, dur_labels)
        
        # Train cost predictor
        cost_data = [(f, c) for f, c in zip(features, labels["cost"]) if c > 0]
        if len(cost_data) > 10:
            cost_features, cost_labels = zip(*cost_data)
            cost_features_scaled = self.feature_scaler.transform(cost_features)
            self.cost_predictor.fit(cost_features_scaled, cost_labels)
        
        self.models_trained = True
        
        # Save models
        await self._save_models()
        
        return {
            "status": "success",
            "success_accuracy": success_accuracy,
            "training_samples": len(features),
            "feature_count": len(self.feature_names)
        }
    
    async def predict_automation_success(self, goal: str, url: str, 
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Predict the likelihood of automation success"""
        
        if not self.models_trained:
            return {"error": "Models not trained", "confidence": 0.5}
        
        # Check cache
        cache_key = self._generate_prediction_cache_key(goal, url, context)
        if cache_key in self.prediction_cache:
            cached_result, timestamp = self.prediction_cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                return cached_result
        
        # Extract features
        features = await self._extract_prediction_features(goal, url, context)
        features_scaled = self.feature_scaler.transform([features])
        
        # Make predictions
        success_probability = self.success_predictor.predict_proba(features_scaled)[0][1]
        predicted_duration = 0
        predicted_cost = 0
        
        try:
            predicted_duration = max(0, self.duration_predictor.predict(features_scaled)[0])
            predicted_cost = max(0, self.cost_predictor.predict(features_scaled)[0])
        except:
            # Fallback if duration/cost models aren't trained
            predicted_duration = 30.0
            predicted_cost = 0.05
        
        # Get feature importance for explanation
        feature_importance = self._explain_prediction(features)
        
        result = {
            "success_probability": float(success_probability),
            "predicted_duration_seconds": float(predicted_duration),
            "predicted_cost_usd": float(predicted_cost),
            "confidence_level": self._calculate_prediction_confidence(features),
            "risk_factors": self._identify_risk_factors(features, feature_importance),
            "optimization_suggestions": await self._generate_prediction_based_optimizations(
                goal, url, context, success_probability, features
            )
        }
        
        # Cache result
        self.prediction_cache[cache_key] = (result, datetime.now())
        
        return result
    
    async def _prepare_training_data(self) -> Tuple[List[List[float]], Dict[str, List]]:
        """Prepare training