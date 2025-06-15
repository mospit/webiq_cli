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
import statistics
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

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
    
    # Helper methods
    def _create_pattern_signature(self, sequence: List[Dict[str, Any]]) -> str:
        """Create a unique signature for an action sequence"""
        action_types = [action.get("type", "unknown") for action in sequence]
        signature_string = "_".join(action_types)
        return hashlib.md5(signature_string.encode()).hexdigest()
    
    def _find_similar_pattern(self, pattern_signature: str, pattern_type: PatternType) -> Optional[AutomationPattern]:
        """Find similar existing pattern"""
        return self.patterns.get(pattern_signature)
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            return urlparse(url).netloc
        except:
            return "unknown"
    
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
    
    def _classify_site_type(self, url: str) -> str:
        """Classify site type based on URL"""
        domain = self._extract_domain(url).lower()
        
        if any(word in domain for word in ["shop", "store", "buy", "cart", "amazon", "ebay"]):
            return "ecommerce"
        elif any(word in domain for word in ["social", "facebook", "twitter", "linkedin"]):
            return "social_media"
        elif any(word in domain for word in ["news", "blog", "article"]):
            return "content"
        elif any(word in domain for word in ["bank", "finance", "payment"]):
            return "financial"
        else:
            return "general"
    
    def _extract_conditions(self, sequence: List[Dict[str, Any]]) -> List[str]:
        """Extract conditions from action sequence"""
        conditions = []
        for action in sequence:
            if "selector" in action:
                conditions.append(f"element_exists:{action['selector']}")
            if "wait_for" in action:
                conditions.append(f"wait_condition:{action['wait_for']}")
        return conditions
    
    def _extract_outcomes(self, sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract outcomes from action sequence"""
        return {
            "sequence_length": len(sequence),
            "action_types": [action.get("type", "unknown") for action in sequence],
            "estimated_duration": sum(action.get("duration", 1.0) for action in sequence)
        }
    
    def _extract_elements_from_dom(self, dom_snapshot: Dict[str, Any]) -> List[str]:
        """Extract element information from DOM snapshot"""
        # Simplified implementation - would need actual DOM parsing
        return dom_snapshot.get("elements", [])
    
    def _extract_timing_patterns(self, action_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extract timing patterns from action history"""
        timing_data = {}
        
        for action in action_history:
            action_type = action.get("type", "unknown")
            duration = action.get("duration", 1.0)
            
            if action_type not in timing_data:
                timing_data[action_type] = []
            timing_data[action_type].append(duration)
        
        # Calculate averages
        return {k: statistics.mean(v) for k, v in timing_data.items()}
    
    def _extract_success_indicators(self, session_data: Dict[str, Any]) -> Set[str]:
        """Extract indicators that correlate with success"""
        indicators = set()
        
        if session_data.get("success", False):
            # Extract elements that were present during successful sessions
            for action in session_data.get("action_history", []):
                if "selector" in action:
                    indicators.add(f"success_element:{action['selector']}")
                if "text" in action:
                    indicators.add(f"success_text:{action['text']}")
        
        return indicators
    
    def _generate_site_optimization_rules(self, domain: str, site_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimization rules for a specific site"""
        rules = []
        
        # Generate timing-based rules
        if "timing_patterns" in site_data:
            for action_type, timings in site_data["timing_patterns"].items():
                if len(timings) >= 3:
                    avg_timing = statistics.mean(timings)
                    rules.append({
                        "type": "timing_optimization",
                        "action_type": action_type,
                        "recommended_wait": avg_timing * 1.2,  # Add 20% buffer
                        "confidence": min(len(timings) / 10, 1.0)
                    })
        
        # Generate element-based rules
        common_elements = site_data.get("common_elements", {})
        reliable_elements = {k: v for k, v in common_elements.items() if v >= 3}
        
        if reliable_elements:
            rules.append({
                "type": "element_preference",
                "reliable_selectors": list(reliable_elements.keys()),
                "confidence": 0.8
            })
        
        return rules
    
    def _extract_canonical_sequence(self, action_history: List[Dict[str, Any]]) -> List[str]:
        """Extract canonical action sequence"""
        return [action.get("type", "unknown") for action in action_history]
    
    def _merge_action_sequences(self, existing: List[str], new: List[str]) -> List[str]:
        """Merge action sequences to find common patterns"""
        if not existing:
            return new
        
        # Simple implementation - find longest common subsequence
        # In practice, this would use more sophisticated sequence alignment
        return existing if len(existing) >= len(new) else new
    
    def _estimate_goal_complexity(self, session_data: Dict[str, Any]) -> float:
        """Estimate goal complexity based on session data"""
        action_count = len(session_data.get("action_history", []))
        error_count = len(session_data.get("errors", []))
        duration = session_data.get("total_duration", 30.0)
        
        # Simple complexity scoring
        complexity = (action_count * 0.3) + (error_count * 2.0) + (duration / 10.0)
        return min(complexity, 10.0)  # Cap at 10
    
    def _analyze_error_context(self, error: Dict[str, Any], session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze context around an error"""
        return {
            "error_type": error.get("type", "unknown"),
            "action_type": error.get("action", {}).get("type", "unknown"),
            "element_type": error.get("element_type", "unknown"),
            "timing": error.get("timestamp", datetime.now()),
            "conditions": ["error_context"]
        }
    
    def _generate_timing_suggestions(self, timing_patterns: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """Generate timing-based suggestions"""
        suggestions = []
        
        for action_type, timings in timing_patterns.items():
            if len(timings) >= 3:
                avg_timing = statistics.mean(timings)
                suggestions.append({
                    "type": "timing_optimization",
                    "description": f"Optimal wait time for {action_type}: {avg_timing:.2f}s",
                    "action_type": action_type,
                    "recommended_wait": avg_timing,
                    "confidence": min(len(timings) / 10, 1.0)
                })
        
        return suggestions
    
    def _generate_selector_suggestions(self, common_elements: Dict[str, int]) -> List[Dict[str, Any]]:
        """Generate selector-based suggestions"""
        suggestions = []
        
        reliable_selectors = {k: v for k, v in common_elements.items() if v >= 3}
        
        if reliable_selectors:
            suggestions.append({
                "type": "selector_optimization",
                "description": "Use proven reliable selectors",
                "reliable_selectors": list(reliable_selectors.keys()),
                "confidence": 0.8
            })
        
        return suggestions
    
    def _apply_optimization_rules(self, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply site-specific optimization rules"""
        suggestions = []
        
        for rule in rules:
            if rule["type"] == "timing_optimization":
                suggestions.append({
                    "type": "timing_rule",
                    "description": f"Use optimized timing for {rule['action_type']}",
                    "recommended_wait": rule["recommended_wait"],
                    "confidence": rule["confidence"]
                })
            elif rule["type"] == "element_preference":
                suggestions.append({
                    "type": "element_rule",
                    "description": "Prefer reliable element selectors",
                    "selectors": rule["reliable_selectors"],
                    "confidence": rule["confidence"]
                })
        
        return suggestions
    
    def _find_relevant_patterns(self, goal: str, url: str, context: Dict[str, Any]) -> List[AutomationPattern]:
        """Find patterns relevant to current context"""
        domain = self._extract_domain(url)
        goal_type = self._classify_goal_type(goal)
        
        relevant = []
        
        for pattern in self.patterns.values():
            if (domain in pattern.sites_applicable or 
                goal in pattern.goals_applicable or
                goal_type in pattern.goals_applicable):
                relevant.append(pattern)
        
        # Sort by confidence and frequency
        relevant.sort(key=lambda p: (p.confidence_score, p.frequency), reverse=True)
        
        return relevant[:10]  # Return top 10
    
    def _find_failure_patterns(self, goal: str, url: str) -> List[AutomationPattern]:
        """Find failure patterns relevant to current context"""
        domain = self._extract_domain(url)
        
        failure_patterns = [
            p for p in self.patterns.values() 
            if p.pattern_type == PatternType.FAILURE_PATTERN and 
               (domain in p.sites_applicable or goal in p.goals_applicable)
        ]
        
        return failure_patterns
    
    def _generate_avoidance_strategy(self, failure_pattern: AutomationPattern) -> Dict[str, Any]:
        """Generate strategy to avoid known failure pattern"""
        error_type = failure_pattern.outcomes.get("error_type", "unknown")
        
        strategies = {
            "timeout": {
                "increase_timeout": True,
                "retry_with_backoff": True
            },
            "element_not_found": {
                "use_alternative_selectors": True,
                "wait_for_element": True,
                "scroll_to_element": True
            },
            "rate_limiting": {
                "add_delays": True,
                "reduce_concurrency": True
            }
        }
        
        return strategies.get(error_type, {"general_retry": True})
    
    async def _identify_optimization_patterns(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify optimization opportunities from session"""
        optimizations = []
        
        action_history = session_data.get("action_history", [])
        
        # Look for repeated actions
        action_counts = Counter(action.get("type", "unknown") for action in action_history)
        repeated_actions = {k: v for k, v in action_counts.items() if v > 2}
        
        if repeated_actions:
            optimizations.append({
                "type": "action_consolidation",
                "description": "Repeated actions detected - consider batching",
                "repeated_actions": repeated_actions,
                "potential_savings": len(action_history) * 0.2
            })
        
        # Look for slow actions
        slow_actions = [a for a in action_history if a.get("duration", 0) > 5.0]
        if slow_actions:
            optimizations.append({
                "type": "performance_optimization",
                "description": "Slow actions detected - consider optimization",
                "slow_actions": len(slow_actions),
                "potential_time_savings": sum(a.get("duration", 0) for a in slow_actions) * 0.3
            })
        
        return optimizations
    
    async def _update_predictive_models(self, session_data: Dict[str, Any]):
        """Update predictive models with new session data"""
        # Placeholder for ML model updates
        # In practice, this would retrain models periodically
        pass
    
    async def _persist_knowledge(self):
        """Persist learned knowledge to disk"""
        try:
            knowledge_data = {
                "patterns": {k: self._serialize_pattern(v) for k, v in self.patterns.items()},
                "site_knowledge": self._serialize_site_knowledge(),
                "goal_templates": self.goal_templates,
                "last_updated": datetime.now()
            }
            
            with open(self.knowledge_file, 'wb') as f:
                pickle.dump(knowledge_data, f)
                
            logger.info(f"Persisted knowledge: {len(self.patterns)} patterns, {len(self.site_knowledge)} sites")
        except Exception as e:
            logger.error(f"Failed to persist knowledge: {e}")
    
    def load_existing_knowledge(self):
        """Load existing knowledge from disk"""
        try:
            with open(self.knowledge_file, 'rb') as f:
                knowledge_data = pickle.load(f)
                
            self.patterns = {k: self._deserialize_pattern(v) for k, v in knowledge_data.get("patterns", {}).items()}
            self.site_knowledge = self._deserialize_site_knowledge(knowledge_data.get("site_knowledge", {}))
            self.goal_templates = knowledge_data.get("goal_templates", {})
            
            logger.info(f"Loaded knowledge: {len(self.patterns)} patterns, {len(self.site_knowledge)} sites")
        except FileNotFoundError:
            logger.info("No existing knowledge file found - starting fresh")
        except Exception as e:
            logger.error(f"Failed to load knowledge: {e}")
    
    def _serialize_pattern(self, pattern: AutomationPattern) -> Dict[str, Any]:
        """Serialize pattern for persistence"""
        return {
            "pattern_id": pattern.pattern_id,
            "pattern_type": pattern.pattern_type.value,
            "confidence_score": pattern.confidence_score,
            "frequency": pattern.frequency,
            "success_rate": pattern.success_rate,
            "context": pattern.context,
            "actions": pattern.actions,
            "conditions": pattern.conditions,
            "outcomes": pattern.outcomes,
            "created_at": pattern.created_at.isoformat(),
            "last_seen": pattern.last_seen.isoformat(),
            "sites_applicable": list(pattern.sites_applicable),
            "goals_applicable": list(pattern.goals_applicable)
        }
    
    def _deserialize_pattern(self, data: Dict[str, Any]) -> AutomationPattern:
        """Deserialize pattern from persistence"""
        return AutomationPattern(
            pattern_id=data["pattern_id"],
            pattern_type=PatternType(data["pattern_type"]),
            confidence_score=data["confidence_score"],
            frequency=data["frequency"],
            success_rate=data["success_rate"],
            context=data["context"],
            actions=data["actions"],
            conditions=data["conditions"],
            outcomes=data["outcomes"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_seen=datetime.fromisoformat(data["last_seen"]),
            sites_applicable=set(data["sites_applicable"]),
            goals_applicable=set(data["goals_applicable"])
        )
    
    def _serialize_site_knowledge(self) -> Dict[str, Any]:
        """Serialize site knowledge for persistence"""
        serialized = {}
        
        for domain, data in self.site_knowledge.items():
            serialized[domain] = {
                **data,
                "first_seen": data["first_seen"].isoformat() if isinstance(data["first_seen"], datetime) else data["first_seen"],
                "last_updated": data.get("last_updated", datetime.now()).isoformat(),
                "success_indicators": list(data.get("success_indicators", set()))
            }
        
        return serialized
    
    def _deserialize_site_knowledge(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize site knowledge from persistence"""
        deserialized = {}
        
        for domain, site_data in data.items():
            deserialized[domain] = {
                **site_data,
                "first_seen": datetime.fromisoformat(site_data["first_seen"]),
                "last_updated": datetime.fromisoformat(site_data.get("last_updated", datetime.now().isoformat())),
                "success_indicators": set(site_data.get("success_indicators", []))
            }
        
        return deserialized