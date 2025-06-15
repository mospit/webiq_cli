import asyncio
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging
import hashlib
import statistics
from urllib.parse import urlparse

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Result of automation prediction"""
    success_probability: float
    estimated_duration: float
    estimated_cost: float
    confidence_score: float
    risk_factors: List[str]
    optimization_suggestions: List[str]
    prediction_timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class FeatureSet:
    """Feature set for ML predictions"""
    goal_features: Dict[str, float]
    site_features: Dict[str, float]
    context_features: Dict[str, float]
    historical_features: Dict[str, float]
    temporal_features: Dict[str, float]

class PredictiveAnalyticsEngine:
    """Advanced predictive analytics for automation success, duration, and cost"""
    
    def __init__(self, cache_size: int = 1000):
        # ML Models
        self.success_predictor = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.duration_predictor = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.1,
            min_samples_split=5,
            random_state=42
        )
        
        self.cost_predictor = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.15,
            min_samples_split=4,
            random_state=42
        )
        
        # Feature processing
        self.feature_scaler = StandardScaler()
        self.goal_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.site_encoder = LabelEncoder()
        
        # Model state
        self.models_trained = False
        self.last_training_time = None
        self.training_data_count = 0
        
        # Feature engineering
        self.feature_importance = {}
        self.feature_names = []
        
        # Caching
        self.prediction_cache = {}
        self.cache_size = cache_size
        
        # Historical data
        self.training_data: List[Dict[str, Any]] = []
        self.prediction_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.model_performance = {
            "success_accuracy": 0.0,
            "duration_r2": 0.0,
            "cost_r2": 0.0,
            "last_evaluation": None
        }
        
        # Knowledge persistence
        self.models_file = "predictive_models.pkl"
        self.data_file = "training_data.pkl"
        
        # Load existing models and data
        self.load_existing_models()
        self.load_training_data()
    
    async def train_models(self, training_data: List[Dict[str, Any]], retrain: bool = False) -> Dict[str, Any]:
        """Train predictive models on historical automation data"""
        
        if not training_data:
            logger.warning("No training data provided")
            return {"success": False, "reason": "No training data"}
        
        # Add new training data
        if retrain:
            self.training_data = training_data
        else:
            self.training_data.extend(training_data)
        
        # Remove duplicates
        seen_sessions = set()
        unique_data = []
        for record in self.training_data:
            session_id = record.get("session_id", "")
            if session_id not in seen_sessions:
                seen_sessions.add(session_id)
                unique_data.append(record)
        
        self.training_data = unique_data
        self.training_data_count = len(self.training_data)
        
        if self.training_data_count < 50:
            logger.warning(f"Insufficient training data: {self.training_data_count} samples (minimum 50 required)")
            return {"success": False, "reason": "Insufficient training data"}
        
        try:
            # Prepare features and targets
            features, success_targets, duration_targets, cost_targets = await self._prepare_training_data(self.training_data)
            
            if len(features) == 0:
                return {"success": False, "reason": "No valid features extracted"}
            
            # Split data
            X_train, X_test, y_success_train, y_success_test = train_test_split(
                features, success_targets, test_size=0.2, random_state=42, stratify=success_targets
            )
            
            _, _, y_duration_train, y_duration_test = train_test_split(
                features, duration_targets, test_size=0.2, random_state=42
            )
            
            _, _, y_cost_train, y_cost_test = train_test_split(
                features, cost_targets, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            training_results = {}
            
            # Train success predictor
            logger.info("Training success predictor...")
            self.success_predictor.fit(X_train_scaled, y_success_train)
            success_pred = self.success_predictor.predict(X_test_scaled)
            success_accuracy = accuracy_score(y_success_test, success_pred)
            training_results["success_accuracy"] = success_accuracy
            
            # Train duration predictor
            logger.info("Training duration predictor...")
            # Filter out failed sessions for duration prediction
            duration_mask = np.array(y_success_train) == 1
            if np.sum(duration_mask) > 10:  # Need at least 10 successful sessions
                X_duration_train = X_train_scaled[duration_mask]
                y_duration_train_filtered = np.array(y_duration_train)[duration_mask]
                
                self.duration_predictor.fit(X_duration_train, y_duration_train_filtered)
                
                # Test on successful sessions only
                duration_test_mask = np.array(y_success_test) == 1
                if np.sum(duration_test_mask) > 0:
                    X_duration_test = X_test_scaled[duration_test_mask]
                    y_duration_test_filtered = np.array(y_duration_test)[duration_test_mask]
                    
                    duration_pred = self.duration_predictor.predict(X_duration_test)
                    duration_r2 = r2_score(y_duration_test_filtered, duration_pred)
                    training_results["duration_r2"] = duration_r2
                else:
                    training_results["duration_r2"] = 0.0
            else:
                training_results["duration_r2"] = 0.0
                logger.warning("Insufficient successful sessions for duration prediction")
            
            # Train cost predictor
            logger.info("Training cost predictor...")
            # Use all sessions for cost prediction (failed sessions also have costs)
            self.cost_predictor.fit(X_train_scaled, y_cost_train)
            cost_pred = self.cost_predictor.predict(X_test_scaled)
            cost_r2 = r2_score(y_cost_test, cost_pred)
            training_results["cost_r2"] = cost_r2
            
            # Update model performance
            self.model_performance.update({
                "success_accuracy": success_accuracy,
                "duration_r2": training_results["duration_r2"],
                "cost_r2": cost_r2,
                "last_evaluation": datetime.now()
            })
            
            # Extract feature importance
            self._extract_feature_importance()
            
            self.models_trained = True
            self.last_training_time = datetime.now()
            
            # Save models
            await self._save_models()
            await self._save_training_data()
            
            logger.info(f"Models trained successfully on {self.training_data_count} samples")
            logger.info(f"Success accuracy: {success_accuracy:.3f}, Duration R²: {training_results['duration_r2']:.3f}, Cost R²: {cost_r2:.3f}")
            
            return {
                "success": True,
                "training_samples": self.training_data_count,
                "performance": training_results,
                "feature_count": len(self.feature_names)
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {"success": False, "reason": str(e)}
    
    async def predict_automation_success(self, goal: str, url: str, context: Dict[str, Any]) -> PredictionResult:
        """Predict automation success, duration, and cost"""
        
        # Check cache first
        cache_key = self._generate_cache_key(goal, url, context)
        if cache_key in self.prediction_cache:
            cached_result = self.prediction_cache[cache_key]
            # Check if cache is still valid (within 1 hour)
            if datetime.now() - cached_result.prediction_timestamp < timedelta(hours=1):
                return cached_result
        
        if not self.models_trained:
            # Return default prediction if models not trained
            return PredictionResult(
                success_probability=0.7,  # Default moderate confidence
                estimated_duration=30.0,
                estimated_cost=1.0,
                confidence_score=0.3,
                risk_factors=["Models not trained - using defaults"],
                optimization_suggestions=["Collect more training data to improve predictions"]
            )
        
        try:
            # Extract features
            features = await self._extract_prediction_features(goal, url, context)
            features_scaled = self.feature_scaler.transform([features])
            
            # Make predictions
            success_prob = self.success_predictor.predict_proba(features_scaled)[0][1]  # Probability of success
            
            # Duration prediction (only meaningful if success is likely)
            if success_prob > 0.5:
                estimated_duration = max(5.0, self.duration_predictor.predict(features_scaled)[0])
            else:
                estimated_duration = 45.0  # Default for likely failures
            
            # Cost prediction
            estimated_cost = max(0.1, self.cost_predictor.predict(features_scaled)[0])
            
            # Calculate confidence based on model performance and feature quality
            confidence_score = self._calculate_prediction_confidence(features, success_prob)
            
            # Identify risk factors
            risk_factors = await self._identify_risk_factors(goal, url, context, features)
            
            # Generate optimization suggestions
            optimization_suggestions = await self._generate_optimization_suggestions(goal, url, context, success_prob, estimated_duration)
            
            result = PredictionResult(
                success_probability=success_prob,
                estimated_duration=estimated_duration,
                estimated_cost=estimated_cost,
                confidence_score=confidence_score,
                risk_factors=risk_factors,
                optimization_suggestions=optimization_suggestions
            )
            
            # Cache result
            self._cache_prediction(cache_key, result)
            
            # Record prediction for later evaluation
            self.prediction_history.append({
                "timestamp": datetime.now(),
                "goal": goal,
                "url": url,
                "context": context,
                "prediction": {
                    "success_probability": success_prob,
                    "estimated_duration": estimated_duration,
                    "estimated_cost": estimated_cost,
                    "confidence_score": confidence_score
                }
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return PredictionResult(
                success_probability=0.5,
                estimated_duration=30.0,
                estimated_cost=1.0,
                confidence_score=0.2,
                risk_factors=[f"Prediction error: {str(e)}"],
                optimization_suggestions=["Check system configuration"]
            )
    
    async def _prepare_training_data(self, training_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features and targets for model training"""
        
        features = []
        success_targets = []
        duration_targets = []
        cost_targets = []
        
        # First pass: collect all goals and sites for encoding
        all_goals = [record.get("goal", "") for record in training_data]
        all_sites = [self._extract_domain(record.get("url", "")) for record in training_data]
        
        # Fit encoders
        if all_goals:
            self.goal_vectorizer.fit(all_goals)
        if all_sites:
            unique_sites = list(set(all_sites))
            self.site_encoder.fit(unique_sites)
        
        # Second pass: extract features
        for record in training_data:
            try:
                feature_vector = await self._extract_training_features(record)
                if feature_vector is not None:
                    features.append(feature_vector)
                    success_targets.append(1 if record.get("success", False) else 0)
                    duration_targets.append(record.get("duration", 30.0))
                    cost_targets.append(record.get("cost", 1.0))
            except Exception as e:
                logger.warning(f"Failed to extract features from record: {e}")
                continue
        
        if features:
            self.feature_names = self._get_feature_names()
            return np.array(features), np.array(success_targets), np.array(duration_targets), np.array(cost_targets)
        else:
            return np.array([]), np.array([]), np.array([]), np.array([])
    
    async def _extract_training_features(self, record: Dict[str, Any]) -> Optional[List[float]]:
        """Extract feature vector from training record"""
        
        try:
            goal = record.get("goal", "")
            url = record.get("url", "")
            context = record.get("context", {})
            
            # Goal features
            goal_features = self._extract_goal_features(goal)
            
            # Site features
            site_features = self._extract_site_features(url)
            
            # Context features
            context_features = self._extract_context_features(context)
            
            # Historical features (if available)
            historical_features = self._extract_historical_features(url, goal)
            
            # Temporal features
            temporal_features = self._extract_temporal_features(record)
            
            # Session-specific features
            session_features = self._extract_session_features(record)
            
            # Combine all features
            all_features = (
                goal_features + 
                site_features + 
                context_features + 
                historical_features + 
                temporal_features + 
                session_features
            )
            
            return all_features
            
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return None
    
    async def _extract_prediction_features(self, goal: str, url: str, context: Dict[str, Any]) -> List[float]:
        """Extract features for prediction"""
        
        # Goal features
        goal_features = self._extract_goal_features(goal)
        
        # Site features
        site_features = self._extract_site_features(url)
        
        # Context features
        context_features = self._extract_context_features(context)
        
        # Historical features
        historical_features = self._extract_historical_features(url, goal)
        
        # Temporal features (current time)
        temporal_features = self._extract_current_temporal_features()
        
        # Default session features (since we don't have actual session data yet)
        session_features = self._extract_default_session_features()
        
        # Combine all features
        all_features = (
            goal_features + 
            site_features + 
            context_features + 
            historical_features + 
            temporal_features + 
            session_features
        )
        
        return all_features
    
    def _extract_goal_features(self, goal: str) -> List[float]:
        """Extract features from goal description"""
        
        features = []
        
        # Basic goal metrics
        features.append(len(goal) / 100.0)  # Goal length (normalized)
        features.append(len(goal.split()) / 20.0)  # Word count (normalized)
        
        # Goal type indicators
        goal_lower = goal.lower()
        features.append(1.0 if any(word in goal_lower for word in ["login", "sign in", "authenticate"]) else 0.0)
        features.append(1.0 if any(word in goal_lower for word in ["search", "find", "look for"]) else 0.0)
        features.append(1.0 if any(word in goal_lower for word in ["buy", "purchase", "order", "checkout"]) else 0.0)
        features.append(1.0 if any(word in goal_lower for word in ["form", "submit", "fill", "register"]) else 0.0)
        features.append(1.0 if any(word in goal_lower for word in ["navigate", "go to", "visit"]) else 0.0)
        features.append(1.0 if any(word in goal_lower for word in ["click", "press", "select"]) else 0.0)
        features.append(1.0 if any(word in goal_lower for word in ["download", "upload", "file"]) else 0.0)
        
        # Complexity indicators
        features.append(1.0 if "and" in goal_lower else 0.0)  # Multiple steps
        features.append(1.0 if "then" in goal_lower else 0.0)  # Sequential steps
        features.append(goal_lower.count("click") / 5.0)  # Number of clicks (normalized)
        
        # TF-IDF features (if vectorizer is fitted)
        try:
            if hasattr(self.goal_vectorizer, 'vocabulary_'):
                tfidf_features = self.goal_vectorizer.transform([goal]).toarray()[0]
                features.extend(tfidf_features.tolist())
            else:
                features.extend([0.0] * 100)  # Default TF-IDF size
        except:
            features.extend([0.0] * 100)
        
        return features
    
    def _extract_site_features(self, url: str) -> List[float]:
        """Extract features from site URL"""
        
        features = []
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
            
            # Domain features
            features.append(len(domain) / 50.0)  # Domain length
            features.append(domain.count('.') / 5.0)  # Subdomain count
            
            # Site type indicators
            features.append(1.0 if any(word in domain for word in ["shop", "store", "buy", "cart", "amazon", "ebay"]) else 0.0)
            features.append(1.0 if any(word in domain for word in ["social", "facebook", "twitter", "linkedin"]) else 0.0)
            features.append(1.0 if any(word in domain for word in ["news", "blog", "article"]) else 0.0)
            features.append(1.0 if any(word in domain for word in ["bank", "finance", "payment"]) else 0.0)
            features.append(1.0 if any(word in domain for word in ["gov", "edu", "org"]) else 0.0)
            
            # Path complexity
            features.append(len(path) / 100.0)  # Path length
            features.append(path.count('/') / 10.0)  # Path depth
            features.append(1.0 if "login" in path else 0.0)
            features.append(1.0 if "search" in path else 0.0)
            features.append(1.0 if "checkout" in path else 0.0)
            
            # Protocol and port
            features.append(1.0 if parsed.scheme == "https" else 0.0)
            features.append(1.0 if parsed.port is not None else 0.0)
            
            # Site encoding (if encoder is fitted)
            try:
                if hasattr(self.site_encoder, 'classes_'):
                    if domain in self.site_encoder.classes_:
                        encoded = self.site_encoder.transform([domain])[0]
                        features.append(encoded / len(self.site_encoder.classes_))
                    else:
                        features.append(0.5)  # Unknown site
                else:
                    features.append(0.5)
            except:
                features.append(0.5)
            
        except Exception as e:
            logger.warning(f"Site feature extraction failed: {e}")
            features = [0.0] * 15  # Default feature count
        
        return features
    
    def _extract_context_features(self, context: Dict[str, Any]) -> List[float]:
        """Extract features from execution context"""
        
        features = []
        
        # Basic context
        features.append(context.get("site_complexity", 5) / 10.0)
        features.append(1.0 if context.get("network_quality") == "excellent" else 0.5 if context.get("network_quality") == "good" else 0.0)
        features.append(1.0 if context.get("task_priority") == "critical" else 0.5 if context.get("task_priority") == "high" else 0.0)
        
        # Browser and environment
        features.append(1.0 if context.get("headless", True) else 0.0)
        features.append(context.get("timeout", 30) / 60.0)  # Normalized timeout
        features.append(1.0 if context.get("screenshots_enabled", False) else 0.0)
        
        # User preferences
        features.append(context.get("retry_attempts", 3) / 10.0)
        features.append(context.get("wait_time", 1.0) / 5.0)
        
        return features
    
    def _extract_historical_features(self, url: str, goal: str) -> List[float]:
        """Extract features based on historical performance"""
        
        features = []
        domain = self._extract_domain(url)
        
        # Site-specific historical performance
        site_sessions = [record for record in self.training_data if self._extract_domain(record.get("url", "")) == domain]
        
        if site_sessions:
            site_success_rate = sum(1 for s in site_sessions if s.get("success", False)) / len(site_sessions)
            site_avg_duration = statistics.mean([s.get("duration", 30) for s in site_sessions if s.get("success", False)]) if any(s.get("success", False) for s in site_sessions) else 30.0
            site_avg_cost = statistics.mean([s.get("cost", 1.0) for s in site_sessions])
            
            features.extend([site_success_rate, site_avg_duration / 60.0, site_avg_cost])
            features.append(len(site_sessions) / 100.0)  # Site familiarity
        else:
            features.extend([0.5, 0.5, 0.5, 0.0])  # Default values
        
        # Goal-specific historical performance
        goal_type = self._classify_goal_type(goal)
        goal_sessions = [record for record in self.training_data if self._classify_goal_type(record.get("goal", "")) == goal_type]
        
        if goal_sessions:
            goal_success_rate = sum(1 for s in goal_sessions if s.get("success", False)) / len(goal_sessions)
            goal_avg_duration = statistics.mean([s.get("duration", 30) for s in goal_sessions if s.get("success", False)]) if any(s.get("success", False) for s in goal_sessions) else 30.0
            
            features.extend([goal_success_rate, goal_avg_duration / 60.0])
            features.append(len(goal_sessions) / 100.0)  # Goal familiarity
        else:
            features.extend([0.5, 0.5, 0.0])  # Default values
        
        return features
    
    def _extract_temporal_features(self, record: Dict[str, Any]) -> List[float]:
        """Extract temporal features from training record"""
        
        features = []
        
        # Time of execution
        timestamp = record.get("timestamp")
        if timestamp:
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            features.append(timestamp.hour / 24.0)  # Hour of day
            features.append(timestamp.weekday() / 7.0)  # Day of week
            features.append(timestamp.month / 12.0)  # Month
        else:
            features.extend([0.5, 0.5, 0.5])  # Default values
        
        return features
    
    def _extract_current_temporal_features(self) -> List[float]:
        """Extract temporal features for current time"""
        
        now = datetime.now()
        return [
            now.hour / 24.0,
            now.weekday() / 7.0,
            now.month / 12.0
        ]
    
    def _extract_session_features(self, record: Dict[str, Any]) -> List[float]:
        """Extract session-specific features from training record"""
        
        features = []
        
        # Action count and complexity
        action_history = record.get("action_history", [])
        features.append(len(action_history) / 50.0)  # Number of actions
        
        # Error information
        errors = record.get("errors", [])
        features.append(len(errors) / 10.0)  # Error count
        
        # Performance metrics
        features.append(record.get("duration", 30) / 120.0)  # Duration (normalized)
        features.append(record.get("cost", 1.0) / 5.0)  # Cost (normalized)
        
        return features
    
    def _extract_default_session_features(self) -> List[float]:
        """Extract default session features for prediction"""
        
        return [0.2, 0.1, 0.25, 0.2]  # Conservative defaults
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names for interpretability"""
        
        names = []
        
        # Goal features
        names.extend([
            "goal_length", "goal_word_count", "is_login", "is_search", "is_ecommerce",
            "is_form", "is_navigation", "is_click", "is_file_operation", "has_multiple_steps",
            "has_sequence", "click_count"
        ])
        
        # TF-IDF features
        names.extend([f"tfidf_{i}" for i in range(100)])
        
        # Site features
        names.extend([
            "domain_length", "subdomain_count", "is_ecommerce_site", "is_social_media",
            "is_news_site", "is_financial", "is_institutional", "path_length", "path_depth",
            "has_login_path", "has_search_path", "has_checkout_path", "is_https",
            "has_custom_port", "site_encoding"
        ])
        
        # Context features
        names.extend([
            "site_complexity", "network_quality", "task_priority", "is_headless",
            "timeout_setting", "screenshots_enabled", "retry_attempts", "wait_time"
        ])
        
        # Historical features
        names.extend([
            "site_success_rate", "site_avg_duration", "site_avg_cost", "site_familiarity",
            "goal_success_rate", "goal_avg_duration", "goal_familiarity"
        ])
        
        # Temporal features
        names.extend(["hour_of_day", "day_of_week", "month"])
        
        # Session features
        names.extend(["action_count", "error_count", "duration", "cost"])
        
        return names
    
    def _calculate_prediction_confidence(self, features: List[float], success_prob: float) -> float:
        """Calculate confidence in prediction based on model performance and feature quality"""
        
        base_confidence = 0.5
        
        # Model performance contribution
        model_confidence = (
            self.model_performance["success_accuracy"] * 0.4 +
            max(0, self.model_performance["duration_r2"]) * 0.3 +
            max(0, self.model_performance["cost_r2"]) * 0.3
        )
        
        # Feature quality contribution
        feature_quality = min(1.0, self.training_data_count / 1000.0)  # More data = higher confidence
        
        # Prediction certainty (how far from 0.5 the success probability is)
        prediction_certainty = abs(success_prob - 0.5) * 2
        
        # Combined confidence
        confidence = (
            base_confidence * 0.2 +
            model_confidence * 0.4 +
            feature_quality * 0.2 +
            prediction_certainty * 0.2
        )
        
        return min(1.0, max(0.0, confidence))
    
    async def _identify_risk_factors(self, goal: str, url: str, context: Dict[str, Any], features: List[float]) -> List[str]:
        """Identify potential risk factors for the automation"""
        
        risk_factors = []
        
        # Site-specific risks
        domain = self._extract_domain(url)
        site_sessions = [record for record in self.training_data if self._extract_domain(record.get("url", "")) == domain]
        
        if site_sessions:
            site_success_rate = sum(1 for s in site_sessions if s.get("success", False)) / len(site_sessions)
            if site_success_rate < 0.6:
                risk_factors.append(f"Low historical success rate for {domain} ({site_success_rate:.1%})")
        
        # Goal complexity risks
        if len(goal.split()) > 15:
            risk_factors.append("Complex goal with many steps")
        
        if "and" in goal.lower() and "then" in goal.lower():
            risk_factors.append("Multi-step sequential goal")
        
        # Context risks
        if context.get("network_quality") == "poor":
            risk_factors.append("Poor network quality may cause timeouts")
        
        if context.get("site_complexity", 5) > 8:
            risk_factors.append("High site complexity detected")
        
        # Feature-based risks
        if len(features) > 0:
            # Check for unusual feature values that might indicate edge cases
            if features[0] > 0.8:  # Very long goal
                risk_factors.append("Unusually long goal description")
        
        return risk_factors
    
    async def _generate_optimization_suggestions(self, goal: str, url: str, context: Dict[str, Any], success_prob: float, estimated_duration: float) -> List[str]:
        """Generate optimization suggestions based on prediction"""
        
        suggestions = []
        
        # Success probability based suggestions
        if success_prob < 0.6:
            suggestions.append("Consider using conservative strategy with longer timeouts")
            suggestions.append("Enable detailed error logging for debugging")
        
        if success_prob > 0.9:
            suggestions.append("High success probability - consider aggressive strategy for speed")
        
        # Duration based suggestions
        if estimated_duration > 60:
            suggestions.append("Long duration expected - consider breaking into smaller tasks")
            suggestions.append("Enable progress monitoring for long-running automation")
        
        if estimated_duration < 10:
            suggestions.append("Quick task - consider batching with other automations")
        
        # Context based suggestions
        if context.get("network_quality") == "poor":
            suggestions.append("Increase timeout values due to poor network quality")
        
        if context.get("site_complexity", 5) > 7:
            suggestions.append("Use element waiting strategies for complex sites")
        
        # Goal based suggestions
        goal_lower = goal.lower()
        if "login" in goal_lower:
            suggestions.append("Consider credential caching for login operations")
        
        if "search" in goal_lower:
            suggestions.append("Implement search result validation")
        
        if "form" in goal_lower:
            suggestions.append("Use form validation to ensure data entry success")
        
        return suggestions
    
    def _extract_feature_importance(self):
        """Extract and store feature importance from trained models"""
        
        try:
            if hasattr(self.success_predictor, 'feature_importances_'):
                success_importance = self.success_predictor.feature_importances_
                
                if len(self.feature_names) == len(success_importance):
                    self.feature_importance = {
                        name: importance for name, importance in zip(self.feature_names, success_importance)
                    }
                    
                    # Log top features
                    top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                    logger.info(f"Top 10 important features: {top_features}")
        except Exception as e:
            logger.warning(f"Failed to extract feature importance: {e}")
    
    def _generate_cache_key(self, goal: str, url: str, context: Dict[str, Any]) -> str:
        """Generate cache key for prediction"""
        
        key_data = {
            "goal": goal,
            "domain": self._extract_domain(url),
            "complexity": context.get("site_complexity", 5),
            "network": context.get("network_quality", "good")
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _cache_prediction(self, cache_key: str, result: PredictionResult):
        """Cache prediction result"""
        
        # Implement LRU cache behavior
        if len(self.prediction_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = min(self.prediction_cache.keys(), 
                           key=lambda k: self.prediction_cache[k].prediction_timestamp)
            del self.prediction_cache[oldest_key]
        
        self.prediction_cache[cache_key] = result
    
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
    
    async def _save_models(self):
        """Save trained models to disk"""
        try:
            models_data = {
                "success_predictor": self.success_predictor,
                "duration_predictor": self.duration_predictor,
                "cost_predictor": self.cost_predictor,
                "feature_scaler": self.feature_scaler,
                "goal_vectorizer": self.goal_vectorizer,
                "site_encoder": self.site_encoder,
                "feature_names": self.feature_names,
                "feature_importance": self.feature_importance,
                "model_performance": self.model_performance,
                "models_trained": self.models_trained,
                "last_training_time": self.last_training_time,
                "training_data_count": self.training_data_count
            }
            
            joblib.dump(models_data, self.models_file)
            logger.info("Models saved successfully")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    def load_existing_models(self):
        """Load existing models from disk"""
        try:
            models_data = joblib.load(self.models_file)
            
            self.success_predictor = models_data["success_predictor"]
            self.duration_predictor = models_data["duration_predictor"]
            self.cost_predictor = models_data["cost_predictor"]
            self.feature_scaler = models_data["feature_scaler"]
            self.goal_vectorizer = models_data["goal_vectorizer"]
            self.site_encoder = models_data["site_encoder"]
            self.feature_names = models_data.get("feature_names", [])
            self.feature_importance = models_data.get("feature_importance", {})
            self.model_performance = models_data.get("model_performance", {})
            self.models_trained = models_data.get("models_trained", False)
            self.last_training_time = models_data.get("last_training_time")
            self.training_data_count = models_data.get("training_data_count", 0)
            
            logger.info(f"Loaded models trained on {self.training_data_count} samples")
        except FileNotFoundError:
            logger.info("No existing models found - starting fresh")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
    
    async def _save_training_data(self):
        """Save training data to disk"""
        try:
            with open(self.data_file, 'wb') as f:
                pickle.dump({
                    "training_data": self.training_data,
                    "prediction_history": self.prediction_history[-1000:]  # Keep last 1000
                }, f)
            logger.info(f"Saved {len(self.training_data)} training samples")
        except Exception as e:
            logger.error(f"Failed to save training data: {e}")
    
    def load_training_data(self):
        """Load training data from disk"""
        try:
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
            
            self.training_data = data.get("training_data", [])
            self.prediction_history = data.get("prediction_history", [])
            self.training_data_count = len(self.training_data)
            
            logger.info(f"Loaded {self.training_data_count} training samples")
        except FileNotFoundError:
            logger.info("No existing training data found - starting fresh")
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
    
    async def evaluate_predictions(self, actual_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate prediction accuracy against actual results"""
        
        if not actual_results:
            return {"error": "No actual results provided"}
        
        try:
            # Match predictions with actual results
            matched_predictions = []
            
            for result in actual_results:
                # Find corresponding prediction
                goal = result.get("goal", "")
                url = result.get("url", "")
                
                # Find prediction in history
                for pred_record in reversed(self.prediction_history):
                    if (pred_record["goal"] == goal and 
                        self._extract_domain(pred_record["url"]) == self._extract_domain(url)):
                        
                        matched_predictions.append({
                            "predicted": pred_record["prediction"],
                            "actual": result
                        })
                        break
            
            if not matched_predictions:
                return {"error": "No matching predictions found"}
            
            # Calculate evaluation metrics
            success_predictions = []
            success_actuals = []
            duration_predictions = []
            duration_actuals = []
            
            for match in matched_predictions:
                pred = match["predicted"]
                actual = match["actual"]
                
                success_predictions.append(1 if pred["success_probability"] > 0.5 else 0)
                success_actuals.append(1 if actual.get("success", False) else 0)
                
                if actual.get("success", False):
                    duration_predictions.append(pred["estimated_duration"])
                    duration_actuals.append(actual.get("duration", 30.0))
            
            # Calculate metrics
            evaluation = {}
            
            if success_predictions:
                success_accuracy = accuracy_score(success_actuals, success_predictions)
                evaluation["success_accuracy"] = success_accuracy
            
            if duration_predictions and len(duration_predictions) > 1:
                duration_mse = mean_squared_error(duration_actuals, duration_predictions)
                duration_r2 = r2_score(duration_actuals, duration_predictions)
                evaluation["duration_mse"] = duration_mse
                evaluation["duration_r2"] = duration_r2
            
            evaluation["evaluated_samples"] = len(matched_predictions)
            
            # Update model performance
            self.model_performance.update(evaluation)
            self.model_performance["last_evaluation"] = datetime.now()
            
            logger.info(f"Evaluated {len(matched_predictions)} predictions: {evaluation}")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Prediction evaluation failed: {e}")
            return {"error": str(e)}
    
    async def get_model_insights(self) -> Dict[str, Any]:
        """Get insights about model performance and feature importance"""
        
        insights = {
            "model_status": {
                "trained": self.models_trained,
                "training_samples": self.training_data_count,
                "last_training": self.last_training_time.isoformat() if self.last_training_time else None
            },
            "performance": self.model_performance,
            "feature_importance": dict(sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]) if self.feature_importance else {},
            "prediction_cache_size": len(self.prediction_cache),
            "recent_predictions": len(self.prediction_history)
        }
        
        return insights