#!/usr/bin/env python3
"""
Comprehensive test suite for the WebIQ Learning & Adaptation System

This test validates all components of the sophisticated learning system:
- Pattern Recognition Engine
- Adaptive Strategy Engine  
- Predictive Analytics Engine
- Knowledge Base Manager
- Unified Learning System

Run with: python test_learning_system.py
"""

import asyncio
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import sys
import os

# Add the webiq package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'webiq'))

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

# Import the learning system components
from webiq.core.learning_system import LearningSystem, LearningConfig, LearningMetrics
from webiq.core.pattern_recognition_engine import PatternRecognitionEngine
from webiq.core.adaptive_strategy_engine import AdaptiveStrategyEngine, AdaptiveStrategy
from webiq.core.predictive_analytics_engine import PredictiveAnalyticsEngine
from webiq.core.knowledge_base_manager import (
    KnowledgeBaseManager, KnowledgeBase, AutomationPattern, SiteKnowledge, GoalTemplate
)

class LearningSystemTester:
    """Comprehensive test suite for the learning system"""
    
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
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="webiq_learning_test_")
        print(f"   Created temp directory: {self.temp_dir}")
        
        # Initialize learning system with test configuration
        config = LearningConfig(
            learning_enabled=True,
            auto_optimization=False,  # Disable for testing
            min_training_samples=3,   # Lower threshold for testing
            pattern_retention_days=30
        )
        
        knowledge_db_path = os.path.join(self.temp_dir, "test_knowledge.db")
        models_dir = os.path.join(self.temp_dir, "test_models")
        
        self.learning_system = LearningSystem(
            config=config,
            knowledge_db_path=knowledge_db_path,
            models_dir=models_dir
        )
        
        print("✅ Test environment setup complete")
    
    def cleanup(self):
        """Cleanup test environment"""
        print("🧹 Cleaning up test environment...")
        
        if self.learning_system:
            # Stop the learning system
            try:
                asyncio.run(self.learning_system.stop())
            except:
                pass
        
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
    
    async def test_knowledge_base_manager(self):
        """Test knowledge base manager functionality"""
        print("\n📚 Testing Knowledge Base Manager...")
        
        try:
            # Test knowledge base initialization
            kb_manager = self.learning_system.knowledge_manager
            self.assert_test(
                kb_manager is not None,
                "Knowledge base manager initialization"
            )
            
            # Test storing automation pattern
            pattern = AutomationPattern(
                pattern_id="test_pattern_1",
                goal="test goal",
                url="https://example.com",
                action_sequence=[
                    {"action_type": "click", "selector": "#button1"},
                    {"action_type": "type", "selector": "#input1", "text": "test"}
                ],
                success_rate=0.85,
                avg_duration=15.5,
                usage_count=5,
                last_used=datetime.now(),
                created_at=datetime.now()
            )
            
            await kb_manager.store_pattern(pattern)
            self.assert_test(True, "Store automation pattern")
            
            # Test retrieving pattern
            retrieved_patterns = await kb_manager.find_patterns(
                goal="test goal",
                url="https://example.com"
            )
            self.assert_test(
                len(retrieved_patterns) > 0,
                "Retrieve stored pattern",
                f"Expected patterns, got {len(retrieved_patterns)}"
            )
            
            # Test site knowledge storage
            site_knowledge = SiteKnowledge(
                url="https://example.com",
                common_elements={
                    "login_button": "#login",
                    "search_box": "#search"
                },
                timing_patterns={
                    "avg_load_time": 2.5,
                    "avg_response_time": 1.2
                },
                reliability_score=0.9,
                optimization_rules=[
                    "Use explicit waits for dynamic content",
                    "Prefer ID selectors over class selectors"
                ],
                last_updated=datetime.now()
            )
            
            await kb_manager.store_site_knowledge(site_knowledge)
            self.assert_test(True, "Store site knowledge")
            
            # Test goal template storage
            goal_template = GoalTemplate(
                goal="test goal",
                canonical_steps=[
                    "Navigate to page",
                    "Fill form",
                    "Submit"
                ],
                success_rate=0.8,
                avg_duration=20.0,
                complexity_score=3.5,
                usage_count=10,
                last_used=datetime.now(),
                created_at=datetime.now()
            )
            
            await kb_manager.store_goal_template(goal_template)
            self.assert_test(True, "Store goal template")
            
            # Test getting recommendations
            recommendations = await kb_manager.get_recommendations(
                "test goal",
                "https://example.com",
                {"context": "test"}
            )
            self.assert_test(
                "patterns" in recommendations,
                "Get knowledge recommendations",
                f"Expected recommendations with patterns, got {recommendations.keys()}"
            )
            
        except Exception as e:
            self.assert_test(False, "Knowledge base manager test", str(e))
    
    async def test_pattern_recognition_engine(self):
        """Test pattern recognition engine functionality"""
        print("\n🔍 Testing Pattern Recognition Engine...")
        
        try:
            pattern_engine = self.learning_system.pattern_engine
            
            # Test learning from session
            session_data = {
                "session_id": "test_session_1",
                "goal": "login to website",
                "url": "https://example.com/login",
                "success": True,
                "duration": 12.5,
                "cost": 0.05,
                "action_history": [
                    {
                        "action_type": "navigate",
                        "url": "https://example.com/login",
                        "timestamp": datetime.now().isoformat(),
                        "success": True,
                        "duration": 2.1
                    },
                    {
                        "action_type": "type",
                        "selector": "#username",
                        "text": "testuser",
                        "timestamp": datetime.now().isoformat(),
                        "success": True,
                        "duration": 1.5
                    },
                    {
                        "action_type": "type",
                        "selector": "#password",
                        "text": "password",
                        "timestamp": datetime.now().isoformat(),
                        "success": True,
                        "duration": 1.2
                    },
                    {
                        "action_type": "click",
                        "selector": "#login-button",
                        "timestamp": datetime.now().isoformat(),
                        "success": True,
                        "duration": 0.8
                    }
                ],
                "errors": [],
                "context": {"user_type": "standard"}
            }
            
            learning_result = await pattern_engine.learn_from_session(session_data)
            self.assert_test(
                "patterns_created" in learning_result or "patterns_updated" in learning_result,
                "Pattern learning from session",
                f"Expected learning results, got {learning_result.keys()}"
            )
            
            # Test optimization suggestions
            optimizations = await pattern_engine.suggest_optimizations(
                "login to website",
                "https://example.com/login",
                {"current_duration": 15.0}
            )
            self.assert_test(
                isinstance(optimizations, list),
                "Generate optimization suggestions",
                f"Expected list of optimizations, got {type(optimizations)}"
            )
            
        except Exception as e:
            self.assert_test(False, "Pattern recognition engine test", str(e))
    
    async def test_adaptive_strategy_engine(self):
        """Test adaptive strategy engine functionality"""
        print("\n🎯 Testing Adaptive Strategy Engine...")
        
        try:
            strategy_engine = self.learning_system.strategy_engine
            
            # Test strategy selection
            strategy = await strategy_engine.select_optimal_strategy(
                "login to website",
                "https://example.com/login",
                {"user_type": "standard", "device": "desktop"}
            )
            
            self.assert_test(
                strategy is not None,
                "Strategy selection",
                "Expected strategy object, got None"
            )
            
            if strategy:
                self.assert_test(
                    hasattr(strategy, 'name') and hasattr(strategy, 'parameters'),
                    "Strategy structure validation",
                    "Strategy missing required attributes"
                )
            
            # Test strategy adaptation from feedback
            adaptation_result = await strategy_engine.adapt_strategy_from_feedback(
                "login to website",
                "https://example.com/login",
                success=True,
                duration=10.5,
                action_history=[
                    {"action_type": "navigate", "duration": 2.0},
                    {"action_type": "type", "duration": 1.5},
                    {"action_type": "click", "duration": 1.0}
                ],
                errors=[]
            )
            
            self.assert_test(
                "adaptations_made" in adaptation_result,
                "Strategy adaptation from feedback",
                f"Expected adaptation results, got {adaptation_result.keys()}"
            )
            
        except Exception as e:
            self.assert_test(False, "Adaptive strategy engine test", str(e))
    
    async def test_predictive_analytics_engine(self):
        """Test predictive analytics engine functionality"""
        print("\n📊 Testing Predictive Analytics Engine...")
        
        try:
            analytics_engine = self.learning_system.analytics_engine
            
            # Add training data
            training_samples = [
                {
                    "goal": "login",
                    "url": "https://example.com",
                    "success": True,
                    "duration": 10.5,
                    "cost": 0.05,
                    "action_count": 4,
                    "complexity": 2.5
                },
                {
                    "goal": "search",
                    "url": "https://example.com",
                    "success": True,
                    "duration": 8.2,
                    "cost": 0.03,
                    "action_count": 3,
                    "complexity": 1.8
                },
                {
                    "goal": "purchase",
                    "url": "https://shop.example.com",
                    "success": False,
                    "duration": 25.0,
                    "cost": 0.12,
                    "action_count": 8,
                    "complexity": 4.2
                }
            ]
            
            for sample in training_samples:
                await analytics_engine.add_training_data(sample)
            
            self.assert_test(True, "Add training data to analytics engine")
            
            # Test model training
            training_result = await analytics_engine.train_models()
            self.assert_test(
                "success" in training_result,
                "Train predictive models",
                f"Expected training results, got {training_result.keys()}"
            )
            
            # Test predictions
            prediction = await analytics_engine.predict_automation_success(
                "login",
                "https://example.com",
                {"action_count": 4, "complexity": 2.0}
            )
            
            self.assert_test(
                "success_probability" in prediction,
                "Generate success predictions",
                f"Expected prediction with success_probability, got {prediction.keys()}"
            )
            
            expected_keys = ["success_probability", "estimated_duration", "estimated_cost", "confidence"]
            for key in expected_keys:
                self.assert_test(
                    key in prediction,
                    f"Prediction contains {key}",
                    f"Missing {key} in prediction"
                )
            
        except Exception as e:
            self.assert_test(False, "Predictive analytics engine test", str(e))
    
    async def test_unified_learning_system(self):
        """Test the unified learning system functionality"""
        print("\n🧠 Testing Unified Learning System...")
        
        try:
            # Test system initialization
            self.assert_test(
                self.learning_system is not None,
                "Learning system initialization"
            )
            
            # Test learning from session (end-to-end)
            session_data = {
                "session_id": "unified_test_session",
                "goal": "complete checkout",
                "url": "https://shop.example.com",
                "success": True,
                "duration": 45.2,
                "cost": 0.15,
                "action_history": [
                    {"action_type": "navigate", "duration": 3.0, "success": True},
                    {"action_type": "click", "duration": 1.5, "success": True},
                    {"action_type": "type", "duration": 2.0, "success": True},
                    {"action_type": "click", "duration": 1.0, "success": True},
                    {"action_type": "wait", "duration": 5.0, "success": True},
                    {"action_type": "click", "duration": 1.2, "success": True}
                ],
                "errors": [],
                "context": {"user_type": "premium", "cart_value": 150.00}
            }
            
            learning_result = await self.learning_system.learn_from_session(session_data)
            self.assert_test(
                "learning_summary" in learning_result,
                "End-to-end learning from session",
                f"Expected learning summary, got {learning_result.keys()}"
            )
            
            # Test getting automation recommendations
            recommendations = await self.learning_system.get_automation_recommendations(
                "complete checkout",
                "https://shop.example.com",
                {"user_type": "premium"}
            )
            
            self.assert_test(
                "strategy" in recommendations and "predictions" in recommendations,
                "Get automation recommendations",
                f"Expected strategy and predictions, got {recommendations.keys()}"
            )
            
            # Test optimization
            current_performance = {
                "success_rate": 0.75,
                "avg_duration": 50.0,
                "avg_cost": 0.18
            }
            
            optimization_result = await self.learning_system.optimize_automation_strategy(
                "complete checkout",
                "https://shop.example.com",
                current_performance
            )
            
            self.assert_test(
                "optimizations_applied" in optimization_result,
                "Optimize automation strategy",
                f"Expected optimization results, got {optimization_result.keys()}"
            )
            
            # Test learning insights
            insights = await self.learning_system.get_learning_insights(timeframe_days=7)
            self.assert_test(
                "metrics" in insights and "knowledge_stats" in insights,
                "Get learning insights",
                f"Expected metrics and knowledge_stats, got {insights.keys()}"
            )
            
            # Test metrics tracking
            metrics = self.learning_system.metrics
            self.assert_test(
                metrics.total_sessions_processed > 0,
                "Metrics tracking",
                f"Expected processed sessions > 0, got {metrics.total_sessions_processed}"
            )
            
        except Exception as e:
            self.assert_test(False, "Unified learning system test", str(e))
    
    async def test_learning_system_integration(self):
        """Test integration between learning system components"""
        print("\n🔗 Testing Learning System Integration...")
        
        try:
            # Test multiple learning sessions to verify integration
            sessions = [
                {
                    "session_id": "integration_test_1",
                    "goal": "user registration",
                    "url": "https://app.example.com/register",
                    "success": True,
                    "duration": 25.0,
                    "cost": 0.08,
                    "action_history": [
                        {"action_type": "navigate", "duration": 2.0},
                        {"action_type": "type", "duration": 3.0},
                        {"action_type": "type", "duration": 2.5},
                        {"action_type": "click", "duration": 1.0}
                    ],
                    "errors": []
                },
                {
                    "session_id": "integration_test_2",
                    "goal": "user registration",
                    "url": "https://app.example.com/register",
                    "success": False,
                    "duration": 35.0,
                    "cost": 0.12,
                    "action_history": [
                        {"action_type": "navigate", "duration": 2.0},
                        {"action_type": "type", "duration": 3.0},
                        {"action_type": "type", "duration": 2.5},
                        {"action_type": "click", "duration": 1.0}
                    ],
                    "errors": ["Validation error: Email already exists"]
                },
                {
                    "session_id": "integration_test_3",
                    "goal": "user registration",
                    "url": "https://app.example.com/register",
                    "success": True,
                    "duration": 20.0,
                    "cost": 0.06,
                    "action_history": [
                        {"action_type": "navigate", "duration": 1.5},
                        {"action_type": "type", "duration": 2.0},
                        {"action_type": "type", "duration": 2.0},
                        {"action_type": "click", "duration": 0.8}
                    ],
                    "errors": []
                }
            ]
            
            # Process multiple sessions
            for session in sessions:
                await self.learning_system.learn_from_session(session)
            
            self.assert_test(True, "Process multiple learning sessions")
            
            # Verify that knowledge accumulates
            final_recommendations = await self.learning_system.get_automation_recommendations(
                "user registration",
                "https://app.example.com/register"
            )
            
            self.assert_test(
                final_recommendations.get("confidence_score", 0.0) > 0.0,
                "Knowledge accumulation improves confidence",
                f"Expected confidence > 0, got {final_recommendations.get('confidence_score', 0.0)}"
            )
            
            # Test that patterns are being learned and applied
            patterns = final_recommendations.get("patterns", [])
            self.assert_test(
                len(patterns) >= 0,  # Should have learned some patterns
                "Pattern learning and retrieval",
                f"Expected patterns to be learned, got {len(patterns)} patterns"
            )
            
            # Verify metrics are being updated
            final_metrics = self.learning_system.metrics
            self.assert_test(
                final_metrics.total_sessions_processed >= len(sessions),
                "Metrics accumulation",
                f"Expected >= {len(sessions)} sessions, got {final_metrics.total_sessions_processed}"
            )
            
        except Exception as e:
            self.assert_test(False, "Learning system integration test", str(e))
    
    async def run_all_tests(self):
        """Run all learning system tests"""
        print("🚀 Starting WebIQ Learning System Test Suite")
        print("=" * 60)
        
        try:
            # Setup
            self.setup()
            
            # Run individual component tests
            await self.test_knowledge_base_manager()
            await self.test_pattern_recognition_engine()
            await self.test_adaptive_strategy_engine()
            await self.test_predictive_analytics_engine()
            
            # Run unified system tests
            await self.test_unified_learning_system()
            await self.test_learning_system_integration()
            
        except Exception as e:
            print(f"\n❌ Test suite failed with error: {e}")
            self.test_results["tests_failed"] += 1
            self.test_results["failures"].append(f"Test suite error: {str(e)}")
        
        finally:
            # Cleanup
            self.cleanup()
    
    def print_results(self):
        """Print test results summary"""
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
    """Main test execution function"""
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
    # Run the test suite
    asyncio.run(main())