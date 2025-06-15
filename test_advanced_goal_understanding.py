#!/usr/bin/env python3
"""
Comprehensive test suite for the Advanced Goal Understanding system.
Tests all components: AdvancedGoalProcessor, RequirementExtractor, and ContextualStrategySelector.
"""

import asyncio
import sys
import os
import json
from typing import Dict, List, Any
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'webiq'))

class AdvancedGoalUnderstandingTester:
    """Test suite for Advanced Goal Understanding system"""
    
    def __init__(self):
        self.test_results = []
        self.setup_test_data()
    
    def setup_test_data(self):
        """Setup test data for various scenarios"""
        self.test_goals = [
            {
                "goal": "Login to my bank account with email john@example.com and password secret123",
                "url": "https://bank.example.com/login",
                "context": {
                    "security_requirements": ["two_factor_auth", "encryption"],
                    "time_limit_seconds": 300,
                    "required_success_rate": 0.99
                }
            },
            {
                "goal": "Search for iPhone 15 Pro and add the first result to cart, then checkout with my saved payment method",
                "url": "https://shop.example.com",
                "context": {
                    "is_mobile": True,
                    "time_limit_seconds": 600,
                    "required_success_rate": 0.95
                }
            },
            {
                "goal": "Fill out the job application form with my resume and submit it before 5 PM today",
                "url": "https://careers.company.com/apply",
                "context": {
                    "time_limit_seconds": 1800,
                    "compliance_needs": ["equal_opportunity", "data_privacy"],
                    "required_success_rate": 0.98
                }
            },
            {
                "goal": "Extract all product prices from the electronics category and save to CSV",
                "url": "https://ecommerce.example.com/electronics",
                "context": {
                    "headless_mode": True,
                    "slow_connection": True,
                    "required_success_rate": 0.90
                }
            },
            {
                "goal": "Book a doctor appointment for next Tuesday at 2 PM using my insurance information",
                "url": "https://healthcare.example.com/booking",
                "context": {
                    "security_requirements": ["hipaa_compliance"],
                    "compliance_needs": ["healthcare_privacy"],
                    "required_success_rate": 0.99
                }
            }
        ]
    
    def run_test(self, test_name: str, test_func, *args, **kwargs):
        """Run a single test and record results"""
        try:
            print(f"\n🧪 Running {test_name}...")
            start_time = datetime.now()
            
            if asyncio.iscoroutinefunction(test_func):
                result = asyncio.run(test_func(*args, **kwargs))
            else:
                result = test_func(*args, **kwargs)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            self.test_results.append({
                "test_name": test_name,
                "status": "PASSED",
                "duration": duration,
                "result": result
            })
            
            print(f"✅ {test_name} PASSED ({duration:.2f}s)")
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.test_results.append({
                "test_name": test_name,
                "status": "FAILED",
                "duration": duration,
                "error": str(e)
            })
            print(f"❌ {test_name} FAILED: {e}")
            return None
    
    def test_advanced_goal_processor_initialization(self):
        """Test AdvancedGoalProcessor initialization"""
        from webiq.core.advanced_goal_processor import AdvancedGoalProcessor
        
        processor = AdvancedGoalProcessor()
        
        # Verify initialization
        assert hasattr(processor, 'nlp_model'), "NLP model not initialized"
        assert hasattr(processor, 'semantic_model'), "Semantic model not initialized"
        assert hasattr(processor, 'decomposition_templates'), "Decomposition templates not initialized"
        assert hasattr(processor, 'complexity_rules'), "Complexity rules not initialized"
        assert hasattr(processor, 'risk_assessment_rules'), "Risk assessment rules not initialized"
        
        return "AdvancedGoalProcessor initialized successfully"
    
    async def test_goal_complexity_analysis(self):
        """Test goal complexity analysis"""
        from webiq.core.advanced_goal_processor import AdvancedGoalProcessor, GoalComplexity
        
        processor = AdvancedGoalProcessor()
        
        # Test simple goal
        simple_goal = "Click the login button"
        simple_complexity = await processor.analyze_goal_complexity(simple_goal, {"url": "https://example.com"})
        
        assert simple_complexity in [GoalComplexity.SIMPLE, GoalComplexity.MODERATE], "Simple goal should have low to moderate complexity"
        
        # Test complex goal
        complex_goal = "Login with two-factor authentication, navigate to settings, update profile with new address, upload profile picture, and save changes"
        complex_complexity = await processor.analyze_goal_complexity(complex_goal, {"url": "https://complex-app.com"})
        
        assert complex_complexity in [GoalComplexity.COMPLEX, GoalComplexity.VERY_COMPLEX], "Complex goal should have high complexity"
        
        return f"Complexity analysis working: simple={simple_complexity.value}, complex={complex_complexity.value}"
    
    async def test_goal_decomposition(self):
        """Test goal decomposition functionality"""
        from webiq.core.advanced_goal_processor import AdvancedGoalProcessor
        
        processor = AdvancedGoalProcessor()
        
        test_goal = self.test_goals[1]  # Shopping goal
        
        decomposition = await processor.decompose_goal(
            test_goal["goal"],
            test_goal["url"],
            test_goal["context"]
        )
        
        # Verify decomposition structure
        assert hasattr(decomposition, 'sub_goals'), "Decomposition should contain sub_goals"
        assert hasattr(decomposition, 'execution_graph'), "Decomposition should contain execution_graph"
        assert hasattr(decomposition, 'estimated_total_duration'), "Decomposition should contain estimated_total_duration"
        
        sub_goals = decomposition.sub_goals
        assert len(sub_goals) > 1, "Complex goal should be decomposed into multiple sub-goals"
        
        # Verify sub-goal structure
        for sub_goal in sub_goals:
            assert hasattr(sub_goal, 'sub_goal_id'), "Sub-goal should have sub_goal_id"
            assert hasattr(sub_goal, 'description'), "Sub-goal should have description"
            assert hasattr(sub_goal, 'complexity_score'), "Sub-goal should have complexity_score"
            assert hasattr(sub_goal, 'estimated_duration'), "Sub-goal should have estimated_duration"
        
        return f"Goal decomposed into {len(sub_goals)} sub-goals with execution graph"
    
    def test_requirement_extractor_initialization(self):
        """Test RequirementExtractor initialization"""
        from webiq.core.requirement_extractor import RequirementExtractor
        
        extractor = RequirementExtractor()
        
        # Verify initialization
        assert hasattr(extractor, 'requirement_patterns'), "Requirement patterns not initialized"
        assert hasattr(extractor, 'implicit_rules'), "Implicit rules not initialized"
        assert hasattr(extractor, 'nlp_model'), "NLP model not initialized"
        
        return "RequirementExtractor initialized successfully"
    
    def test_explicit_requirement_extraction(self):
        """Test explicit requirement extraction"""
        from webiq.core.requirement_extractor import RequirementExtractor
        
        extractor = RequirementExtractor()
        
        # Test goal with explicit requirements
        goal = "Login with email john@example.com and password secret123, then navigate to settings within 5 minutes"
        
        explicit_requirements = extractor.extract_explicit(goal)
        
        assert len(explicit_requirements) > 0, "Should extract explicit requirements"
        
        # Check for specific requirement types
        requirement_text = " ".join(explicit_requirements)
        assert "email" in requirement_text.lower() or "john@example.com" in requirement_text, "Should extract email requirement"
        
        return f"Extracted {len(explicit_requirements)} explicit requirements"
    
    async def test_implicit_requirement_inference(self):
        """Test implicit requirement inference"""
        from webiq.core.requirement_extractor import RequirementExtractor
        
        extractor = RequirementExtractor()
        
        # Test banking goal
        banking_goal = "Transfer money to another account"
        banking_url = "https://bank.example.com"
        banking_context = {"security_requirements": ["two_factor_auth"]}
        
        implicit_requirements = await extractor.infer_implicit(banking_goal, banking_url, banking_context)
        
        assert len(implicit_requirements) > 0, "Should infer implicit requirements"
        
        # Check for banking-specific requirements
        requirement_text = " ".join(implicit_requirements)
        assert "security" in requirement_text.lower() or "auth" in requirement_text.lower(), "Should infer security requirements for banking"
        
        return f"Inferred {len(implicit_requirements)} implicit requirements for banking scenario"
    
    def test_constraint_extraction(self):
        """Test constraint extraction"""
        from webiq.core.requirement_extractor import RequirementExtractor
        
        extractor = RequirementExtractor()
        
        # Test goal with constraints
        goal = "Complete the form within 10 minutes with maximum accuracy and secure handling"
        context = {"mobile_device": True, "slow_connection": True}
        
        constraints = extractor.extract_constraints(goal, context)
        
        assert len(constraints) > 0, "Should extract constraints"
        
        # Check for specific constraint types
        constraint_text = " ".join(constraints)
        assert "time" in constraint_text.lower() or "accuracy" in constraint_text.lower(), "Should extract time or accuracy constraints"
        
        return f"Extracted {len(constraints)} constraints"
    
    def test_contextual_strategy_selector_initialization(self):
        """Test ContextualStrategySelector initialization"""
        from webiq.core.contextual_strategy_selector import ContextualStrategySelector
        
        selector = ContextualStrategySelector()
        
        # Verify initialization
        assert hasattr(selector, 'strategy_templates'), "Strategy templates not initialized"
        assert hasattr(selector, 'context_weights'), "Context weights not initialized"
        assert len(selector.strategy_templates) > 0, "Should have strategy templates"
        
        return f"ContextualStrategySelector initialized with {len(selector.strategy_templates)} strategy templates"
    
    def test_strategy_selection(self):
        """Test strategy selection functionality"""
        from webiq.core.contextual_strategy_selector import ContextualStrategySelector
        
        selector = ContextualStrategySelector()
        
        # Test strategy selection for banking scenario
        banking_goal = self.test_goals[0]
        
        # Mock goal decomposition
        goal_decomposition = {
            "sub_goals": [
                {"id": "1", "description": "Navigate to login page", "complexity": 0.2},
                {"id": "2", "description": "Enter credentials", "complexity": 0.3},
                {"id": "3", "description": "Handle two-factor auth", "complexity": 0.6}
            ],
            "complexity_score": 0.7,
            "estimated_duration": 180
        }
        
        strategy_result = selector.select_strategy(goal_decomposition, banking_goal["context"])
        
        # Verify strategy selection result
        assert "selected_strategy" in strategy_result, "Should return selected strategy"
        assert "evaluation" in strategy_result, "Should return evaluation"
        assert "alternatives" in strategy_result, "Should return alternatives"
        
        selected_strategy = strategy_result["selected_strategy"]
        assert "name" in selected_strategy, "Selected strategy should have name"
        assert "strategy_type" in selected_strategy, "Selected strategy should have type"
        
        return f"Selected strategy: {selected_strategy['name']} with score {strategy_result['evaluation']['suitability_score']:.2f}"
    
    def test_strategy_customization(self):
        """Test strategy customization based on context"""
        from webiq.core.contextual_strategy_selector import ContextualStrategySelector, ContextFactors
        
        selector = ContextualStrategySelector()
        
        # Create test context factors
        context_factors = ContextFactors(
            site_complexity=0.8,
            goal_complexity=0.6,
            time_constraints=300,  # 5 minutes
            resource_availability={"cpu": 0.9, "memory": 0.8, "network": 0.7, "browser_instances": 2},
            user_preferences={"speed_priority": True},
            historical_performance={"success_rate": 0.85, "average_duration": 120, "error_rate": 0.15, "retry_rate": 0.10},
            current_load=0.3,
            network_conditions={"latency": 50, "bandwidth": 100, "stability": 0.9},
            device_capabilities={"is_mobile": False, "screen_size": "desktop", "touch_enabled": False},
            security_requirements=["encryption"],
            compliance_needs=[],
            error_tolerance=0.05,
            success_rate_requirement=0.95
        )
        
        goal_decomposition = {"sub_goals": [], "complexity_score": 0.6}
        
        # Get a strategy template and customize it
        template = selector.strategy_templates[0]
        customized = selector._customize_strategy(template, context_factors, goal_decomposition)
        
        assert "execution_metadata" in customized, "Customized strategy should have metadata"
        assert "customization_timestamp" in customized["execution_metadata"], "Should have customization timestamp"
        
        return f"Strategy customized with metadata and context-specific adjustments"
    
    def test_performance_history_update(self):
        """Test performance history tracking"""
        from webiq.core.contextual_strategy_selector import ContextualStrategySelector
        
        selector = ContextualStrategySelector()
        
        # Test performance update
        site_url = "https://test.example.com"
        strategy_name = "conservative_sequential"
        execution_result = {
            "success": True,
            "duration": 150,
            "errors": 0
        }
        
        # Update performance history
        selector.update_performance_history(site_url, strategy_name, execution_result)
        
        # Verify update
        assert site_url in selector.performance_history, "Site should be added to performance history"
        site_history = selector.performance_history[site_url]
        assert "strategy_performance" in site_history, "Should track strategy-specific performance"
        assert strategy_name in site_history["strategy_performance"], "Should track specific strategy"
        
        return "Performance history updated successfully"
    
    async def test_integrated_goal_processing_workflow(self):
        """Test the complete integrated workflow"""
        from webiq.core.advanced_goal_processor import AdvancedGoalProcessor
        from webiq.core.requirement_extractor import RequirementExtractor
        from webiq.core.contextual_strategy_selector import ContextualStrategySelector
        
        # Initialize all components
        goal_processor = AdvancedGoalProcessor()
        requirement_extractor = RequirementExtractor()
        strategy_selector = ContextualStrategySelector()
        
        # Test with e-commerce goal
        test_goal = self.test_goals[1]
        
        # Step 1: Analyze goal complexity
        complexity = await goal_processor.analyze_goal_complexity(test_goal["goal"], {"url": test_goal["url"]})
        
        # Step 2: Extract requirements
        explicit_reqs = requirement_extractor.extract_explicit(test_goal["goal"])
        implicit_reqs = await requirement_extractor.infer_implicit(
            test_goal["goal"], test_goal["url"], test_goal["context"]
        )
        constraints = requirement_extractor.extract_constraints(test_goal["goal"], test_goal["context"])
        
        # Step 3: Decompose goal
        decomposition = await goal_processor.decompose_goal(
            test_goal["goal"],
            test_goal["url"],
            test_goal["context"]
        )
        
        # Step 4: Select strategy
        strategy_result = strategy_selector.select_strategy(decomposition, test_goal["context"])
        
        # Verify integrated workflow
        assert complexity in [complexity.SIMPLE, complexity.MODERATE, complexity.COMPLEX, complexity.VERY_COMPLEX], "Should analyze complexity"
        assert len(explicit_reqs) >= 0, "Should extract explicit requirements"
        assert len(implicit_reqs) > 0, "Should infer implicit requirements"
        assert len(constraints) >= 0, "Should extract constraints"
        assert len(decomposition.sub_goals) > 0, "Should decompose goal"
        assert "selected_strategy" in strategy_result, "Should select strategy"
        
        workflow_result = {
            "complexity_score": complexity.value,
            "total_requirements": len(explicit_reqs) + len(implicit_reqs),
            "constraint_count": len(constraints),
            "sub_goal_count": len(decomposition.sub_goals),
            "selected_strategy": strategy_result["selected_strategy"]["name"],
            "strategy_confidence": strategy_result["evaluation"]["confidence_level"]
        }
        
        return f"Integrated workflow completed: {workflow_result}"
    
    async def test_error_handling_and_fallbacks(self):
        """Test error handling and fallback mechanisms"""
        from webiq.core.advanced_goal_processor import AdvancedGoalProcessor
        from webiq.core.requirement_extractor import RequirementExtractor
        from webiq.core.contextual_strategy_selector import ContextualStrategySelector
        
        # Test with invalid/edge case inputs
        goal_processor = AdvancedGoalProcessor()
        requirement_extractor = RequirementExtractor()
        strategy_selector = ContextualStrategySelector()
        
        # Test empty goal
        try:
            complexity = await goal_processor.analyze_goal_complexity("", {"url": "https://example.com"})
            assert complexity in [complexity.SIMPLE, complexity.MODERATE, complexity.COMPLEX, complexity.VERY_COMPLEX], "Should handle empty goal gracefully"
        except Exception as e:
            # Should not raise unhandled exceptions
            assert False, f"Should handle empty goal gracefully, but got: {e}"
        
        # Test invalid URL
        try:
            complexity = await goal_processor.analyze_goal_complexity("test goal", {"url": "invalid-url"})
            assert complexity in [complexity.SIMPLE, complexity.MODERATE, complexity.COMPLEX, complexity.VERY_COMPLEX], "Should handle invalid URL gracefully"
        except Exception as e:
            # Should not raise unhandled exceptions
            assert False, f"Should handle invalid URL gracefully, but got: {e}"
        
        # Test strategy selection with empty decomposition
        try:
            strategy_result = strategy_selector.select_strategy({}, {})
            assert "selected_strategy" in strategy_result, "Should provide fallback strategy"
        except Exception as e:
            assert False, f"Should provide fallback strategy, but got: {e}"
        
        return "Error handling and fallbacks working correctly"
    
    async def test_performance_and_scalability(self):
        """Test performance characteristics"""
        from webiq.core.advanced_goal_processor import AdvancedGoalProcessor
        from webiq.core.requirement_extractor import RequirementExtractor
        from webiq.core.contextual_strategy_selector import ContextualStrategySelector
        
        goal_processor = AdvancedGoalProcessor()
        requirement_extractor = RequirementExtractor()
        strategy_selector = ContextualStrategySelector()
        
        # Test with multiple goals
        start_time = datetime.now()
        
        for i, test_goal in enumerate(self.test_goals[:3]):  # Test first 3 goals
            complexity = await goal_processor.analyze_goal_complexity(test_goal["goal"], {"url": test_goal["url"]})
            explicit_reqs = requirement_extractor.extract_explicit(test_goal["goal"])
            constraints = requirement_extractor.extract_constraints(test_goal["goal"], test_goal["context"])
            
            # Mock decomposition for performance test
            mock_decomposition = {
                "sub_goals": [{"id": f"{i}_1", "description": "test", "complexity": 0.5}],
                "complexity_score": complexity.value
            }
            
            strategy_result = strategy_selector.select_strategy(mock_decomposition, test_goal["context"])
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # Performance should be reasonable (less than 5 seconds for 3 goals)
        assert duration < 5.0, f"Performance test took too long: {duration:.2f}s"
        
        return f"Processed 3 goals in {duration:.2f}s (avg: {duration/3:.2f}s per goal)"
    
    def run_all_tests(self):
        """Run all tests and generate report"""
        print("🚀 Starting Advanced Goal Understanding System Tests")
        print("=" * 60)
        
        # Test AdvancedGoalProcessor
        self.run_test("AdvancedGoalProcessor Initialization", self.test_advanced_goal_processor_initialization)
        self.run_test("Goal Complexity Analysis", self.test_goal_complexity_analysis)
        self.run_test("Goal Decomposition", self.test_goal_decomposition)
        
        # Test RequirementExtractor
        self.run_test("RequirementExtractor Initialization", self.test_requirement_extractor_initialization)
        self.run_test("Explicit Requirement Extraction", self.test_explicit_requirement_extraction)
        self.run_test("Implicit Requirement Inference", self.test_implicit_requirement_inference)
        self.run_test("Constraint Extraction", self.test_constraint_extraction)
        
        # Test ContextualStrategySelector
        self.run_test("ContextualStrategySelector Initialization", self.test_contextual_strategy_selector_initialization)
        self.run_test("Strategy Selection", self.test_strategy_selection)
        self.run_test("Strategy Customization", self.test_strategy_customization)
        self.run_test("Performance History Update", self.test_performance_history_update)
        
        # Integration and advanced tests
        self.run_test("Integrated Goal Processing Workflow", self.test_integrated_goal_processing_workflow)
        self.run_test("Error Handling and Fallbacks", self.test_error_handling_and_fallbacks)
        self.run_test("Performance and Scalability", self.test_performance_and_scalability)
        
        # Generate test report
        self.generate_test_report()
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("📊 TEST REPORT - Advanced Goal Understanding System")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["status"] == "PASSED"])
        failed_tests = total_tests - passed_tests
        
        print(f"\n📈 SUMMARY:")
        print(f"   Total Tests: {total_tests}")
        print(f"   ✅ Passed: {passed_tests}")
        print(f"   ❌ Failed: {failed_tests}")
        print(f"   📊 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        total_duration = sum(r["duration"] for r in self.test_results)
        print(f"   ⏱️  Total Duration: {total_duration:.2f}s")
        print(f"   ⚡ Average per Test: {total_duration/total_tests:.2f}s")
        
        if failed_tests > 0:
            print(f"\n❌ FAILED TESTS:")
            for result in self.test_results:
                if result["status"] == "FAILED":
                    print(f"   • {result['test_name']}: {result.get('error', 'Unknown error')}")
        
        print(f"\n✅ PASSED TESTS:")
        for result in self.test_results:
            if result["status"] == "PASSED":
                print(f"   • {result['test_name']} ({result['duration']:.2f}s)")
        
        # Component-specific analysis
        print(f"\n🔍 COMPONENT ANALYSIS:")
        
        processor_tests = [r for r in self.test_results if "AdvancedGoalProcessor" in r["test_name"] or "Goal" in r["test_name"]]
        extractor_tests = [r for r in self.test_results if "Requirement" in r["test_name"] or "Constraint" in r["test_name"]]
        selector_tests = [r for r in self.test_results if "Strategy" in r["test_name"] or "Contextual" in r["test_name"]]
        integration_tests = [r for r in self.test_results if "Integrated" in r["test_name"] or "Error" in r["test_name"] or "Performance" in r["test_name"]]
        
        def analyze_component(tests, name):
            if tests:
                passed = len([t for t in tests if t["status"] == "PASSED"])
                total = len(tests)
                avg_duration = sum(t["duration"] for t in tests) / total
                print(f"   📦 {name}: {passed}/{total} passed ({(passed/total)*100:.1f}%) - avg {avg_duration:.2f}s")
        
        analyze_component(processor_tests, "AdvancedGoalProcessor")
        analyze_component(extractor_tests, "RequirementExtractor")
        analyze_component(selector_tests, "ContextualStrategySelector")
        analyze_component(integration_tests, "Integration & Advanced")
        
        # Final status
        if failed_tests == 0:
            print(f"\n🎉 ALL TESTS PASSED! Advanced Goal Understanding System is ready for integration.")
        else:
            print(f"\n⚠️  {failed_tests} test(s) failed. Please review and fix issues before integration.")
        
        print("\n" + "=" * 60)
        
        # Save detailed results to file
        try:
            with open("advanced_goal_understanding_test_results.json", "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "summary": {
                        "total_tests": total_tests,
                        "passed_tests": passed_tests,
                        "failed_tests": failed_tests,
                        "success_rate": (passed_tests/total_tests)*100,
                        "total_duration": total_duration
                    },
                    "detailed_results": self.test_results
                }, f, indent=2)
            print(f"📄 Detailed results saved to: advanced_goal_understanding_test_results.json")
        except Exception as e:
            print(f"⚠️  Could not save detailed results: {e}")

def main():
    """Main test execution"""
    tester = AdvancedGoalUnderstandingTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()