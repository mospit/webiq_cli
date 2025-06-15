#!/usr/bin/env python3
"""
Enhanced Multi-Agent Recording Example

This example demonstrates how to use the new enhanced multi-agent system
for goal-aware web automation recording with LangGraph coordination.

Features demonstrated:
- Multi-agent coordination (Orchestrator, Web, Observer, Sequence)
- Intelligent retry strategies
- Real-time progress monitoring
- Optimized Playwright sequence generation
- Comprehensive error handling
"""

import asyncio
import os
import json
from datetime import datetime
from typing import Dict, Any

# Import the enhanced recording system
from webiq.core import (
    enhanced_goal_aware_recording,
    validate_enhanced_recording_options,
    get_enhanced_recording_capabilities
)

async def example_basic_recording():
    """Basic example of enhanced goal-aware recording"""
    
    print("\n=== Basic Enhanced Recording Example ===")
    
    # Configuration
    url = "https://example.com/login"
    goal = "Log into the website using the demo account"
    
    # API keys (replace with your actual keys)
    gemini_api_key = os.getenv("GEMINI_API_KEY", "your_gemini_api_key_here")
    steel_api_key = os.getenv("STEEL_API_KEY", "your_steel_api_key_here")
    
    # Basic options
    options = {
        "max_attempts": 3,
        "timeout_seconds": 180,
        "enable_screenshots": True
    }
    
    try:
        print(f"Starting recording for goal: {goal}")
        print(f"Target URL: {url}")
        
        # Execute enhanced recording
        results = await enhanced_goal_aware_recording(
            url=url,
            goal=goal,
            options=options,
            gemini_api_key=gemini_api_key,
            steel_api_key=steel_api_key
        )
        
        # Display results
        print("\n--- Recording Results ---")
        print(f"Goal Achieved: {results['goal_achieved']}")
        print(f"Completion: {results['completion_percentage']:.1f}%")
        print(f"Confidence: {results['confidence_score']:.2f}")
        print(f"Total Actions: {results['total_actions']}")
        print(f"Execution Time: {results['execution_time_seconds']:.1f}s")
        
        if results['sequence_available']:
            print("\n--- Generated Playwright Script ---")
            print(results['playwright_script'][:500] + "..." if len(results['playwright_script']) > 500 else results['playwright_script'])
            
        return results
        
    except Exception as e:
        print(f"Recording failed: {e}")
        return None

async def example_advanced_recording():
    """Advanced example with custom retry strategies and monitoring"""
    
    print("\n=== Advanced Enhanced Recording Example ===")
    
    # Configuration for complex scenario
    url = "https://example.com/complex-form"
    goal = "Fill out the multi-step registration form with email verification"
    
    # API keys
    gemini_api_key = os.getenv("GEMINI_API_KEY", "your_gemini_api_key_here")
    steel_api_key = os.getenv("STEEL_API_KEY", "your_steel_api_key_here")
    
    # Advanced options
    options = {
        "max_attempts": 5,
        "timeout_seconds": 300,
        "retry_strategy": "intelligent",
        "observer_frequency": "after_each_action",
        "enable_screenshots": True,
        "enable_dom_snapshots": True,
        "sequence_optimization": True,
        "custom_success_indicators": [
            "Thank you for registering",
            "Registration successful",
            "Welcome to"
        ],
        "custom_error_patterns": [
            "Email already exists",
            "Invalid captcha",
            "Server error"
        ]
    }
    
    # Validate options
    validated_options = validate_enhanced_recording_options(options)
    print(f"Using validated options: {json.dumps(validated_options, indent=2)}")
    
    try:
        print(f"Starting advanced recording for goal: {goal}")
        
        # Execute enhanced recording with progress monitoring
        results = await enhanced_goal_aware_recording(
            url=url,
            goal=goal,
            options=validated_options,
            gemini_api_key=gemini_api_key,
            steel_api_key=steel_api_key
        )
        
        # Comprehensive results analysis
        print("\n--- Comprehensive Results Analysis ---")
        print(f"Execution Status: {results['execution_status']}")
        print(f"Final Phase: {results['final_phase']}")
        print(f"Attempts Used: {results['attempts_used']}/{results['max_attempts']}")
        
        # Agent performance breakdown
        if 'agent_performance' in results:
            print("\n--- Agent Performance ---")
            for agent, metrics in results['agent_performance'].items():
                print(f"{agent}: {metrics}")
        
        # Observer insights
        if results.get('observer_insights'):
            print("\n--- Observer Insights ---")
            for insight in results['observer_insights']:
                print(f"- {insight}")
        
        # Issues detected
        if results.get('issues_detected'):
            print("\n--- Issues Detected ---")
            for issue in results['issues_detected']:
                print(f"- {issue}")
        
        # Sequence analysis
        if results['sequence_available']:
            print("\n--- Sequence Analysis ---")
            print(f"Reliability Score: {results['sequence_reliability']:.2f}")
            print(f"Optimizations Applied: {results['optimizations_applied']}")
            print(f"Required Setup: {results['required_setup']}")
            
            # Save the generated script
            script_filename = f"generated_script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.js"
            with open(script_filename, 'w') as f:
                f.write(results['playwright_script'])
            print(f"Playwright script saved to: {script_filename}")
        
        return results
        
    except Exception as e:
        print(f"Advanced recording failed: {e}")
        return None

async def example_error_recovery():
    """Example demonstrating error recovery and retry strategies"""
    
    print("\n=== Error Recovery Example ===")
    
    # Scenario designed to test error recovery
    url = "https://httpstat.us/500"  # This will return a 500 error
    goal = "Navigate to a page that might have issues and handle errors gracefully"
    
    # API keys
    gemini_api_key = os.getenv("GEMINI_API_KEY", "your_gemini_api_key_here")
    steel_api_key = os.getenv("STEEL_API_KEY", "your_steel_api_key_here")
    
    # Options focused on error recovery
    options = {
        "max_attempts": 3,
        "timeout_seconds": 60,
        "retry_strategy": "aggressive",
        "observer_frequency": "on_error",
        "enable_screenshots": True
    }
    
    try:
        print(f"Testing error recovery with URL: {url}")
        
        results = await enhanced_goal_aware_recording(
            url=url,
            goal=goal,
            options=options,
            gemini_api_key=gemini_api_key,
            steel_api_key=steel_api_key
        )
        
        print("\n--- Error Recovery Results ---")
        print(f"Final Status: {results['execution_status']}")
        print(f"Total Errors: {results['total_errors']}")
        print(f"Failed Actions: {results['failed_actions']}")
        
        if results.get('raw_errors'):
            print("\n--- Error Log ---")
            for error in results['raw_errors']:
                print(f"- {error}")
        
        return results
        
    except Exception as e:
        print(f"Error recovery test completed with exception: {e}")
        print("This is expected behavior for error recovery testing.")
        return None

async def show_capabilities():
    """Display the capabilities of the enhanced recording system"""
    
    print("\n=== Enhanced Recording System Capabilities ===")
    
    capabilities = await get_enhanced_recording_capabilities()
    
    print(f"Version: {capabilities['version']}")
    print(f"Architecture: {capabilities['architecture']}")
    
    print("\n--- Agents ---")
    for agent_name, agent_info in capabilities['agents'].items():
        print(f"\n{agent_name.title()}:")
        print(f"  Model: {agent_info['model']}")
        if 'tools' in agent_info:
            print(f"  Tools: {', '.join(agent_info['tools'])}")
        print(f"  Responsibilities:")
        for responsibility in agent_info['responsibilities']:
            print(f"    - {responsibility}")
    
    print("\n--- Features ---")
    for feature in capabilities['features']:
        print(f"- {feature}")
    
    print("\n--- Supported Options ---")
    for option, description in capabilities['supported_options'].items():
        print(f"- {option}: {description}")

async def main():
    """Main function to run all examples"""
    
    print("Enhanced Multi-Agent Recording System Examples")
    print("=" * 50)
    
    # Check for API keys
    if not os.getenv("GEMINI_API_KEY") or not os.getenv("STEEL_API_KEY"):
        print("\nWARNING: Please set GEMINI_API_KEY and STEEL_API_KEY environment variables")
        print("Example: export GEMINI_API_KEY='your_key_here'")
        print("Example: export STEEL_API_KEY='your_key_here'")
        print("\nRunning with mock keys for demonstration...\n")
    
    # Show system capabilities
    await show_capabilities()
    
    # Run examples
    examples = [
        ("Basic Recording", example_basic_recording),
        ("Advanced Recording", example_advanced_recording),
        ("Error Recovery", example_error_recovery)
    ]
    
    results = {}
    
    for example_name, example_func in examples:
        print(f"\n{'='*60}")
        print(f"Running: {example_name}")
        print(f"{'='*60}")
        
        try:
            result = await example_func()
            results[example_name] = result
            print(f"\n✅ {example_name} completed")
        except Exception as e:
            print(f"\n❌ {example_name} failed: {e}")
            results[example_name] = None
    
    # Summary
    print(f"\n{'='*60}")
    print("EXECUTION SUMMARY")
    print(f"{'='*60}")
    
    for example_name, result in results.items():
        status = "✅ SUCCESS" if result else "❌ FAILED"
        print(f"{example_name}: {status}")
        
        if result and isinstance(result, dict):
            goal_achieved = result.get('goal_achieved', False)
            completion = result.get('completion_percentage', 0)
            print(f"  Goal Achieved: {goal_achieved}")
            print(f"  Completion: {completion:.1f}%")
    
    print("\nExample execution completed!")
    print("\nNext steps:")
    print("1. Set up your actual API keys")
    print("2. Modify the URLs and goals for your specific use cases")
    print("3. Experiment with different options and retry strategies")
    print("4. Review the generated Playwright scripts")

if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())