#!/usr/bin/env python3
"""
Test script for Performance Agent implementation
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'webiq'))

try:
    from webiq.core.performance_agent import (
        PerformanceAgent,
        PerformanceMetric,
        MetricType,
        OptimizationSuggestion,
        CostTracker,
        ResourceMonitor,
        RealTimeOptimizer,
        PerformanceDashboard
    )
    print("✅ Successfully imported Performance Agent components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

async def test_performance_agent():
    """Test basic Performance Agent functionality"""
    print("\n🧪 Testing Performance Agent...")
    
    try:
        # Initialize Performance Agent
        agent = PerformanceAgent()
        print("✅ Performance Agent initialized")
        
        # Create test metric
        test_metric = PerformanceMetric(
            metric_id="test_001",
            metric_type=MetricType.EXECUTION_TIME,
            value=2.5,
            timestamp=datetime.now(),
            agent_id="test_agent",
            session_id="test_session",
            action_context={"action": "test_action", "success": True}
        )
        print("✅ Test metric created")
        
        # Track metric
        await agent.track_agent_execution(
            agent_id="test_agent",
            action="test_action",
            execution_time=2.5,
            success=True,
            session_id="test_session"
        )
        print("✅ Metric tracking successful")
        
        # Test optimization suggestions
        suggestions = await agent.analyze_metric_optimization(test_metric)
        print(f"✅ Generated {len(suggestions)} optimization suggestions")
        
        # Test performance summary
        summary = await agent.get_performance_summary(time_window_hours=1)
        print("✅ Performance summary generated")
        
        print("\n🎉 All Performance Agent tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_imports():
    """Test that all required components can be imported"""
    print("🔍 Testing imports...")
    
    components = [
        'PerformanceAgent',
        'PerformanceMetric', 
        'MetricType',
        'OptimizationSuggestion',
        'CostTracker',
        'ResourceMonitor',
        'RealTimeOptimizer',
        'PerformanceDashboard'
    ]
    
    for component in components:
        try:
            globals()[component]
            print(f"  ✅ {component}")
        except NameError:
            print(f"  ❌ {component} - Not found")
            return False
    
    return True

if __name__ == "__main__":
    print("🚀 Performance Agent Test Suite")
    print("=" * 40)
    
    # Test imports first
    if not test_imports():
        print("\n❌ Import tests failed")
        sys.exit(1)
    
    # Test async functionality
    try:
        success = asyncio.run(test_performance_agent())
        if success:
            print("\n🎯 All tests completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        sys.exit(1)