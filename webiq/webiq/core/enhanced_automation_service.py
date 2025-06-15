import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .enhanced_agents import (
    OrchestratorAgent, WebAgent, ObserverAgent, SequenceAgent,
    RetryStrategy, SuccessDetector, create_recording_graph,
    PerformanceMonitoringAgent, create_enhanced_recording_graph
)
from .enhanced_recording_session import EnhancedLangGraphRecordingSession
from .performance_agent import (
    PerformanceAgent, PerformanceDashboard, RealTimeOptimizer,
    CostTracker, ResourceMonitor
)
from .stagehand_config import get_stagehand_config
from .exceptions import GoalNotAchievableError, GoalBlockedError, GoalNotCompletedError

logger = logging.getLogger(__name__)

# Mock Stagehand for development - replace with real Stagehand when ready
class MockStagehand:
    """Mock Stagehand implementation for development and testing"""
    
    def __init__(self, config):
        self.config = config
        self.current_url = ""
        self.page = MockPage()
        
    async def init(self):
        logger.info("MockStagehand: Initialized")
        
    async def close(self):
        logger.info("MockStagehand: Closed")
        
class MockPage:
    """Mock page object for Stagehand"""
    
    def __init__(self):
        self.url = ""
        self._content = "<html><body><h1>Mock Page</h1></body></html>"
        
    async def goto(self, url, **kwargs):
        self.url = url
        logger.info(f"MockPage: Navigated to {url}")
        
    async def content(self):
        return self._content
        
    async def screenshot(self, **kwargs):
        return b"mock_screenshot_data"
        
    async def click(self, selector, **kwargs):
        logger.info(f"MockPage: Clicked {selector}")
        
    async def fill(self, selector, text, **kwargs):
        logger.info(f"MockPage: Filled {selector} with '{text}'")
        
    async def evaluate(self, script, **kwargs):
        logger.info(f"MockPage: Evaluated script: {script}")
        return None

async def enhanced_goal_aware_recording(
    goal: str,
    url: str,
    gemini_api_key: str,
    options: Optional[Dict[str, Any]] = None,
    steel_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Enhanced goal-aware recording using multi-agent LangGraph coordination with Performance Monitoring.
    
    This function orchestrates multiple specialized agents to achieve complex web automation goals:
    - OrchestratorAgent: Plans and coordinates the overall execution
    - WebAgent: Executes web interactions using Stagehand
    - ObserverAgent: Monitors progress and validates success
    - SequenceAgent: Generates optimized automation sequences
    - PerformanceMonitoringAgent: Real-time performance optimization and monitoring
    
    Args:
        goal: The automation goal to achieve
        url: Starting URL for the automation
        gemini_api_key: API key for Gemini models
        options: Optional configuration parameters
        steel_api_key: Optional API key for Steel browser sessions
        
    Returns:
        Dict containing execution results, generated sequences, performance metrics, and optimization insights
    """
    logger.info(f"Starting enhanced multi-agent goal-aware recording")
    logger.info(f"URL: {url}")
    logger.info(f"Goal: {goal}")
    logger.info(f"Options: {options}")
    
    # Validate inputs
    if not url or not goal:
        raise ValueError("URL and goal are required")
        
    if not gemini_api_key:
        raise ValueError("Gemini API key is required")
        
    # Enhanced options with defaults
    enhanced_options = {
        "max_attempts": 3,
        "timeout_seconds": 300,
        "enable_screenshots": True,
        "enable_dom_snapshots": True,
        "retry_strategy": "intelligent",
        "observer_frequency": "after_each_action",
        "sequence_optimization": True,
        **options  # User options override defaults
    }
    
    session = None
    stagehand = None
    
    try:
        # Create Steel session
        steel_session_id = f"enhanced_steel_session_{int(datetime.now().timestamp())}"
        logger.info(f"Created Steel session: {steel_session_id}")
        
        # Get Stagehand configuration
        stagehand_config = get_stagehand_config(gemini_api_key, steel_api_key, steel_session_id)
        
        # Initialize Stagehand (using MockStagehand for now)
        # TODO: Replace with real Stagehand when ready
        # from stagehand import Stagehand
        # stagehand = Stagehand(stagehand_config)
        stagehand = MockStagehand(stagehand_config)
        await stagehand.init()
        
        # Navigate to initial URL
        logger.info(f"Navigating to URL: {url}")
        await stagehand.page.goto(url, waitUntil="networkidle")
        stagehand.current_url = url
        
        # Create enhanced recording session
        session = EnhancedLangGraphRecordingSession(
            stagehand=stagehand,
            steel_session=steel_session_id,
            goal=goal,
            options=enhanced_options,
            gemini_api_key=gemini_api_key,
            steel_api_key=steel_api_key
        )
        
        # Start the enhanced multi-agent workflow
        logger.info("Starting enhanced multi-agent workflow...")
        await session.start_monitoring()
        
        # Get comprehensive results
        results = await _compile_enhanced_results(session)
        
        logger.info(f"Enhanced recording completed successfully")
        logger.info(f"Final status: {results.get('execution_status')}")
        logger.info(f"Actions executed: {results.get('total_actions', 0)}")
        logger.info(f"Sequence generated: {results.get('sequence_available', False)}")
        
        return results
        
    except Exception as e:
        logger.error(f"Enhanced goal-aware recording failed: {e}")
        
        # Determine appropriate exception type
        error_message = str(e).lower()
        if "timeout" in error_message or "time" in error_message:
            raise GoalNotCompletedError(f"Recording timed out: {e}")
        elif "blocked" in error_message or "captcha" in error_message:
            raise GoalBlockedError(f"Goal execution blocked: {e}")
        elif "impossible" in error_message or "cannot" in error_message:
            raise GoalNotAchievableError(f"Goal not achievable: {e}")
        else:
            raise GoalNotCompletedError(f"Recording failed: {e}")
            
    finally:
        # Cleanup resources
        if stagehand:
            try:
                await stagehand.close()
                logger.info("Stagehand closed successfully")
            except Exception as e:
                logger.warning(f"Error closing Stagehand: {e}")
                
async def _compile_enhanced_results(session: EnhancedLangGraphRecordingSession) -> Dict[str, Any]:
    """Compile comprehensive results from the enhanced recording session"""
    
    # Get basic progress analysis
    progress_analysis = await session.analyze_goal_progress_enhanced()
    
    # Get detailed action log
    action_log = session.get_detailed_action_log()
    
    # Get generated sequence
    sequence_data = session.get_generated_sequence()
    
    # Get execution metrics
    metrics = session.get_execution_metrics()
    
    # Compile comprehensive results
    results = {
        # Execution Status
        "execution_status": progress_analysis.get("goal_achieved", False),
        "completion_percentage": progress_analysis.get("completion_percentage", 0.0),
        "confidence_score": progress_analysis.get("confidence_score", 0.0),
        "final_phase": progress_analysis.get("current_phase", "unknown"),
        
        # Action Information
        "total_actions": len(action_log),
        "successful_actions": len([a for a in action_log if a.get("success", False)]),
        "failed_actions": len([a for a in action_log if not a.get("success", True)]),
        "action_log": action_log,
        
        # Sequence Generation
        "sequence_available": sequence_data.get("available", False),
        "playwright_script": sequence_data.get("playwright_script", ""),
        "sequence_reliability": sequence_data.get("estimated_reliability", 0.0),
        "optimizations_applied": sequence_data.get("optimizations_applied", []),
        "required_setup": sequence_data.get("required_setup", []),
        
        # Progress Analysis
        "goal_achieved": progress_analysis.get("goal_achieved", False),
        "critical_error": progress_analysis.get("critical_error", False),
        "next_recommended_step": progress_analysis.get("next_step", "Review results"),
        "observer_insights": progress_analysis.get("observer_insights", []),
        "success_indicators": progress_analysis.get("success_indicators_detected", []),
        "issues_detected": progress_analysis.get("issues_detected", []),
        
        # Execution Metrics
        "execution_time_seconds": metrics.get("total_execution_time_seconds", 0),
        "attempts_used": metrics.get("attempts_used", 1),
        "max_attempts": metrics.get("max_attempts", 3),
        "total_errors": metrics.get("total_errors", 0),
        "agent_performance": metrics.get("agent_performance", {}),
        "phase_timings": metrics.get("phase_timings", {}),
        
        # Additional Metadata
        "session_id": session.agent_state.get("session_id", "unknown"),
        "goal": session.goal,
        "url": session.agent_state.get("url", ""),
        "analysis_method": "Enhanced Multi-Agent LangGraph",
        "timestamp": datetime.now().isoformat(),
        
        # Raw Data (for advanced users)
        "raw_agent_state": session.agent_state,
        "raw_observations": session.agent_state.get("observations", []),
        "raw_errors": session.agent_state.get("errors", [])
    }
    
    return results

def validate_enhanced_recording_options(options: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize options for enhanced recording"""
    
    validated_options = {}
    
    # Max attempts validation
    max_attempts = options.get("max_attempts", 3)
    if not isinstance(max_attempts, int) or max_attempts < 1 or max_attempts > 10:
        logger.warning(f"Invalid max_attempts: {max_attempts}, using default: 3")
        max_attempts = 3
    validated_options["max_attempts"] = max_attempts
    
    # Timeout validation
    timeout = options.get("timeout_seconds", 300)
    if not isinstance(timeout, (int, float)) or timeout < 30 or timeout > 1800:
        logger.warning(f"Invalid timeout_seconds: {timeout}, using default: 300")
        timeout = 300
    validated_options["timeout_seconds"] = timeout
    
    # Boolean options
    bool_options = [
        "enable_screenshots", "enable_dom_snapshots", "sequence_optimization"
    ]
    for option in bool_options:
        value = options.get(option, True)
        validated_options[option] = bool(value)
    
    # String options with validation
    retry_strategy = options.get("retry_strategy", "intelligent")
    if retry_strategy not in ["intelligent", "simple", "aggressive", "conservative"]:
        logger.warning(f"Invalid retry_strategy: {retry_strategy}, using default: intelligent")
        retry_strategy = "intelligent"
    validated_options["retry_strategy"] = retry_strategy
    
    observer_frequency = options.get("observer_frequency", "after_each_action")
    if observer_frequency not in ["after_each_action", "periodic", "on_error", "minimal"]:
        logger.warning(f"Invalid observer_frequency: {observer_frequency}, using default: after_each_action")
        observer_frequency = "after_each_action"
    validated_options["observer_frequency"] = observer_frequency
    
    # Copy any additional options
    for key, value in options.items():
        if key not in validated_options:
            validated_options[key] = value
            
    return validated_options

async def get_enhanced_recording_capabilities() -> Dict[str, Any]:
    """Get information about enhanced recording capabilities"""
    
    return {
        "version": "2.0.0",
        "architecture": "Multi-Agent LangGraph",
        "agents": {
            "orchestrator": {
                "model": "Gemini 2.5 Pro",
                "responsibilities": [
                    "Goal decomposition",
                    "Agent coordination",
                    "Error recovery",
                    "Execution planning"
                ]
            },
            "web_agent": {
                "model": "Gemini 2.5 Flash",
                "tools": ["Stagehand"],
                "responsibilities": [
                    "Web action execution",
                    "DOM interaction",
                    "Action logging",
                    "Context awareness"
                ]
            },
            "observer": {
                "model": "Gemini 2.5 Flash",
                "responsibilities": [
                    "Progress monitoring",
                    "Issue detection",
                    "Success prediction",
                    "Alternative suggestions"
                ]
            },
            "sequence_generator": {
                "model": "Gemini 2.5 Pro",
                "responsibilities": [
                    "Playwright script generation",
                    "Sequence optimization",
                    "Reliability estimation",
                    "Setup requirements"
                ]
            }
        },
        "features": [
            "Intelligent retry strategies",
            "Real-time progress monitoring",
            "Dynamic goal decomposition",
            "Context-aware decision making",
            "Optimized sequence generation",
            "Comprehensive error handling",
            "Multi-attempt execution",
            "Success pattern recognition"
        ],
        "supported_options": {
            "max_attempts": "1-10 (default: 3)",
            "timeout_seconds": "30-1800 (default: 300)",
            "retry_strategy": "intelligent|simple|aggressive|conservative",
            "observer_frequency": "after_each_action|periodic|on_error|minimal",
            "enable_screenshots": "boolean (default: true)",
            "enable_dom_snapshots": "boolean (default: true)",
            "sequence_optimization": "boolean (default: true)"
        }
    }