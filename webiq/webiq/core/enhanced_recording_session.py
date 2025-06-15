import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .enhanced_agents import (
    create_recording_graph, 
    AgentState, 
    RetryStrategy, 
    SuccessDetector
)
from .recording_session import GoalAwareRecordingSession

logger = logging.getLogger(__name__)

class EnhancedLangGraphRecordingSession(GoalAwareRecordingSession):
    """Enhanced recording session using the new multi-agent architecture"""
    
    def __init__(self, stagehand, steel_session, goal: str, options: dict, 
                 gemini_api_key: str, steel_api_key: str):
        # Initialize parent class with minimal goal_analysis
        initial_goal_analysis = {
            "goal_steps": ["Initialize multi-agent workflow"],
            "executed_actions": [],
            "completion_percentage": 0.0,
            "agent_state": "initializing",
            "error_message": None,
            "success_indicators": [],
            "estimated_difficulty": 5
        }
        
        super().__init__(stagehand, steel_session, goal, initial_goal_analysis, options)
        
        # Enhanced agent system components
        self.gemini_api_key = gemini_api_key
        self.steel_api_key = steel_api_key
        self.workflow = create_recording_graph(gemini_api_key)
        self.retry_strategy = RetryStrategy()
        self.success_detector = SuccessDetector(gemini_api_key)
        
        # Initialize enhanced state
        self.agent_state = self._initialize_agent_state()
        self.execution_start_time = datetime.now()
        self.phase_timings = {}
        
        logger.info(f"EnhancedLangGraphRecordingSession initialized for goal: {goal}")
        
    def _initialize_agent_state(self) -> AgentState:
        """Initialize the enhanced agent state"""
        session_id = f"enhanced_session_{int(datetime.now().timestamp())}"
        
        return {
            "goal": self.goal,
            "url": getattr(self.stagehand, 'current_url', ''),
            "session_id": session_id,
            "current_action": "initialization",
            "action_history": [],
            "observations": [],
            "dom_snapshots": [],
            "errors": [],
            "sequence": {},
            "status": "planning",
            "current_phase": "initialization",
            "max_attempts": self.options.get("max_attempts", 3),
            "current_attempt": 1,
            "stagehand_instance": self.stagehand,
            "steel_session": self.steel_session,
            "options": self.options,
            "progress_score": 0.0,
            "completion_confidence": 0.0,
            "retry_strategy": {
                "enabled": True,
                "max_retries_per_action": 2,
                "backoff_strategy": "exponential"
            }
        }
    
    async def start_monitoring(self):
        """Start the enhanced multi-agent monitoring and execution workflow"""
        logger.info(f"Starting enhanced multi-agent workflow for goal: {self.goal}")
        
        try:
            # Record start time for this phase
            self.phase_timings["workflow_start"] = datetime.now()
            
            # Execute the multi-agent workflow
            final_state = await self.workflow.ainvoke(self.agent_state)
            
            # Record completion time
            self.phase_timings["workflow_complete"] = datetime.now()
            
            # Update our internal state with results
            self.agent_state = final_state
            
            # Extract results for parent class compatibility
            await self._extract_results_for_compatibility(final_state)
            
            # Log final results
            self._log_execution_summary(final_state)
            
            logger.info(f"Enhanced workflow completed with status: {final_state.get('status')}")
            
        except Exception as e:
            logger.error(f"Enhanced workflow failed: {e}")
            self.agent_state["status"] = "failed"
            self.agent_state["errors"].append({
                "timestamp": datetime.now().isoformat(),
                "agent": "workflow",
                "error_type": "workflow_exception",
                "message": str(e)
            })
            
    async def _extract_results_for_compatibility(self, final_state: AgentState):
        """Extract results from enhanced workflow for parent class compatibility"""
        # Update executed actions from action history
        web_actions = [
            action for action in final_state.get("action_history", [])
            if action.get("agent") == "web_agent"
        ]
        
        self.executed_actions = web_actions
        
        # Update screenshots from DOM snapshots
        if final_state.get("dom_snapshots"):
            # Convert DOM snapshots to screenshot references
            self.screenshots = [
                f"dom_snapshot_{i}.html" for i, _ in enumerate(final_state["dom_snapshots"])
            ]
            
        # Update goal progress from observations
        self.goal_progress = final_state.get("observations", [])
        
        # Update completion percentage
        self.completion_percentage = final_state.get("progress_score", 0.0) * 100
        
        # Update goal analysis with enhanced data
        self.goal_analysis.update({
            "executed_actions": self.executed_actions,
            "completion_percentage": self.completion_percentage,
            "agent_state": final_state.get("status", "unknown"),
            "error_message": self._extract_error_summary(final_state),
            "success_indicators": self._extract_success_indicators(final_state),
            "enhanced_sequence": final_state.get("sequence", {})
        })
        
    def _extract_error_summary(self, final_state: AgentState) -> Optional[str]:
        """Extract a summary of errors from the final state"""
        errors = final_state.get("errors", [])
        if not errors:
            return None
            
        # Get the most recent critical error
        critical_errors = [e for e in errors if "failure" in e.get("error_type", "")]
        if critical_errors:
            return critical_errors[-1].get("message", "Unknown error")
            
        # Otherwise return the most recent error
        return errors[-1].get("message", "Unknown error") if errors else None
        
    def _extract_success_indicators(self, final_state: AgentState) -> List[str]:
        """Extract success indicators from observations"""
        indicators = []
        
        for observation in final_state.get("observations", []):
            if isinstance(observation, dict):
                success_indicators = observation.get("success_indicators", [])
                if isinstance(success_indicators, list):
                    indicators.extend(success_indicators)
                    
        return list(set(indicators))  # Remove duplicates
        
    def _log_execution_summary(self, final_state: AgentState):
        """Log a comprehensive summary of the execution"""
        total_duration = (datetime.now() - self.execution_start_time).total_seconds()
        
        summary = {
            "goal": final_state.get("goal"),
            "final_status": final_state.get("status"),
            "total_duration_seconds": total_duration,
            "total_actions": len(final_state.get("action_history", [])),
            "web_actions": len([a for a in final_state.get("action_history", []) if a.get("agent") == "web_agent"]),
            "observations": len(final_state.get("observations", [])),
            "errors": len(final_state.get("errors", [])),
            "final_progress_score": final_state.get("progress_score", 0.0),
            "completion_confidence": final_state.get("completion_confidence", 0.0),
            "attempts_used": final_state.get("current_attempt", 1),
            "sequence_generated": bool(final_state.get("sequence"))
        }
        
        logger.info(f"Enhanced workflow execution summary: {summary}")
        
    async def analyze_goal_progress_enhanced(self) -> Dict[str, Any]:
        """Enhanced goal progress analysis using multi-agent insights"""
        logger.info(f"Enhanced goal progress analysis for: {self.goal}")
        
        if not self.agent_state:
            return await self.analyze_goal_progress()  # Fallback to parent method
            
        # Get the latest observation
        observations = self.agent_state.get("observations", [])
        latest_observation = observations[-1] if observations else {}
        
        # Enhanced progress analysis
        progress_analysis = {
            "completion_percentage": self.agent_state.get("progress_score", 0.0) * 100,
            "goal_achieved": self.agent_state.get("status") == "complete",
            "critical_error": self.agent_state.get("status") == "failed",
            "next_step": self._determine_next_step(),
            "analysis_method": "Enhanced Multi-Agent",
            "confidence_score": self.agent_state.get("completion_confidence", 0.0),
            "current_phase": self.agent_state.get("current_phase", "unknown"),
            "total_actions_executed": len(self.agent_state.get("action_history", [])),
            "errors_encountered": len(self.agent_state.get("errors", [])),
            "observer_insights": latest_observation.get("recommendations", []),
            "success_indicators_detected": latest_observation.get("success_indicators", []),
            "issues_detected": latest_observation.get("issues_detected", []),
            "estimated_time_remaining": self._estimate_time_remaining()
        }
        
        return progress_analysis
        
    def _determine_next_step(self) -> str:
        """Determine the recommended next step based on current state"""
        status = self.agent_state.get("status", "unknown")
        
        if status == "complete":
            return "Review generated sequence"
        elif status == "failed":
            if self.agent_state.get("current_attempt", 1) < self.agent_state.get("max_attempts", 3):
                return "Retry with different strategy"
            else:
                return "Manual intervention required"
        elif status == "executing":
            return "Continue automated execution"
        elif status == "observing":
            return "Analyzing current state"
        else:
            return "Continue monitoring"
            
    def _estimate_time_remaining(self) -> str:
        """Estimate remaining time based on current progress"""
        progress = self.agent_state.get("progress_score", 0.0)
        elapsed = (datetime.now() - self.execution_start_time).total_seconds()
        
        if progress > 0.1:
            estimated_total = elapsed / progress
            remaining = max(0, estimated_total - elapsed)
            return f"{int(remaining)} seconds"
        else:
            return "Unknown"
            
    def get_detailed_action_log(self) -> List[Dict[str, Any]]:
        """Get detailed log of all executed actions with enhanced metadata"""
        if not self.agent_state:
            return super().get_detailed_action_log()
            
        detailed_log = []
        
        for action in self.agent_state.get("action_history", []):
            # Enhanced action log entry
            log_entry = {
                "timestamp": action.get("timestamp"),
                "agent": action.get("agent"),
                "action_type": action.get("action"),
                "target": action.get("target", {}),
                "success": action.get("success", False),
                "duration_ms": action.get("duration_ms", 0),
                "context": {
                    "dom_available": bool(action.get("context", {}).get("dom_before")),
                    "screenshot_available": bool(action.get("context", {}).get("screenshot_before")),
                    "result_data": action.get("result", {})
                },
                "error_info": action.get("error") if not action.get("success") else None
            }
            
            detailed_log.append(log_entry)
            
        return detailed_log
        
    def get_generated_sequence(self) -> Dict[str, Any]:
        """Get the generated Playwright sequence with metadata"""
        if not self.agent_state or not self.agent_state.get("sequence"):
            return {
                "available": False,
                "reason": "Sequence not yet generated or workflow incomplete"
            }
            
        sequence_data = self.agent_state["sequence"]
        
        return {
            "available": True,
            "playwright_script": sequence_data.get("playwright_script", ""),
            "optimizations_applied": sequence_data.get("optimizations_applied", []),
            "estimated_reliability": sequence_data.get("estimated_reliability", 0.0),
            "required_setup": sequence_data.get("required_setup", []),
            "generation_timestamp": datetime.now().isoformat(),
            "source_actions": len([a for a in self.agent_state.get("action_history", []) if a.get("agent") == "web_agent"]),
            "total_observations": len(self.agent_state.get("observations", []))
        }
        
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get comprehensive execution metrics"""
        if not self.agent_state:
            return {"available": False}
            
        total_duration = (datetime.now() - self.execution_start_time).total_seconds()
        
        # Calculate agent-specific metrics
        agent_actions = {}
        for action in self.agent_state.get("action_history", []):
            agent = action.get("agent", "unknown")
            if agent not in agent_actions:
                agent_actions[agent] = {"count": 0, "success_count": 0, "total_duration": 0}
            agent_actions[agent]["count"] += 1
            if action.get("success"):
                agent_actions[agent]["success_count"] += 1
            agent_actions[agent]["total_duration"] += action.get("duration_ms", 0)
            
        return {
            "available": True,
            "total_execution_time_seconds": total_duration,
            "final_status": self.agent_state.get("status"),
            "progress_score": self.agent_state.get("progress_score", 0.0),
            "completion_confidence": self.agent_state.get("completion_confidence", 0.0),
            "attempts_used": self.agent_state.get("current_attempt", 1),
            "max_attempts": self.agent_state.get("max_attempts", 3),
            "total_errors": len(self.agent_state.get("errors", [])),
            "agent_performance": agent_actions,
            "observations_count": len(self.agent_state.get("observations", [])),
            "dom_snapshots_count": len(self.agent_state.get("dom_snapshots", [])),
            "phase_timings": self.phase_timings
        }