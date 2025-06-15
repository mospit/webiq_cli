import asyncio
import logging
from typing import Dict, List, Any, Optional
from .agents import WebAutomationOrchestrator, WebAutomationState, AgentState
from .recording_session import GoalAwareRecordingSession

logger = logging.getLogger(__name__)

class LangGraphRecordingSession(GoalAwareRecordingSession):
    """Enhanced recording session that integrates LangGraph multi-agent automation results"""
    
    def __init__(self, stagehand, steel_session, goal: str, automation_result: WebAutomationState, 
                 options: dict, orchestrator: WebAutomationOrchestrator):
        # Initialize parent class with automation results as goal_analysis
        goal_analysis = self._extract_goal_analysis_from_automation(automation_result)
        super().__init__(stagehand, steel_session, goal, goal_analysis, options)
        
        # Store LangGraph-specific data
        self.automation_result = automation_result
        self.orchestrator = orchestrator
        self.agent_messages = automation_result.messages
        self.executed_actions = automation_result.executed_actions
        self.action_plan = automation_result.action_plan
        self.completion_percentage = automation_result.completion_percentage
        self.agent_state = automation_result.agent_state
        
        # Override parent attributes with LangGraph data
        self.screenshots = automation_result.screenshots
        self.goal_progress = automation_result.goal_progress
        
        logger.info(f"LangGraphRecordingSession initialized with {len(self.executed_actions)} executed actions")
        
    def _extract_goal_analysis_from_automation(self, automation_result: WebAutomationState) -> Dict[str, Any]:
        """Extract goal analysis data from automation result for compatibility with parent class"""
        return {
            "goal_steps": [action.get('description', '') for action in automation_result.action_plan],
            "executed_actions": automation_result.executed_actions,
            "completion_percentage": automation_result.completion_percentage,
            "agent_state": automation_result.agent_state.value,
            "error_message": automation_result.error_message,
            "success_indicators": self._extract_success_indicators(automation_result),
            "estimated_difficulty": self._estimate_difficulty(automation_result)
        }
        
    def _extract_success_indicators(self, automation_result: WebAutomationState) -> List[str]:
        """Extract success indicators from goal progress data"""
        indicators = []
        for progress in automation_result.goal_progress:
            if isinstance(progress, dict) and 'success_indicators_met' in progress:
                indicators.extend(progress['success_indicators_met'])
        return list(set(indicators))  # Remove duplicates
        
    def _estimate_difficulty(self, automation_result: WebAutomationState) -> str:
        """Estimate difficulty based on automation results"""
        if automation_result.agent_state == AgentState.ERROR:
            return "complex"
        elif len(automation_result.action_plan) > 10:
            return "complex"
        elif len(automation_result.action_plan) > 5:
            return "medium"
        else:
            return "simple"
            
    async def start_monitoring(self):
        """Enhanced monitoring that can continue automation if needed"""
        logger.info(f"Starting LangGraph-enhanced monitoring for goal: {self.goal}")
        
        # If automation is not completed, continue the workflow
        if self.agent_state not in [AgentState.COMPLETED, AgentState.ERROR]:
            logger.info("Automation not completed, continuing multi-agent workflow...")
            try:
                # Continue the automation workflow
                continued_result = await self.orchestrator.workflow.ainvoke(self.automation_result)
                
                # Update our state with continued results
                self.automation_result = continued_result
                self.executed_actions = continued_result.executed_actions
                self.screenshots = continued_result.screenshots
                self.goal_progress = continued_result.goal_progress
                self.completion_percentage = continued_result.completion_percentage
                self.agent_state = continued_result.agent_state
                
                logger.info(f"Continued automation completed with state: {self.agent_state}")
                
            except Exception as e:
                logger.error(f"Error continuing automation: {e}")
                self.agent_state = AgentState.ERROR
        
        # Start traditional monitoring
        self.is_recording = True
        logger.info(f"Starting traditional monitoring for goal: {self.goal}")
        
        while self.is_recording:
            # Enhanced progress analysis using LangGraph agents
            progress = await self.analyze_goal_progress_enhanced()
            self.goal_progress.append(progress)
            
            # Check if goal is achieved or if we should stop
            if progress.get('goal_achieved', False) or self.agent_state == AgentState.COMPLETED:
                logger.info("Goal achieved, stopping monitoring")
                self.is_recording = False
                break
                
            if progress.get('critical_error', False) or self.agent_state == AgentState.ERROR:
                logger.error("Critical error detected, stopping monitoring")
                self.is_recording = False
                break
                
            # Wait before next check
            await asyncio.sleep(2)
            
            # Safety break after reasonable time
            if len(self.goal_progress) > 30:  # 60 seconds of monitoring
                logger.info("Monitoring timeout reached")
                self.is_recording = False
                
        logger.info(f"Finished LangGraph-enhanced monitoring for goal: {self.goal}")
        
    async def analyze_goal_progress_enhanced(self) -> Dict[str, Any]:
        """Enhanced goal progress analysis using LangGraph monitoring agent"""
        logger.info(f"Enhanced goal progress analysis for: {self.goal}")
        
        try:
            # Use the monitoring agent for enhanced analysis
            updated_state = await self.orchestrator.monitoring_agent.monitor_progress(self.automation_result)
            
            # Extract the latest progress analysis
            if updated_state.goal_progress:
                latest_progress = updated_state.goal_progress[-1]
            else:
                latest_progress = {
                    "completion_percentage": self.completion_percentage,
                    "goal_achieved": self.agent_state == AgentState.COMPLETED,
                    "critical_error": self.agent_state == AgentState.ERROR,
                    "next_step": "Continue monitoring",
                    "analysis_method": "LangGraph enhanced"
                }
                
            # Update our automation result with the latest state
            self.automation_result = updated_state
            self.completion_percentage = updated_state.completion_percentage
            self.agent_state = updated_state.agent_state
            
            logger.info(f"Enhanced goal progress analysis result: {latest_progress}")
            return latest_progress
            
        except Exception as e:
            logger.error(f"Error in enhanced goal progress analysis: {e}")
            # Fallback to parent method
            return await super().analyze_goal_progress()
            
    def get_automation_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the automation process"""
        return {
            "goal": self.goal,
            "agent_state": self.agent_state.value if hasattr(self.agent_state, 'value') else str(self.agent_state),
            "completion_percentage": self.completion_percentage,
            "total_actions_planned": len(self.action_plan),
            "total_actions_executed": len(self.executed_actions),
            "screenshots_captured": len(self.screenshots),
            "progress_analyses": len(self.goal_progress),
            "error_message": self.automation_result.error_message,
            "success_indicators_met": self._extract_success_indicators(self.automation_result),
            "estimated_difficulty": self._estimate_difficulty(self.automation_result),
            "agent_messages_count": len(self.agent_messages)
        }
        
    def get_detailed_action_log(self) -> List[Dict[str, Any]]:
        """Get detailed log of all executed actions"""
        detailed_log = []
        
        for i, action in enumerate(self.executed_actions):
            log_entry = {
                "step_number": i + 1,
                "action_type": action.get('action_type', 'unknown'),
                "description": action.get('description', 'No description'),
                "target": action.get('target', 'No target'),
                "value": action.get('value', 'No value'),
                "status": action.get('status', 'unknown'),
                "executed_at": action.get('executed_at', 'unknown'),
                "success_criteria": action.get('success_criteria', 'Not specified'),
                "estimated_time": action.get('estimated_time', 'Not specified')
            }
            
            # Add result if available
            if 'result' in action:
                log_entry['result'] = action['result']
                
            detailed_log.append(log_entry)
            
        return detailed_log
        
    def export_session_data(self) -> Dict[str, Any]:
        """Export complete session data for analysis or replay"""
        return {
            "session_metadata": {
                "goal": self.goal,
                "url": self.automation_result.url,
                "session_id": self.steel_session.get('id', 'unknown'),
                "options": self.options,
                "completion_percentage": self.completion_percentage,
                "final_state": self.agent_state.value if hasattr(self.agent_state, 'value') else str(self.agent_state)
            },
            "automation_summary": self.get_automation_summary(),
            "action_plan": self.action_plan,
            "executed_actions": self.get_detailed_action_log(),
            "goal_progress": self.goal_progress,
            "screenshots": self.screenshots,
            "agent_messages": [{
                "type": type(msg).__name__,
                "content": msg.content if hasattr(msg, 'content') else str(msg)
            } for msg in self.agent_messages],
            "error_details": {
                "error_message": self.automation_result.error_message,
                "error_occurred": self.agent_state == AgentState.ERROR
            }
        }