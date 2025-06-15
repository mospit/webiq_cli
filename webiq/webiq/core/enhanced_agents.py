import asyncio
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, TypedDict, Annotated
from enum import Enum
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Import Performance Agent from the core directory
try:
    from ..core.performance_agent import PerformanceAgent, PerformanceMetric, MetricType
except ImportError:
    # Fallback import if the structure is different
    from .performance_agent import PerformanceAgent, PerformanceMetric, MetricType

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """Enhanced state for the multi-agent recording workflow"""
    goal: str
    url: str
    session_id: str
    current_action: str
    action_history: Annotated[List[Dict[str, Any]], operator.add]
    observations: Annotated[List[Dict[str, Any]], operator.add]
    dom_snapshots: Annotated[List[str], operator.add]
    errors: Annotated[List[Dict[str, Any]], operator.add]
    sequence: Dict[str, Any]
    status: str  # "planning", "executing", "observing", "complete", "failed"
    current_phase: str
    max_attempts: int
    current_attempt: int
    stagehand_instance: Any
    steel_session: Any
    options: Dict[str, Any]
    progress_score: float
    completion_confidence: float
    retry_strategy: Dict[str, Any]
    
    # Performance monitoring
    performance_agent: Optional[PerformanceAgent]
    current_optimizations: Dict[str, Any]
    real_time_suggestions: List[Dict[str, Any]]

class PerformanceMonitoringAgent:
    """Performance monitoring and optimization agent"""
    
    def __init__(self):
        self.performance_agent = PerformanceAgent()
    
    async def monitor_execution(self, state: AgentState) -> AgentState:
        """Monitor and optimize agent execution"""
        try:
            # Initialize performance agent if not present
            if not state.get("performance_agent"):
                state["performance_agent"] = self.performance_agent
            
            # Get current performance metrics
            recent_metrics = self.performance_agent.get_recent_metrics(minutes=5)
            
            # Generate real-time optimizations
            if recent_metrics:
                current_state_dict = {
                    "session_id": state["session_id"],
                    "current_step": state.get("current_step", 0),
                    "recent_performance": recent_metrics
                }
                
                upcoming_action = state.get("next_agent", "unknown")
                optimizations = await self.performance_agent.optimizer.optimize_next_action(
                    current_state_dict, upcoming_action
                )
                
                state["current_optimizations"] = optimizations
                
                # Generate suggestions for immediate improvements
                suggestions = await self.performance_agent.generate_optimization_suggestions(recent_metrics)
                state["real_time_suggestions"] = [
                    {
                        "type": s.suggestion_type,
                        "description": s.description,
                        "impact_score": s.impact_score,
                        "priority": s.priority
                    } for s in suggestions[:3]  # Top 3 suggestions
                ]
            
            logger.info(f"Performance monitoring completed for session {state['session_id']}")
            
        except Exception as e:
            logger.error(f"Performance monitoring failed: {e}")
            # Don't fail the entire workflow due to monitoring issues
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append({
                "agent": "performance_monitor",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
        
        return state
    
    async def track_agent_execution(self, agent_id: str, action: str, state: AgentState) -> PerformanceMetric:
        """Track individual agent execution performance"""
        execution_context = {
            "session_id": state["session_id"],
            "current_step": state.get("current_step", 0),
            "goal": state["goal"]
        }
        
        return await self.performance_agent.monitor_agent_execution(
            agent_id, action, execution_context
        )


class OrchestratorAgent:
    """Main orchestrator using Gemini 2.5 Pro for high-level coordination"""
    
    def __init__(self, gemini_api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",  # Using latest available model
            google_api_key=gemini_api_key,
            temperature=0.1
        )
        self.parser = JsonOutputParser()
        self.performance_monitor = PerformanceMonitoringAgent()
        
    async def plan_execution(self, state: AgentState) -> AgentState:
        """Break down goal into sub-tasks and create execution plan"""
        logger.info(f"OrchestratorAgent: Planning execution for goal: {state['goal']}")
        
        # Track orchestrator execution performance
        metric = await self.performance_monitor.track_agent_execution(
            "orchestrator", "plan_execution", state
        )
        
        # Apply performance optimizations if available
        optimizations = state.get("current_optimizations", {})
        model_selection = optimizations.get("model_selection", {})
        recommended_model = model_selection.get("recommended_model", "gemini-2.0-flash-exp")
        
        # Use optimized model if suggested
        if recommended_model != "gemini-2.0-flash-exp":
            optimized_llm = ChatGoogleGenerativeAI(
                model=recommended_model,
                google_api_key=self.llm.google_api_key,
                temperature=0.1
            )
            logger.info(f"Using optimized model: {recommended_model}")
        else:
            optimized_llm = self.llm
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert web automation orchestrator. Analyze the given goal and create a detailed execution plan.
            
            Break down the goal into atomic sub-tasks that can be executed by web automation agents.
            Consider potential challenges, error scenarios, and success criteria.
            
            Return a JSON object with:
            - subtasks: List of atomic actions with descriptions and priorities
            - success_criteria: What indicates successful completion
            - risk_factors: Potential issues to watch for
            - estimated_complexity: 1-10 scale
            - timeout_strategy: How long to wait for each phase
            """),
            ("human", "Goal: {goal}\nURL: {url}\nCurrent attempt: {current_attempt}/{max_attempts}\nPerformance suggestions: {suggestions}")
        ])
        
        try:
            chain = prompt | optimized_llm | self.parser
            result = await chain.ainvoke({
                "goal": state["goal"],
                "url": state["url"],
                "current_attempt": state["current_attempt"],
                "max_attempts": state["max_attempts"],
                "suggestions": str(state.get("real_time_suggestions", []))
            })
            
            # Update state with execution plan
            state["current_phase"] = "planning_complete"
            state["status"] = "executing"
            
            # Add planning result to action history
            planning_action = {
                "timestamp": datetime.now().isoformat(),
                "agent": "orchestrator",
                "action": "plan_execution",
                "result": result,
                "success": True,
                "model_used": recommended_model,
                "optimization_applied": bool(optimizations)
            }
            state["action_history"].append(planning_action)
            
            logger.info(f"OrchestratorAgent: Created plan with {len(result.get('subtasks', []))} subtasks using {recommended_model}")
            
        except Exception as e:
            logger.error(f"OrchestratorAgent: Planning failed: {e}")
            error_entry = {
                "timestamp": datetime.now().isoformat(),
                "agent": "orchestrator",
                "error_type": "planning_failure",
                "message": str(e),
                "attempt": state["current_attempt"]
            }
            state["errors"].append(error_entry)
            state["status"] = "failed" if state["current_attempt"] >= state["max_attempts"] else "planning"
            
        return state
    
    async def coordinate_agents(self, state: AgentState) -> AgentState:
        """Coordinate communication between agents and handle recovery"""
        logger.info(f"OrchestratorAgent: Coordinating agents in phase: {state['current_phase']}")
        
        # Analyze current state and decide next action
        if state["status"] == "failed" and state["current_attempt"] < state["max_attempts"]:
            state["current_attempt"] += 1
            state["status"] = "planning"
            logger.info(f"OrchestratorAgent: Retrying attempt {state['current_attempt']}")
            
        return state

class WebAgent:
    """Web interaction agent using Gemini 2.5 Flash + Stagehand"""
    
    def __init__(self, gemini_api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=gemini_api_key,
            temperature=0.2
        )
        self.stagehand_tools = [
            "navigate", "click", "type", "scroll",
            "wait", "extract", "screenshot"
        ]
        
    async def execute_action(self, state: AgentState) -> AgentState:
        """Execute web actions using Stagehand with natural language processing"""
        logger.info(f"WebAgent: Executing action in current phase: {state['current_phase']}")
        
        if not state.get("action_history") or len(state["action_history"]) == 0:
            logger.warning("WebAgent: No action plan available")
            return state
            
        # Get the latest planning result
        planning_result = None
        for action in reversed(state["action_history"]):
            if action.get("agent") == "orchestrator" and action.get("action") == "plan_execution":
                planning_result = action.get("result")
                break
                
        if not planning_result:
            logger.error("WebAgent: No execution plan found")
            return state
            
        subtasks = planning_result.get("subtasks", [])
        current_step = len([a for a in state["action_history"] if a.get("agent") == "web_agent"])
        
        if current_step >= len(subtasks):
            state["status"] = "observing"
            return state
            
        current_subtask = subtasks[current_step]
        
        try:
            # Execute the current subtask using Stagehand
            stagehand = state["stagehand_instance"]
            
            # Take screenshot before action
            screenshot_before = await stagehand.page.screenshot()
            
            # Execute the action based on subtask type
            action_result = await self._execute_stagehand_action(
                stagehand, current_subtask, state
            )
            
            # Take screenshot after action
            screenshot_after = await stagehand.page.screenshot()
            
            # Log the action
            action_log = {
                "timestamp": datetime.now().isoformat(),
                "agent": "web_agent",
                "action": current_subtask.get("action", "unknown"),
                "target": current_subtask.get("target", {}),
                "context": {
                    "dom_before": await stagehand.page.content(),
                    "screenshot_before": screenshot_before,
                    "screenshot_after": screenshot_after
                },
                "success": action_result.get("success", False),
                "duration_ms": action_result.get("duration_ms", 0),
                "result": action_result
            }
            
            state["action_history"].append(action_log)
            state["dom_snapshots"].append(await stagehand.page.content())
            
            if action_result.get("success"):
                state["status"] = "observing"
            else:
                error_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "agent": "web_agent",
                    "error_type": "action_execution_failure",
                    "message": action_result.get("error", "Unknown error"),
                    "action": current_subtask
                }
                state["errors"].append(error_entry)
                
        except Exception as e:
            logger.error(f"WebAgent: Action execution failed: {e}")
            error_entry = {
                "timestamp": datetime.now().isoformat(),
                "agent": "web_agent",
                "error_type": "execution_exception",
                "message": str(e),
                "action": current_subtask
            }
            state["errors"].append(error_entry)
            
        return state
    
    async def _execute_stagehand_action(self, stagehand, subtask: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        """Execute specific Stagehand action based on subtask"""
        action_type = subtask.get("action", "")
        target = subtask.get("target", {})
        
        start_time = datetime.now()
        
        try:
            if action_type == "navigate":
                await stagehand.page.goto(target.get("url", state["url"]))
            elif action_type == "click":
                await stagehand.page.click(target.get("selector", ""))
            elif action_type == "type":
                await stagehand.page.fill(target.get("selector", ""), target.get("text", ""))
            elif action_type == "scroll":
                await stagehand.page.evaluate(f"window.scrollBy(0, {target.get('pixels', 300)})")
            elif action_type == "wait":
                await asyncio.sleep(target.get("seconds", 1))
            elif action_type == "screenshot":
                return {
                    "success": True,
                    "screenshot": await stagehand.page.screenshot(),
                    "duration_ms": (datetime.now() - start_time).total_seconds() * 1000
                }
            else:
                return {
                    "success": False,
                    "error": f"Unknown action type: {action_type}",
                    "duration_ms": (datetime.now() - start_time).total_seconds() * 1000
                }
                
            return {
                "success": True,
                "duration_ms": (datetime.now() - start_time).total_seconds() * 1000
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "duration_ms": (datetime.now() - start_time).total_seconds() * 1000
            }

class ObserverAgent:
    """Observer agent for real-time state analysis and progress tracking"""
    
    def __init__(self, gemini_api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=gemini_api_key,
            temperature=0.1
        )
        self.observation_history = []
        self.patterns = []
        
    async def observe_state(self, state: AgentState) -> AgentState:
        """Analyze current page state and detect progress/issues"""
        logger.info(f"ObserverAgent: Observing state in phase: {state['current_phase']}")
        
        if not state["dom_snapshots"]:
            logger.warning("ObserverAgent: No DOM snapshots available")
            return state
            
        current_dom = state["dom_snapshots"][-1]
        recent_actions = state["action_history"][-5:] if len(state["action_history"]) >= 5 else state["action_history"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert web automation observer. Analyze the current page state and recent actions.
            
            Provide analysis on:
            - Progress indicators (form completion %, navigation depth)
            - Potential issues or blockers
            - Success/failure predictions
            - Anti-pattern detection (infinite loops, stuck states)
            - Alternative path suggestions
            
            Return JSON with:
            - progress_score: 0.0-1.0
            - current_state_description: Brief description
            - issues_detected: List of potential problems
            - success_indicators: List of positive signals
            - recommendations: List of suggested next actions
            - completion_confidence: 0.0-1.0
            """),
            ("human", "Goal: {goal}\nCurrent DOM: {dom}\nRecent Actions: {actions}")
        ])
        
        try:
            chain = prompt | self.llm | JsonOutputParser()
            observation = await chain.ainvoke({
                "goal": state["goal"],
                "dom": current_dom[:5000],  # Limit DOM size for API
                "actions": str(recent_actions)
            })
            
            # Update state with observation
            observation["timestamp"] = datetime.now().isoformat()
            observation["agent"] = "observer"
            
            state["observations"].append(observation)
            state["progress_score"] = observation.get("progress_score", 0.0)
            state["completion_confidence"] = observation.get("completion_confidence", 0.0)
            
            # Determine next status based on observation
            if observation.get("completion_confidence", 0) > 0.8:
                state["status"] = "complete"
            elif observation.get("progress_score", 0) < 0.1 and len(state["action_history"]) > 10:
                state["status"] = "failed"
            else:
                state["status"] = "executing"
                
            logger.info(f"ObserverAgent: Progress score: {state['progress_score']}, Confidence: {state['completion_confidence']}")
            
        except Exception as e:
            logger.error(f"ObserverAgent: Observation failed: {e}")
            error_entry = {
                "timestamp": datetime.now().isoformat(),
                "agent": "observer",
                "error_type": "observation_failure",
                "message": str(e)
            }
            state["errors"].append(error_entry)
            
        return state

class SequenceAgent:
    """Sequence generation agent using Gemini 2.5 Pro for creating optimized Playwright sequences"""
    
    def __init__(self, gemini_api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=gemini_api_key,
            temperature=0.1
        )
        
    async def generate_sequence(self, state: AgentState) -> AgentState:
        """Generate optimized Playwright-compatible sequence from action logs"""
        logger.info(f"SequenceAgent: Generating sequence from {len(state['action_history'])} actions")
        
        # Filter successful web agent actions
        web_actions = [
            action for action in state["action_history"]
            if action.get("agent") == "web_agent" and action.get("success", False)
        ]
        
        if not web_actions:
            logger.warning("SequenceAgent: No successful web actions to process")
            return state
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert at converting web automation logs into optimized Playwright test sequences.
            
            Analyze the successful actions and create a clean, efficient Playwright test script.
            
            Focus on:
            - Removing redundant actions
            - Optimizing selectors
            - Adding appropriate waits
            - Including error handling
            - Adding assertions for verification
            
            Return JSON with:
            - playwright_script: Complete Playwright test code
            - optimizations_applied: List of improvements made
            - estimated_reliability: 0.0-1.0
            - required_setup: Any setup requirements
            """),
            ("human", "Goal: {goal}\nSuccessful Actions: {actions}\nObserver Insights: {observations}")
        ])
        
        try:
            chain = prompt | self.llm | JsonOutputParser()
            sequence_result = await chain.ainvoke({
                "goal": state["goal"],
                "actions": str(web_actions),
                "observations": str(state["observations"][-3:]) if state["observations"] else "[]"
            })
            
            state["sequence"] = sequence_result
            state["status"] = "complete"
            
            logger.info(f"SequenceAgent: Generated sequence with reliability: {sequence_result.get('estimated_reliability', 'unknown')}")
            
        except Exception as e:
            logger.error(f"SequenceAgent: Sequence generation failed: {e}")
            error_entry = {
                "timestamp": datetime.now().isoformat(),
                "agent": "sequence",
                "error_type": "sequence_generation_failure",
                "message": str(e)
            }
            state["errors"].append(error_entry)
            state["status"] = "failed"
            
        return state

class RetryStrategy:
    """Intelligent retry logic based on error type and context"""
    
    @staticmethod
    def should_retry(error_type: str, attempt: int, context: Dict[str, Any]) -> bool:
        """Determine if action should be retried based on error type and context"""
        max_retries = {
            "element_not_found": 2,
            "timeout": 3,
            "captcha_failed": 3,
            "network_error": 2,
            "execution_exception": 1
        }
        
        return attempt < max_retries.get(error_type, 1)

class SuccessDetector:
    """Success pattern recognition for goal completion detection"""
    
    def __init__(self, gemini_api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=gemini_api_key,
            temperature=0.1
        )
        
    async def detect_completion(self, dom: str, goal: str, action_history: List[Dict[str, Any]]) -> float:
        """Use LLM to determine completion confidence"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Analyze if the given goal has been completed based on the current page state and action history.
            
            Look for success indicators like:
            - Redirects to success/confirmation pages
            - Thank you messages
            - Completion confirmations
            - Goal-specific success patterns
            
            Return a confidence score from 0.0 to 1.0.
            """),
            ("human", "Goal: {goal}\nCurrent Page: {dom}\nActions: {actions}")
        ])
        
        try:
            chain = prompt | self.llm
            result = await chain.ainvoke({
                "goal": goal,
                "dom": dom[:3000],
                "actions": str(action_history[-5:])
            })
            
            # Extract confidence score from response
            confidence_text = result.content.lower()
            if "1.0" in confidence_text or "100%" in confidence_text:
                return 1.0
            elif "0.9" in confidence_text or "90%" in confidence_text:
                return 0.9
            elif "0.8" in confidence_text or "80%" in confidence_text:
                return 0.8
            elif "0.0" in confidence_text or "0%" in confidence_text:
                return 0.0
            else:
                return 0.5  # Default moderate confidence
                
        except Exception as e:
            logger.error(f"SuccessDetector: Error detecting completion: {e}")
            return 0.0

def create_enhanced_recording_graph() -> StateGraph:
    """Create the enhanced LangGraph workflow with all agents"""
    
    # Initialize agents
    orchestrator = OrchestratorAgent()
    web_agent = WebAgent()
    observer = ObserverAgent()
    sequence_agent = SequenceAgent()
    performance_monitor = PerformanceMonitoringAgent()
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("orchestrator", orchestrator.plan_execution)
    workflow.add_node("web_agent", web_agent.execute_web_action)
    workflow.add_node("observer", observer.analyze_state)
    workflow.add_node("sequence_generator", sequence_agent.generate_sequence)
    workflow.add_node("performance_monitor", performance_monitor.monitor_execution)
    
    # Define the workflow edges
    workflow.add_edge(START, "orchestrator")
    
    # Conditional routing from orchestrator
    workflow.add_conditional_edges(
        "orchestrator",
        lambda state: state.get("next_agent", "web_agent"),
        {
            "web_agent": "web_agent",
            "observer": "observer",
            "sequence_generator": "sequence_generator",
            "performance_monitor": "performance_monitor",
            "END": END
        }
    )
    
    # From web_agent, always go to observer
    workflow.add_edge("web_agent", "observer")
    
    # From observer, decide next step
    workflow.add_conditional_edges(
        "observer",
        lambda state: "performance_monitor" if state.get("current_step", 0) % 3 == 0 else ("sequence_generator" if state.get("should_generate_sequence", False) else "orchestrator"),
        {
            "orchestrator": "orchestrator",
            "sequence_generator": "sequence_generator",
            "performance_monitor": "performance_monitor"
        }
    )
    
    # From performance_monitor, decide next step
    workflow.add_conditional_edges(
        "performance_monitor",
        lambda state: "sequence_generator" if state.get("should_generate_sequence", False) else "orchestrator",
        {
            "orchestrator": "orchestrator",
            "sequence_generator": "sequence_generator"
        }
    )
    
    # From sequence_generator, end the workflow
    workflow.add_edge("sequence_generator", END)
    
    return workflow.compile()

def create_recording_graph(gemini_api_key: str) -> StateGraph:
    """Create the enhanced LangGraph workflow for multi-agent recording"""
    
    # Initialize agents
    orchestrator = OrchestratorAgent(gemini_api_key)
    web_agent = WebAgent(gemini_api_key)
    observer = ObserverAgent(gemini_api_key)
    sequence_agent = SequenceAgent(gemini_api_key)
    performance_monitor = PerformanceMonitoringAgent()
    
    # Create workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("orchestrator", orchestrator.plan_execution)
    workflow.add_node("coordinator", orchestrator.coordinate_agents)
    workflow.add_node("web_agent", web_agent.execute_action)
    workflow.add_node("observer", observer.observe_state)
    workflow.add_node("sequence_generator", sequence_agent.generate_sequence)
    workflow.add_node("performance_monitor", performance_monitor.monitor_execution)
    
    # Define conditional routing
    def should_continue_execution(state: AgentState) -> str:
        """Determine next step based on current state"""
        status = state.get("status", "planning")
        
        if status == "planning":
            return "orchestrator"
        elif status == "executing":
            return "web_agent"
        elif status == "observing":
            return "observer"
        elif status == "complete":
            return "sequence_generator"
        elif status == "failed":
            if state.get("current_attempt", 0) < state.get("max_attempts", 3):
                return "coordinator"
            else:
                return END
        else:
            return "orchestrator"
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "orchestrator",
        should_continue_execution,
        {
            "web_agent": "web_agent",
            "coordinator": "coordinator",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "coordinator",
        should_continue_execution,
        {
            "orchestrator": "orchestrator",
            END: END
        }
    )
    
    workflow.add_edge("web_agent", "observer")
    
    workflow.add_conditional_edges(
        "observer",
        lambda state: "performance_monitor" if state.get("current_step", 0) % 3 == 0 else should_continue_execution(state),
        {
            "web_agent": "web_agent",
            "sequence_generator": "sequence_generator",
            "coordinator": "coordinator",
            "performance_monitor": "performance_monitor",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "performance_monitor",
        should_continue_execution,
        {
            "web_agent": "web_agent",
            "sequence_generator": "sequence_generator",
            "coordinator": "coordinator",
            END: END
        }
    )
    
    workflow.add_edge("sequence_generator", END)
    
    # Set entry point
    workflow.set_entry_point("orchestrator")
    
    return workflow.compile()