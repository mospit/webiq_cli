import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from langgraph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger(__name__)

class AgentState(Enum):
    """States for the multi-agent workflow"""
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    MONITORING = "monitoring"
    RECORDING = "recording"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class WebAutomationState:
    """State object for the multi-agent web automation workflow"""
    url: str
    goal: str
    current_page_state: Dict[str, Any]
    action_plan: List[Dict[str, Any]]
    executed_actions: List[Dict[str, Any]]
    screenshots: List[str]
    goal_progress: List[Dict[str, Any]]
    current_step: int
    agent_state: AgentState
    error_message: Optional[str]
    completion_percentage: float
    stagehand_instance: Any
    steel_session: Any
    options: Dict[str, Any]
    messages: List[Any]

class WebAnalysisAgent:
    """Agent responsible for analyzing web pages and understanding goal context"""
    
    def __init__(self, gemini_api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=gemini_api_key,
            temperature=0.1
        )
        self.parser = JsonOutputParser()
        
    async def analyze_page(self, state: WebAutomationState) -> WebAutomationState:
        """Analyze the current page state and understand goal requirements"""
        logger.info(f"WebAnalysisAgent: Analyzing page for goal: {state.goal}")
        
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a web analysis expert. Analyze the current webpage to understand how to accomplish the given goal.
            
            Provide a detailed JSON analysis with:
            1. goal_steps: Array of specific steps needed to achieve the goal
            2. required_elements: Elements that must be interacted with
            3. optional_elements: Elements that might be helpful
            4. obstacles: Potential challenges or anti-bot measures
            5. success_indicators: How to know when the goal is achieved
            6. estimated_difficulty: "simple", "medium", or "complex"
            7. success_probability: Float between 0 and 1
            """),
            ("human", "Goal: {goal}\nCurrent URL: {url}\nPage State: {page_state}")
        ])
        
        try:
            # Get current page state from Stagehand
            if state.stagehand_instance:
                page_content = await state.stagehand_instance.page.extract({
                    "instruction": "Extract all visible text, form fields, buttons, and interactive elements",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "text_content": {"type": "string"},
                            "form_fields": {"type": "array"},
                            "buttons": {"type": "array"},
                            "links": {"type": "array"}
                        }
                    }
                })
                state.current_page_state = page_content
            
            # Analyze with LLM
            chain = analysis_prompt | self.llm | self.parser
            analysis_result = await chain.ainvoke({
                "goal": state.goal,
                "url": state.url,
                "page_state": str(state.current_page_state)
            })
            
            state.messages.append(AIMessage(content=f"Page analysis completed: {analysis_result}"))
            state.agent_state = AgentState.PLANNING
            logger.info(f"WebAnalysisAgent: Analysis completed with difficulty: {analysis_result.get('estimated_difficulty')}")
            
        except Exception as e:
            logger.error(f"WebAnalysisAgent: Error during analysis: {e}")
            state.error_message = str(e)
            state.agent_state = AgentState.ERROR
            
        return state

class ActionPlanningAgent:
    """Agent responsible for creating detailed action plans based on analysis"""
    
    def __init__(self, gemini_api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=gemini_api_key,
            temperature=0.2
        )
        self.parser = JsonOutputParser()
        
    async def create_action_plan(self, state: WebAutomationState) -> WebAutomationState:
        """Create a detailed step-by-step action plan"""
        logger.info(f"ActionPlanningAgent: Creating action plan for goal: {state.goal}")
        
        planning_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert web automation planner. Create a detailed, executable action plan.
            
            Return a JSON array of actions, each with:
            1. action_type: "click", "type", "wait", "navigate", "extract", "verify"
            2. target: CSS selector or description of target element
            3. value: Text to type or expected value (if applicable)
            4. description: Human-readable description of the action
            5. success_criteria: How to verify this step succeeded
            6. fallback_actions: Alternative actions if this fails
            7. estimated_time: Expected time in seconds
            """),
            ("human", "Goal: {goal}\nPage Analysis: {analysis}\nCurrent State: {page_state}")
        ])
        
        try:
            # Extract analysis from previous messages
            analysis_data = None
            for msg in reversed(state.messages):
                if isinstance(msg, AIMessage) and "analysis completed" in msg.content:
                    # Parse the analysis from the message
                    analysis_data = msg.content
                    break
            
            chain = planning_prompt | self.llm | self.parser
            action_plan = await chain.ainvoke({
                "goal": state.goal,
                "analysis": analysis_data or "No previous analysis found",
                "page_state": str(state.current_page_state)
            })
            
            state.action_plan = action_plan if isinstance(action_plan, list) else [action_plan]
            state.current_step = 0
            state.messages.append(AIMessage(content=f"Action plan created with {len(state.action_plan)} steps"))
            state.agent_state = AgentState.EXECUTING
            logger.info(f"ActionPlanningAgent: Created plan with {len(state.action_plan)} steps")
            
        except Exception as e:
            logger.error(f"ActionPlanningAgent: Error during planning: {e}")
            state.error_message = str(e)
            state.agent_state = AgentState.ERROR
            
        return state

class ActionExecutionAgent:
    """Agent responsible for executing planned actions"""
    
    def __init__(self, gemini_api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=gemini_api_key,
            temperature=0.1
        )
        
    async def execute_next_action(self, state: WebAutomationState) -> WebAutomationState:
        """Execute the next action in the plan"""
        if state.current_step >= len(state.action_plan):
            state.agent_state = AgentState.COMPLETED
            return state
            
        current_action = state.action_plan[state.current_step]
        logger.info(f"ActionExecutionAgent: Executing step {state.current_step + 1}: {current_action.get('description')}")
        
        try:
            action_type = current_action.get('action_type')
            target = current_action.get('target')
            value = current_action.get('value')
            
            if not state.stagehand_instance:
                raise Exception("Stagehand instance not available")
                
            # Execute the action based on type
            if action_type == "click":
                await state.stagehand_instance.page.act({"action": "click", "coordinate": target})
            elif action_type == "type":
                await state.stagehand_instance.page.act({"action": "type", "text": value, "coordinate": target})
            elif action_type == "wait":
                await asyncio.sleep(float(value or 1))
            elif action_type == "navigate":
                await state.stagehand_instance.page.goto(value)
            elif action_type == "extract":
                extracted_data = await state.stagehand_instance.page.extract({
                    "instruction": current_action.get('description'),
                    "schema": {"type": "object", "properties": {"result": {"type": "string"}}}
                })
                current_action['result'] = extracted_data
            
            # Record the executed action
            executed_action = {
                **current_action,
                'executed_at': asyncio.get_event_loop().time(),
                'status': 'success'
            }
            state.executed_actions.append(executed_action)
            state.current_step += 1
            
            # Update completion percentage
            state.completion_percentage = (state.current_step / len(state.action_plan)) * 100
            
            state.messages.append(AIMessage(content=f"Executed action: {current_action.get('description')}"))
            
            # Check if we should continue executing or monitor
            if state.current_step < len(state.action_plan):
                state.agent_state = AgentState.MONITORING  # Monitor before next action
            else:
                state.agent_state = AgentState.COMPLETED
                
            logger.info(f"ActionExecutionAgent: Successfully executed step {state.current_step}")
            
        except Exception as e:
            logger.error(f"ActionExecutionAgent: Error executing action: {e}")
            # Try fallback actions if available
            fallback_actions = current_action.get('fallback_actions', [])
            if fallback_actions:
                logger.info(f"ActionExecutionAgent: Trying fallback actions")
                # Add fallback actions to the plan
                state.action_plan[state.current_step:state.current_step] = fallback_actions
            else:
                state.error_message = str(e)
                state.agent_state = AgentState.ERROR
                
        return state

class MonitoringAgent:
    """Agent responsible for monitoring progress and recording state"""
    
    def __init__(self, gemini_api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=gemini_api_key,
            temperature=0.1
        )
        self.parser = JsonOutputParser()
        
    async def monitor_progress(self, state: WebAutomationState) -> WebAutomationState:
        """Monitor current progress and decide next steps"""
        logger.info(f"MonitoringAgent: Monitoring progress for goal: {state.goal}")
        
        try:
            # Take screenshot
            if state.stagehand_instance:
                screenshot_path = f"screenshot_{len(state.screenshots)}.png"
                # In real implementation, save actual screenshot
                state.screenshots.append(screenshot_path)
                
                # Analyze current progress
                progress_analysis = await self._analyze_progress(state)
                state.goal_progress.append(progress_analysis)
                
                # Decide next action based on progress
                if progress_analysis.get('goal_achieved', False):
                    state.agent_state = AgentState.COMPLETED
                elif progress_analysis.get('needs_replanning', False):
                    state.agent_state = AgentState.PLANNING
                elif progress_analysis.get('critical_error', False):
                    state.agent_state = AgentState.ERROR
                    state.error_message = progress_analysis.get('error_message')
                else:
                    state.agent_state = AgentState.EXECUTING
                    
            state.messages.append(AIMessage(content=f"Progress monitoring completed: {state.completion_percentage}%"))
            logger.info(f"MonitoringAgent: Progress analysis completed")
            
        except Exception as e:
            logger.error(f"MonitoringAgent: Error during monitoring: {e}")
            state.error_message = str(e)
            state.agent_state = AgentState.ERROR
            
        return state
        
    async def _analyze_progress(self, state: WebAutomationState) -> Dict[str, Any]:
        """Analyze current progress toward goal completion"""
        progress_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Analyze the current state to determine progress toward the goal.
            
            Return JSON with:
            1. goal_achieved: Boolean - is the goal fully completed?
            2. progress_percentage: Float 0-100 - how close are we?
            3. next_recommended_action: String - what should happen next?
            4. needs_replanning: Boolean - does the plan need to be revised?
            5. critical_error: Boolean - is there a blocking error?
            6. error_message: String - description of any error
            7. success_indicators_met: Array - which success criteria are met
            """),
            ("human", "Goal: {goal}\nExecuted Actions: {actions}\nCurrent Step: {step}/{total}")
        ])
        
        chain = progress_prompt | self.llm | self.parser
        return await chain.ainvoke({
            "goal": state.goal,
            "actions": str(state.executed_actions[-3:]),  # Last 3 actions
            "step": state.current_step,
            "total": len(state.action_plan)
        })

class WebAutomationOrchestrator:
    """Main orchestrator for the multi-agent web automation workflow"""
    
    def __init__(self, gemini_api_key: str, steel_api_key: str):
        self.gemini_api_key = gemini_api_key
        self.steel_api_key = steel_api_key
        
        # Initialize agents
        self.analysis_agent = WebAnalysisAgent(gemini_api_key)
        self.planning_agent = ActionPlanningAgent(gemini_api_key)
        self.execution_agent = ActionExecutionAgent(gemini_api_key)
        self.monitoring_agent = MonitoringAgent(gemini_api_key)
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for multi-agent coordination"""
        workflow = StateGraph(WebAutomationState)
        
        # Add nodes for each agent
        workflow.add_node("analyze", self.analysis_agent.analyze_page)
        workflow.add_node("plan", self.planning_agent.create_action_plan)
        workflow.add_node("execute", self.execution_agent.execute_next_action)
        workflow.add_node("monitor", self.monitoring_agent.monitor_progress)
        
        # Define the workflow edges
        workflow.set_entry_point("analyze")
        
        # Conditional routing based on agent state
        def route_next_step(state: WebAutomationState) -> str:
            if state.agent_state == AgentState.PLANNING:
                return "plan"
            elif state.agent_state == AgentState.EXECUTING:
                return "execute"
            elif state.agent_state == AgentState.MONITORING:
                return "monitor"
            elif state.agent_state == AgentState.COMPLETED:
                return END
            elif state.agent_state == AgentState.ERROR:
                return END
            else:
                return "analyze"
                
        workflow.add_conditional_edges(
            "analyze",
            route_next_step,
            {
                "plan": "plan",
                "execute": "execute",
                "monitor": "monitor",
                END: END
            }
        )
        
        workflow.add_conditional_edges(
            "plan",
            route_next_step,
            {
                "execute": "execute",
                "analyze": "analyze",
                END: END
            }
        )
        
        workflow.add_conditional_edges(
            "execute",
            route_next_step,
            {
                "monitor": "monitor",
                "execute": "execute",
                END: END
            }
        )
        
        workflow.add_conditional_edges(
            "monitor",
            route_next_step,
            {
                "execute": "execute",
                "plan": "plan",
                "analyze": "analyze",
                END: END
            }
        )
        
        return workflow.compile()
        
    async def run_automation(self, url: str, goal: str, stagehand_instance: Any, 
                           steel_session: Any, options: Dict[str, Any]) -> WebAutomationState:
        """Run the complete multi-agent web automation workflow"""
        logger.info(f"WebAutomationOrchestrator: Starting automation for goal: {goal}")
        
        # Initialize state
        initial_state = WebAutomationState(
            url=url,
            goal=goal,
            current_page_state={},
            action_plan=[],
            executed_actions=[],
            screenshots=[],
            goal_progress=[],
            current_step=0,
            agent_state=AgentState.ANALYZING,
            error_message=None,
            completion_percentage=0.0,
            stagehand_instance=stagehand_instance,
            steel_session=steel_session,
            options=options,
            messages=[HumanMessage(content=f"Starting automation for goal: {goal}")]
        )
        
        try:
            # Run the workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            logger.info(f"WebAutomationOrchestrator: Automation completed with state: {final_state.agent_state}")
            return final_state
            
        except Exception as e:
            logger.error(f"WebAutomationOrchestrator: Error during automation: {e}")
            initial_state.error_message = str(e)
            initial_state.agent_state = AgentState.ERROR
            return initial_state