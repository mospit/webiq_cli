#!/usr/bin/env python3
"""
Real-Time Adaptation System - Example Usage and Integration Guide

This file demonstrates how to integrate and use the sophisticated real-time adaptation system
with the WebIQ CLI framework. It shows practical examples of:

1. System initialization and configuration
2. Integration with existing WebIQ components
3. Custom adaptation strategies
4. Monitoring and metrics collection
5. Context-aware optimizations
6. Predictive interventions
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# Import the real-time adaptation system components
from .real_time_adaptation_integration import (
    RealTimeAdaptationSystem, SystemConfiguration, AdaptationSystemMetrics,
    create_real_time_adaptation_system, setup_and_start_adaptation_system
)
from .real_time_adaptation_system import (
    AdaptationEvent, AdaptationTrigger, AdaptationSeverity, SystemState,
    SystemHealthStatus, InterventionType
)
from .intelligent_context_adapter import (
    ExecutionContext, ContextType, SiteCategory, UserPriority
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebIQAdaptationIntegration:
    """Integration layer between WebIQ CLI and the real-time adaptation system"""
    
    def __init__(self, webiq_config: Optional[Dict[str, Any]] = None):
        self.webiq_config = webiq_config or {}
        self.adaptation_system: Optional[RealTimeAdaptationSystem] = None
        self.execution_history: List[Dict[str, Any]] = []
        
        # Configure adaptation system based on WebIQ settings
        self.adaptation_config = self._create_adaptation_config()
    
    def _create_adaptation_config(self) -> SystemConfiguration:
        """Create adaptation system configuration based on WebIQ settings"""
        
        # Extract relevant settings from WebIQ config
        monitoring_interval = self.webiq_config.get("monitoring_interval", 1.0)
        max_retries = self.webiq_config.get("max_retries", 3)
        enable_predictions = self.webiq_config.get("enable_predictive_interventions", True)
        enable_context_adaptation = self.webiq_config.get("enable_context_adaptation", True)
        
        return SystemConfiguration(
            monitoring_interval=monitoring_interval,
            adaptation_threshold=0.7,
            prediction_horizon=10,
            context_analysis_interval=5.0,
            max_concurrent_adaptations=3,
            rollback_timeout=30.0,
            learning_enabled=True,
            proactive_interventions_enabled=enable_predictions,
            context_adaptation_enabled=enable_context_adaptation
        )
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the adaptation system integration"""
        
        try:
            # Create and start the adaptation system
            self.adaptation_system = await setup_and_start_adaptation_system(self.adaptation_config)
            
            # Register callbacks for WebIQ integration
            self.adaptation_system.register_adaptation_callback(self._on_adaptation_applied)
            self.adaptation_system.register_prediction_callback(self._on_prediction_made)
            self.adaptation_system.register_context_callback(self._on_context_changed)
            
            logger.info("WebIQ adaptation integration initialized successfully")
            
            return {
                "status": "initialized",
                "system_status": self.adaptation_system.get_system_status(),
                "configuration": self.adaptation_config.__dict__
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize adaptation integration: {e}")
            raise
    
    async def execute_webiq_task(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a WebIQ task with real-time adaptation"""
        
        if not self.adaptation_system:
            raise RuntimeError("Adaptation system not initialized")
        
        task_id = task_config.get("task_id", f"task_{int(time.time())}")
        
        try:
            # Create execution context
            context = await self._create_execution_context(task_config)
            
            # Notify adaptation system of new context
            await self._notify_context_change(context, task_config)
            
            # Execute the task with monitoring
            start_time = time.time()
            result = await self._execute_task_with_monitoring(task_config, context)
            execution_time = time.time() - start_time
            
            # Record execution outcome
            execution_record = {
                "task_id": task_id,
                "context": context,
                "result": result,
                "execution_time": execution_time,
                "timestamp": datetime.now()
            }
            
            self.execution_history.append(execution_record)
            
            # Learn from execution outcome
            await self._learn_from_execution(execution_record)
            
            return {
                "task_id": task_id,
                "status": "completed",
                "result": result,
                "execution_time": execution_time,
                "adaptations_applied": await self._get_task_adaptations(task_id),
                "system_health": await self.adaptation_system.get_system_health()
            }
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            
            # Create failure adaptation event
            failure_event = AdaptationEvent(
                event_id=f"failure_{task_id}",
                trigger=AdaptationTrigger.ERROR_RATE_HIGH,
                severity=AdaptationSeverity.HIGH,
                confidence_score=0.9,
                suggested_adaptations=[
                    {
                        "type": "error_handling_enhancement",
                        "action": "increase_retry_attempts",
                        "parameters": {"additional_retries": 2}
                    }
                ],
                context={"task_id": task_id, "error": str(e)}
            )
            
            await self.adaptation_system.force_adaptation(failure_event)
            
            raise
    
    async def _create_execution_context(self, task_config: Dict[str, Any]) -> ExecutionContext:
        """Create execution context from task configuration"""
        
        # Determine context type
        task_type = task_config.get("type", "website_interaction")
        context_type = self._map_task_type_to_context(task_type)
        
        # Determine site category
        url = task_config.get("url", "")
        site_category = self._determine_site_category_from_url(url)
        
        # Determine user priority
        priority = task_config.get("priority", "comprehensive")
        user_priority = UserPriority(priority.lower()) if priority.lower() in [p.value for p in UserPriority] else UserPriority.COMPREHENSIVE
        
        # Extract constraints
        time_limit = task_config.get("time_limit")
        time_constraints = timedelta(seconds=time_limit) if time_limit else None
        
        return ExecutionContext(
            context_id=f"ctx_{task_config.get('task_id', int(time.time()))}",
            context_type=context_type,
            site_category=site_category,
            user_priority=user_priority,
            complexity_level=task_config.get("complexity", "medium"),
            time_constraints=time_constraints,
            resource_constraints=task_config.get("resource_constraints", {}),
            quality_requirements=task_config.get("quality_requirements", {}),
            environmental_factors=task_config.get("environmental_factors", {})
        )
    
    def _map_task_type_to_context(self, task_type: str) -> ContextType:
        """Map WebIQ task type to context type"""
        
        mapping = {
            "web_scraping": ContextType.DATA_EXTRACTION,
            "form_filling": ContextType.FORM_SUBMISSION,
            "navigation": ContextType.NAVIGATION,
            "content_analysis": ContextType.CONTENT_ANALYSIS,
            "automation": ContextType.AUTOMATION_TASK,
            "testing": ContextType.TESTING,
            "monitoring": ContextType.MONITORING
        }
        
        return mapping.get(task_type, ContextType.WEBSITE_INTERACTION)
    
    def _determine_site_category_from_url(self, url: str) -> Optional[SiteCategory]:
        """Determine site category from URL"""
        
        url_lower = url.lower()
        
        if any(keyword in url_lower for keyword in ["shop", "store", "buy", "cart", "amazon", "ebay"]):
            return SiteCategory.E_COMMERCE
        elif any(keyword in url_lower for keyword in ["facebook", "twitter", "instagram", "linkedin"]):
            return SiteCategory.SOCIAL_MEDIA
        elif any(keyword in url_lower for keyword in ["news", "cnn", "bbc", "reuters"]):
            return SiteCategory.NEWS_PORTAL
        elif any(keyword in url_lower for keyword in ["gov", "government"]):
            return SiteCategory.GOVERNMENT
        elif any(keyword in url_lower for keyword in ["edu", "university", "school"]):
            return SiteCategory.EDUCATIONAL
        elif any(keyword in url_lower for keyword in ["bank", "finance", "trading"]):
            return SiteCategory.FINANCIAL
        elif any(keyword in url_lower for keyword in ["health", "medical", "hospital"]):
            return SiteCategory.HEALTHCARE
        elif any(keyword in url_lower for keyword in ["tech", "software", "api", "github"]):
            return SiteCategory.TECHNOLOGY
        
        return SiteCategory.CORPORATE  # Default
    
    async def _notify_context_change(self, context: ExecutionContext, task_config: Dict[str, Any]):
        """Notify adaptation system of context change"""
        
        context_data = {
            "context_id": context.context_id,
            "task_type": task_config.get("type"),
            "url": task_config.get("url"),
            "planned_actions": task_config.get("actions", []),
            "user_priority": context.user_priority.value if context.user_priority else None,
            "complexity_level": context.complexity_level,
            "time_limit": task_config.get("time_limit"),
            "quality_requirements": context.quality_requirements,
            "resource_constraints": context.resource_constraints
        }
        
        # This will trigger context analysis in the adaptation system
        await self.adaptation_system.context_adapter.analyze_context_change(context_data)
    
    async def _execute_task_with_monitoring(self, task_config: Dict[str, Any], 
                                          context: ExecutionContext) -> Dict[str, Any]:
        """Execute task with real-time monitoring and adaptation"""
        
        # This is where you would integrate with actual WebIQ task execution
        # For demonstration, we'll simulate task execution with monitoring
        
        task_type = task_config.get("type", "website_interaction")
        
        if task_type == "web_scraping":
            return await self._simulate_web_scraping_task(task_config, context)
        elif task_type == "form_filling":
            return await self._simulate_form_filling_task(task_config, context)
        elif task_type == "navigation":
            return await self._simulate_navigation_task(task_config, context)
        else:
            return await self._simulate_generic_task(task_config, context)
    
    async def _simulate_web_scraping_task(self, task_config: Dict[str, Any], 
                                        context: ExecutionContext) -> Dict[str, Any]:
        """Simulate web scraping task with monitoring"""
        
        url = task_config.get("url", "")
        selectors = task_config.get("selectors", [])
        
        logger.info(f"Starting web scraping task for {url}")
        
        # Simulate progressive execution with monitoring points
        results = {}
        
        for i, selector in enumerate(selectors):
            # Simulate processing time
            await asyncio.sleep(0.5)
            
            # Simulate potential issues that might trigger adaptations
            if i == 2 and len(selectors) > 3:  # Simulate slowdown
                await asyncio.sleep(2.0)  # This might trigger timeout adaptation
            
            # Simulate data extraction
            results[f"element_{i}"] = f"extracted_data_from_{selector}"
            
            # Report progress (this could trigger monitoring events)
            progress = (i + 1) / len(selectors)
            logger.debug(f"Scraping progress: {progress:.1%}")
        
        return {
            "status": "success",
            "data_extracted": results,
            "elements_found": len(results),
            "success_rate": 0.95,  # Simulate 95% success rate
            "response_time": 2.5
        }
    
    async def _simulate_form_filling_task(self, task_config: Dict[str, Any], 
                                        context: ExecutionContext) -> Dict[str, Any]:
        """Simulate form filling task with monitoring"""
        
        form_data = task_config.get("form_data", {})
        
        logger.info(f"Starting form filling task with {len(form_data)} fields")
        
        # Simulate form filling with potential issues
        filled_fields = []
        
        for field_name, field_value in form_data.items():
            await asyncio.sleep(0.3)  # Simulate field filling time
            
            # Simulate occasional field filling issues
            if "email" in field_name.lower() and "@" not in str(field_value):
                # This might trigger error handling adaptation
                logger.warning(f"Invalid email format for field {field_name}")
                continue
            
            filled_fields.append(field_name)
        
        success_rate = len(filled_fields) / len(form_data) if form_data else 1.0
        
        return {
            "status": "success" if success_rate > 0.8 else "partial_success",
            "fields_filled": filled_fields,
            "success_rate": success_rate,
            "response_time": 1.8
        }
    
    async def _simulate_navigation_task(self, task_config: Dict[str, Any], 
                                      context: ExecutionContext) -> Dict[str, Any]:
        """Simulate navigation task with monitoring"""
        
        steps = task_config.get("navigation_steps", [])
        
        logger.info(f"Starting navigation task with {len(steps)} steps")
        
        completed_steps = []
        
        for i, step in enumerate(steps):
            await asyncio.sleep(0.4)  # Simulate navigation time
            
            # Simulate potential navigation issues
            if "click" in step.get("action", "").lower():
                # Simulate element not found occasionally
                if i == len(steps) - 1:  # Last step might fail
                    logger.warning(f"Element not found for step {i}")
                    break
            
            completed_steps.append(step)
        
        success_rate = len(completed_steps) / len(steps) if steps else 1.0
        
        return {
            "status": "success" if success_rate == 1.0 else "partial_success",
            "steps_completed": len(completed_steps),
            "total_steps": len(steps),
            "success_rate": success_rate,
            "response_time": 2.1
        }
    
    async def _simulate_generic_task(self, task_config: Dict[str, Any], 
                                   context: ExecutionContext) -> Dict[str, Any]:
        """Simulate generic task execution"""
        
        await asyncio.sleep(1.0)  # Simulate execution time
        
        return {
            "status": "success",
            "result": "Task completed successfully",
            "success_rate": 0.92,
            "response_time": 1.0
        }
    
    async def _learn_from_execution(self, execution_record: Dict[str, Any]):
        """Learn from task execution outcome"""
        
        if not self.adaptation_system:
            return
        
        context = execution_record["context"]
        result = execution_record["result"]
        
        # Extract strategy used (this would come from actual execution)
        strategy_used = {
            "model_used": "gemini-2.5-flash",  # Example
            "timeout_setting": 30,
            "retry_attempts": 3,
            "verification_level": "medium"
        }
        
        # Extract outcome metrics
        outcome = {
            "success_rate": result.get("success_rate", 0.0),
            "response_time": result.get("response_time", 0.0),
            "accuracy": result.get("accuracy", result.get("success_rate", 0.0)),
            "errors_encountered": result.get("errors", [])
        }
        
        # Learn through context adapter
        await self.adaptation_system.context_adapter.learn_from_execution_outcome(
            context, strategy_used, outcome
        )
    
    async def _get_task_adaptations(self, task_id: str) -> List[Dict[str, Any]]:
        """Get adaptations applied during task execution"""
        
        if not self.adaptation_system:
            return []
        
        # This would track adaptations applied during the specific task
        # For now, return recent adaptations
        active_adaptations = self.adaptation_system.get_active_adaptations()
        
        return [
            {
                "adaptation_id": aid,
                "type": info["event"].trigger.value,
                "status": info["status"]
            }
            for aid, info in active_adaptations.items()
        ]
    
    # Callback methods for adaptation system events
    
    async def _on_adaptation_applied(self, event: AdaptationEvent, result: Dict[str, Any]):
        """Handle adaptation applied event"""
        
        logger.info(f"Adaptation applied: {event.trigger.value} - {result.get('status')}")
        
        # You could integrate this with WebIQ's logging/monitoring system
        # For example, update UI, send notifications, etc.
    
    async def _on_prediction_made(self, prediction: Dict[str, Any]):
        """Handle prediction made event"""
        
        logger.info(f"Prediction made: {prediction.get('type')} - confidence: {prediction.get('confidence', 0)}")
        
        # You could use predictions to update UI or alert users
    
    async def _on_context_changed(self, context_data: Dict[str, Any]):
        """Handle context change event"""
        
        logger.info(f"Context changed: {context_data.get('context_id')}")
        
        # You could update UI to reflect context changes
    
    # Public API methods
    
    async def get_adaptation_metrics(self) -> Dict[str, Any]:
        """Get adaptation system metrics"""
        
        if not self.adaptation_system:
            return {"error": "Adaptation system not initialized"}
        
        metrics = self.adaptation_system.get_system_metrics()
        health = await self.adaptation_system.get_system_health()
        
        return {
            "metrics": metrics.__dict__,
            "health": health,
            "execution_history_count": len(self.execution_history)
        }
    
    async def get_system_recommendations(self) -> Dict[str, Any]:
        """Get system optimization recommendations"""
        
        if not self.adaptation_system:
            return {"error": "Adaptation system not initialized"}
        
        # Analyze recent execution history
        recent_executions = self.execution_history[-10:]  # Last 10 executions
        
        recommendations = []
        
        if recent_executions:
            avg_response_time = sum(e["execution_time"] for e in recent_executions) / len(recent_executions)
            avg_success_rate = sum(e["result"].get("success_rate", 0) for e in recent_executions) / len(recent_executions)
            
            if avg_response_time > 5.0:
                recommendations.append({
                    "type": "performance",
                    "message": "Consider enabling speed-optimized strategies",
                    "action": "Set user_priority to 'speed' for time-sensitive tasks"
                })
            
            if avg_success_rate < 0.8:
                recommendations.append({
                    "type": "reliability",
                    "message": "Consider enabling reliability-focused strategies",
                    "action": "Set user_priority to 'reliability' and increase retry attempts"
                })
        
        return {
            "recommendations": recommendations,
            "strategy_effectiveness": self.adaptation_system.context_adapter.get_strategy_effectiveness_scores()
        }
    
    async def shutdown(self):
        """Shutdown the adaptation integration"""
        
        if self.adaptation_system:
            await self.adaptation_system.stop()
            logger.info("WebIQ adaptation integration shutdown complete")

# Example usage and demonstration
async def demonstrate_adaptation_system():
    """Demonstrate the real-time adaptation system"""
    
    logger.info("=== Real-Time Adaptation System Demonstration ===")
    
    # Initialize the integration
    webiq_config = {
        "monitoring_interval": 0.5,
        "max_retries": 5,
        "enable_predictive_interventions": True,
        "enable_context_adaptation": True
    }
    
    integration = WebIQAdaptationIntegration(webiq_config)
    await integration.initialize()
    
    try:
        # Demonstrate different types of tasks
        
        # 1. Web scraping task (data extraction context)
        logger.info("\n--- Demonstrating Web Scraping Task ---")
        scraping_task = {
            "task_id": "scrape_001",
            "type": "web_scraping",
            "url": "https://example-ecommerce.com/products",
            "selectors": ["h1.title", ".price", ".description", ".reviews", ".availability"],
            "priority": "accuracy",
            "complexity": "medium",
            "quality_requirements": {"accuracy": 0.95}
        }
        
        result1 = await integration.execute_webiq_task(scraping_task)
        logger.info(f"Scraping task result: {result1['status']} in {result1['execution_time']:.2f}s")
        
        # 2. Form filling task (form submission context)
        logger.info("\n--- Demonstrating Form Filling Task ---")
        form_task = {
            "task_id": "form_001",
            "type": "form_filling",
            "url": "https://government-portal.gov/application",
            "form_data": {
                "first_name": "John",
                "last_name": "Doe",
                "email": "john.doe@example.com",
                "phone": "555-1234",
                "address": "123 Main St"
            },
            "priority": "reliability",
            "complexity": "high",
            "time_limit": 120
        }
        
        result2 = await integration.execute_webiq_task(form_task)
        logger.info(f"Form filling task result: {result2['status']} in {result2['execution_time']:.2f}s")
        
        # 3. Navigation task (navigation context)
        logger.info("\n--- Demonstrating Navigation Task ---")
        navigation_task = {
            "task_id": "nav_001",
            "type": "navigation",
            "url": "https://social-media.com",
            "navigation_steps": [
                {"action": "click", "selector": ".login-button"},
                {"action": "fill", "selector": "#username", "value": "testuser"},
                {"action": "fill", "selector": "#password", "value": "password123"},
                {"action": "click", "selector": ".submit-button"},
                {"action": "wait", "selector": ".dashboard"}
            ],
            "priority": "speed",
            "complexity": "low",
            "time_limit": 30
        }
        
        result3 = await integration.execute_webiq_task(navigation_task)
        logger.info(f"Navigation task result: {result3['status']} in {result3['execution_time']:.2f}s")
        
        # Wait a bit for adaptations to process
        await asyncio.sleep(2)
        
        # Show system metrics and recommendations
        logger.info("\n--- System Metrics and Health ---")
        metrics = await integration.get_adaptation_metrics()
        logger.info(f"Total adaptations applied: {metrics['metrics']['total_adaptations_applied']}")
        logger.info(f"System health score: {metrics['health']['health_score']:.2f}")
        
        recommendations = await integration.get_system_recommendations()
        if recommendations["recommendations"]:
            logger.info("\n--- System Recommendations ---")
            for rec in recommendations["recommendations"]:
                logger.info(f"- {rec['type'].upper()}: {rec['message']}")
        
        logger.info("\n=== Demonstration Complete ===")
        
    finally:
        # Cleanup
        await integration.shutdown()

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_adaptation_system())