"""Self-healing automation module for WebIQ."""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class FailureType(Enum):
    """Types of failures that can be detected and healed."""
    ELEMENT_NOT_FOUND = "element_not_found"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    AUTHENTICATION_ERROR = "authentication_error"
    VALIDATION_ERROR = "validation_error"
    UNEXPECTED_STATE = "unexpected_state"
    RESOURCE_UNAVAILABLE = "resource_unavailable"


@dataclass
class FailureEvent:
    """Represents a failure event in the automation system."""
    failure_type: FailureType
    description: str
    timestamp: datetime
    context: Dict[str, Any]
    severity: str = "medium"  # "low", "medium", "high", "critical"
    resolved: bool = False
    resolution_attempts: int = 0


@dataclass
class HealingStrategy:
    """Represents a healing strategy for a specific failure type."""
    name: str
    failure_types: List[FailureType]
    handler: Callable
    max_attempts: int = 3
    cooldown_seconds: int = 5
    priority: int = 1  # Higher number = higher priority


class SelfHealingAgent:
    """Handles self-healing capabilities for WebIQ automation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.failure_history: List[FailureEvent] = []
        self.healing_strategies: List[HealingStrategy] = []
        self.is_enabled = True
        self.max_history_size = 1000
        self._setup_default_strategies()
        
    def _setup_default_strategies(self):
        """Setup default healing strategies."""
        # Element not found strategy
        self.register_strategy(HealingStrategy(
            name="wait_and_retry",
            failure_types=[FailureType.ELEMENT_NOT_FOUND],
            handler=self._wait_and_retry_handler,
            max_attempts=3,
            cooldown_seconds=2,
            priority=1
        ))
        
        # Timeout strategy
        self.register_strategy(HealingStrategy(
            name="extend_timeout",
            failure_types=[FailureType.TIMEOUT],
            handler=self._extend_timeout_handler,
            max_attempts=2,
            cooldown_seconds=5,
            priority=2
        ))
        
        # Network error strategy
        self.register_strategy(HealingStrategy(
            name="retry_with_backoff",
            failure_types=[FailureType.NETWORK_ERROR],
            handler=self._retry_with_backoff_handler,
            max_attempts=5,
            cooldown_seconds=10,
            priority=3
        ))
        
        # Authentication error strategy
        self.register_strategy(HealingStrategy(
            name="refresh_auth",
            failure_types=[FailureType.AUTHENTICATION_ERROR],
            handler=self._refresh_auth_handler,
            max_attempts=2,
            cooldown_seconds=1,
            priority=4
        ))
        
    def register_strategy(self, strategy: HealingStrategy) -> None:
        """Register a new healing strategy."""
        self.healing_strategies.append(strategy)
        self.healing_strategies.sort(key=lambda s: s.priority, reverse=True)
        self.logger.info(f"Registered healing strategy: {strategy.name}")
        
    def record_failure(self, failure_type: FailureType, description: str, 
                      context: Optional[Dict[str, Any]] = None, 
                      severity: str = "medium") -> FailureEvent:
        """Record a failure event."""
        event = FailureEvent(
            failure_type=failure_type,
            description=description,
            timestamp=datetime.now(),
            context=context or {},
            severity=severity
        )
        
        self.failure_history.append(event)
        
        # Maintain history size limit
        if len(self.failure_history) > self.max_history_size:
            self.failure_history = self.failure_history[-self.max_history_size:]
            
        self.logger.warning(f"Recorded failure: {failure_type.value} - {description}")
        return event
        
    async def attempt_healing(self, failure_event: FailureEvent) -> bool:
        """Attempt to heal a failure using registered strategies."""
        if not self.is_enabled:
            self.logger.info("Self-healing is disabled")
            return False
            
        # Find applicable strategies
        applicable_strategies = [
            strategy for strategy in self.healing_strategies
            if failure_event.failure_type in strategy.failure_types
        ]
        
        if not applicable_strategies:
            self.logger.warning(f"No healing strategy found for {failure_event.failure_type.value}")
            return False
            
        # Try strategies in priority order
        for strategy in applicable_strategies:
            if failure_event.resolution_attempts >= strategy.max_attempts:
                continue
                
            try:
                self.logger.info(f"Attempting healing with strategy: {strategy.name}")
                failure_event.resolution_attempts += 1
                
                # Apply cooldown
                if strategy.cooldown_seconds > 0:
                    await asyncio.sleep(strategy.cooldown_seconds)
                    
                # Execute healing strategy
                success = await strategy.handler(failure_event)
                
                if success:
                    failure_event.resolved = True
                    self.logger.info(f"Successfully healed failure using {strategy.name}")
                    return True
                    
            except Exception as e:
                self.logger.error(f"Healing strategy {strategy.name} failed: {str(e)}")
                continue
                
        self.logger.warning(f"All healing attempts failed for {failure_event.failure_type.value}")
        return False
        
    async def _wait_and_retry_handler(self, failure_event: FailureEvent) -> bool:
        """Handler for wait and retry strategy."""
        wait_time = min(2 ** failure_event.resolution_attempts, 10)  # Exponential backoff, max 10s
        self.logger.debug(f"Waiting {wait_time}s before retry")
        await asyncio.sleep(wait_time)
        return True  # Indicate strategy was applied (actual retry happens in calling code)
        
    async def _extend_timeout_handler(self, failure_event: FailureEvent) -> bool:
        """Handler for extending timeout strategy."""
        # This would typically modify timeout settings in the context
        context = failure_event.context
        current_timeout = context.get('timeout', 30)
        new_timeout = min(current_timeout * 1.5, 120)  # Increase by 50%, max 2 minutes
        context['timeout'] = new_timeout
        self.logger.debug(f"Extended timeout from {current_timeout}s to {new_timeout}s")
        return True
        
    async def _retry_with_backoff_handler(self, failure_event: FailureEvent) -> bool:
        """Handler for retry with exponential backoff strategy."""
        backoff_time = min(2 ** failure_event.resolution_attempts, 60)  # Max 1 minute
        self.logger.debug(f"Applying backoff: {backoff_time}s")
        await asyncio.sleep(backoff_time)
        return True
        
    async def _refresh_auth_handler(self, failure_event: FailureEvent) -> bool:
        """Handler for refreshing authentication strategy."""
        # This would typically trigger auth refresh in the context
        self.logger.debug("Triggering authentication refresh")
        # Mark in context that auth should be refreshed
        failure_event.context['refresh_auth'] = True
        return True
        
    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get statistics about failures and healing attempts."""
        if not self.failure_history:
            return {"total_failures": 0, "resolved_failures": 0, "resolution_rate": 0.0}
            
        total_failures = len(self.failure_history)
        resolved_failures = sum(1 for f in self.failure_history if f.resolved)
        resolution_rate = (resolved_failures / total_failures) * 100
        
        # Failure type breakdown
        failure_types = {}
        for failure in self.failure_history:
            failure_type = failure.failure_type.value
            if failure_type not in failure_types:
                failure_types[failure_type] = {"total": 0, "resolved": 0}
            failure_types[failure_type]["total"] += 1
            if failure.resolved:
                failure_types[failure_type]["resolved"] += 1
                
        return {
            "total_failures": total_failures,
            "resolved_failures": resolved_failures,
            "resolution_rate": round(resolution_rate, 2),
            "failure_types": failure_types,
            "strategies_count": len(self.healing_strategies)
        }
        
    def clear_history(self) -> None:
        """Clear failure history."""
        self.failure_history.clear()
        self.logger.info("Cleared failure history")
        
    def enable(self) -> None:
        """Enable self-healing."""
        self.is_enabled = True
        self.logger.info("Self-healing enabled")
        
    def disable(self) -> None:
        """Disable self-healing."""
        self.is_enabled = False
        self.logger.info("Self-healing disabled")
        
    def get_recent_failures(self, hours: int = 24) -> List[FailureEvent]:
        """Get failures from the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [f for f in self.failure_history if f.timestamp >= cutoff_time]