# Optional agents import (requires langgraph)
try:
    from .agents import WebIQAgent
    _AGENTS_AVAILABLE = True
except ImportError as e:
    # Create mock class for missing dependencies
    class WebIQAgent:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"WebIQAgent requires langgraph: {e}")
    
    _AGENTS_AVAILABLE = False
# Optional automation service import (may require langgraph)
try:
    from .automation_service import AutomationService
    _AUTOMATION_SERVICE_AVAILABLE = True
except ImportError as e:
    # Create mock class for missing dependencies
    class AutomationService:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"AutomationService requires additional dependencies: {e}")
    
    _AUTOMATION_SERVICE_AVAILABLE = False
from .recording_session import GoalAwareRecordingSession as RecordingSession
from .exceptions import WebIQError, AutomationError, ConfigurationError
from .cost_optimizer import CostOptimizer
# Optional enhanced agents import (requires langgraph)
try:
    from .enhanced_agents import (
        EnhancedWebIQAgent,
        PerformanceAgent,
        MetricType,
        PerformanceMetric,
        OptimizationSuggestion
    )
    _ENHANCED_AGENTS_AVAILABLE = True
except ImportError as e:
    # Create mock classes for missing dependencies
    class EnhancedWebIQAgent:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Enhanced agents require langgraph: {e}")
    
    class PerformanceAgent:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Performance agent requires langgraph: {e}")
    
    class MetricType:
        pass
    
    class PerformanceMetric:
        pass
    
    class OptimizationSuggestion:
        pass
    
    _ENHANCED_AGENTS_AVAILABLE = False
# Optional enhanced services import (may require langgraph)
try:
    from .enhanced_automation_service import EnhancedAutomationService
    from .enhanced_recording_session import EnhancedRecordingSession
    _ENHANCED_SERVICES_AVAILABLE = True
except ImportError as e:
    # Create mock classes for missing dependencies
    class EnhancedAutomationService:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Enhanced automation service requires additional dependencies: {e}")
    
    class EnhancedRecordingSession:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Enhanced recording session requires additional dependencies: {e}")
    
    _ENHANCED_SERVICES_AVAILABLE = False
from .self_healing import SelfHealingAgent
from .learning_system import LearningSystem, LearningConfig, LearningMetrics
from .pattern_recognition_engine import PatternRecognitionEngine
from .adaptive_strategy_engine import AdaptiveStrategyEngine, AdaptiveStrategy
from .predictive_analytics_engine import PredictiveAnalyticsEngine
from .knowledge_base_manager import (
    KnowledgeBaseManager,
    KnowledgeBase,
    AutomationPattern,
    SiteKnowledge,
    GoalTemplate
)

from .advanced_goal_processor import (
    AdvancedGoalProcessor,
    GoalComplexity,
    GoalType,
    GoalContext,
    SubGoal,
    GoalDecomposition
)

from .requirement_extractor import RequirementExtractor

from .contextual_strategy_selector import (
    ContextualStrategySelector,
    StrategyTemplate,
    StrategyType,
    ExecutionMode,
    ContextFactors,
    StrategyEvaluation
)

__all__ = [
    'WebIQAgent',
    'AutomationService', 
    'RecordingSession',
    'WebIQError',
    'AutomationError', 
    'ConfigurationError',
    'CostOptimizer',
    'EnhancedWebIQAgent',
    'PerformanceAgent',
    'MetricType',
    'PerformanceMetric',
    'OptimizationSuggestion',
    'EnhancedAutomationService',
    'EnhancedRecordingSession',
    'SelfHealingAgent',
    'LearningSystem',
    'LearningConfig',
    'LearningMetrics',
    'PatternRecognitionEngine',
    'AdaptiveStrategyEngine',
    'AdaptiveStrategy',
    'PredictiveAnalyticsEngine',
    'KnowledgeBaseManager',
    'KnowledgeBase',
    'AutomationPattern',
    'SiteKnowledge',
    'GoalTemplate',
    'AdvancedGoalProcessor',
    'GoalComplexity',
    'GoalType',
    'GoalContext',
    'SubGoal',
    'GoalDecomposition',
    'RequirementExtractor',
    'ContextualStrategySelector',
    'StrategyTemplate',
    'StrategyType',
    'ExecutionMode',
    'ContextFactors',
    'StrategyEvaluation'
]
