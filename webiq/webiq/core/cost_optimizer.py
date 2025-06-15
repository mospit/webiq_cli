"""Cost optimization module for WebIQ automation."""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CostMetric:
    """Represents a cost metric for automation operations."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    category: str = "general"


@dataclass
class OptimizationRecommendation:
    """Represents a cost optimization recommendation."""
    title: str
    description: str
    potential_savings: float
    implementation_effort: str  # "low", "medium", "high"
    priority: str  # "low", "medium", "high", "critical"
    category: str


class CostOptimizer:
    """Handles cost optimization for WebIQ automation operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics: List[CostMetric] = []
        self.recommendations: List[OptimizationRecommendation] = []
        
    def track_cost(self, name: str, value: float, unit: str = "USD", category: str = "general") -> None:
        """Track a cost metric."""
        metric = CostMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            category=category
        )
        self.metrics.append(metric)
        self.logger.debug(f"Tracked cost metric: {name} = {value} {unit}")
        
    def get_total_cost(self, category: Optional[str] = None) -> float:
        """Get total cost for a specific category or all categories."""
        if category:
            return sum(m.value for m in self.metrics if m.category == category)
        return sum(m.value for m in self.metrics)
        
    def get_cost_breakdown(self) -> Dict[str, float]:
        """Get cost breakdown by category."""
        breakdown = {}
        for metric in self.metrics:
            if metric.category not in breakdown:
                breakdown[metric.category] = 0
            breakdown[metric.category] += metric.value
        return breakdown
        
    def analyze_costs(self) -> List[OptimizationRecommendation]:
        """Analyze costs and generate optimization recommendations."""
        recommendations = []
        
        # Analyze high-cost categories
        breakdown = self.get_cost_breakdown()
        total_cost = sum(breakdown.values())
        
        if total_cost > 0:
            for category, cost in breakdown.items():
                percentage = (cost / total_cost) * 100
                if percentage > 30:  # High-cost category
                    recommendations.append(OptimizationRecommendation(
                        title=f"Optimize {category} operations",
                        description=f"Category '{category}' accounts for {percentage:.1f}% of total costs",
                        potential_savings=cost * 0.2,  # Assume 20% potential savings
                        implementation_effort="medium",
                        priority="high" if percentage > 50 else "medium",
                        category=category
                    ))
        
        # Check for frequent operations
        operation_counts = {}
        for metric in self.metrics:
            if metric.name not in operation_counts:
                operation_counts[metric.name] = 0
            operation_counts[metric.name] += 1
            
        for operation, count in operation_counts.items():
            if count > 10:  # Frequent operation
                avg_cost = sum(m.value for m in self.metrics if m.name == operation) / count
                if avg_cost > 1.0:  # High average cost
                    recommendations.append(OptimizationRecommendation(
                        title=f"Optimize frequent operation: {operation}",
                        description=f"Operation '{operation}' runs {count} times with avg cost ${avg_cost:.2f}",
                        potential_savings=avg_cost * count * 0.15,  # 15% savings
                        implementation_effort="low",
                        priority="medium",
                        category="optimization"
                    ))
        
        self.recommendations.extend(recommendations)
        return recommendations
        
    def get_recommendations(self, priority: Optional[str] = None) -> List[OptimizationRecommendation]:
        """Get optimization recommendations, optionally filtered by priority."""
        if priority:
            return [r for r in self.recommendations if r.priority == priority]
        return self.recommendations.copy()
        
    def clear_metrics(self) -> None:
        """Clear all tracked metrics."""
        self.metrics.clear()
        self.logger.info("Cleared all cost metrics")
        
    def export_metrics(self) -> Dict[str, Any]:
        """Export metrics for external analysis."""
        return {
            "total_cost": self.get_total_cost(),
            "breakdown": self.get_cost_breakdown(),
            "metrics_count": len(self.metrics),
            "recommendations_count": len(self.recommendations),
            "last_updated": datetime.now().isoformat()
        }