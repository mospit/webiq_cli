import asyncio
import json
import sqlite3
import pickle
import hashlib
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging
from pathlib import Path
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class AutomationPattern:
    """Represents a learned automation pattern"""
    pattern_id: str
    pattern_type: str  # 'success', 'failure', 'optimization'
    goal_type: str
    site_domain: str
    action_sequence: List[Dict[str, Any]]
    success_rate: float
    avg_duration: float
    avg_cost: float
    confidence_score: float
    usage_count: int = 0
    last_used: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        if self.last_used:
            data['last_used'] = self.last_used.isoformat()
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AutomationPattern':
        """Create from dictionary"""
        # Convert ISO strings back to datetime objects
        if data.get('last_used'):
            data['last_used'] = datetime.fromisoformat(data['last_used'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)

@dataclass
class SiteKnowledge:
    """Knowledge about a specific site"""
    domain: str
    common_elements: Dict[str, List[str]]  # element_type -> [selectors]
    timing_patterns: Dict[str, float]  # action_type -> avg_wait_time
    success_indicators: List[str]  # selectors for success detection
    failure_indicators: List[str]  # selectors for failure detection
    optimization_rules: List[Dict[str, Any]]  # site-specific optimizations
    complexity_score: float
    reliability_score: float
    last_updated: datetime = field(default_factory=datetime.now)
    session_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['last_updated'] = self.last_updated.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SiteKnowledge':
        """Create from dictionary"""
        data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)

@dataclass
class GoalTemplate:
    """Template for a specific goal type"""
    goal_type: str
    canonical_steps: List[Dict[str, Any]]  # standardized action sequence
    success_rate: float
    avg_duration: float
    complexity_score: float
    required_elements: List[str]  # element types needed for this goal
    common_variations: List[List[Dict[str, Any]]]  # alternative sequences
    optimization_hints: List[str]
    last_updated: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['last_updated'] = self.last_updated.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GoalTemplate':
        """Create from dictionary"""
        data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)

class KnowledgeBase:
    """Core knowledge storage and retrieval system"""
    
    def __init__(self, db_path: str = "webiq_knowledge.db"):
        self.db_path = db_path
        self.db_lock = threading.RLock()
        
        # In-memory caches for performance
        self.patterns_cache: Dict[str, AutomationPattern] = {}
        self.site_knowledge_cache: Dict[str, SiteKnowledge] = {}
        self.goal_templates_cache: Dict[str, GoalTemplate] = {}
        
        # Cache management
        self.cache_max_size = 1000
        self.cache_ttl = timedelta(hours=1)
        self.last_cache_cleanup = datetime.now()
        
        # Initialize database
        self._init_database()
        
        # Load initial cache
        asyncio.create_task(self._load_initial_cache())
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Automation patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS automation_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    goal_type TEXT NOT NULL,
                    site_domain TEXT NOT NULL,
                    action_sequence TEXT NOT NULL,
                    success_rate REAL NOT NULL,
                    avg_duration REAL NOT NULL,
                    avg_cost REAL NOT NULL,
                    confidence_score REAL NOT NULL,
                    usage_count INTEGER DEFAULT 0,
                    last_used TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Site knowledge table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS site_knowledge (
                    domain TEXT PRIMARY KEY,
                    common_elements TEXT NOT NULL,
                    timing_patterns TEXT NOT NULL,
                    success_indicators TEXT NOT NULL,
                    failure_indicators TEXT NOT NULL,
                    optimization_rules TEXT NOT NULL,
                    complexity_score REAL NOT NULL,
                    reliability_score REAL NOT NULL,
                    last_updated TEXT NOT NULL,
                    session_count INTEGER DEFAULT 0
                )
            """)
            
            # Goal templates table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS goal_templates (
                    goal_type TEXT PRIMARY KEY,
                    canonical_steps TEXT NOT NULL,
                    success_rate REAL NOT NULL,
                    avg_duration REAL NOT NULL,
                    complexity_score REAL NOT NULL,
                    required_elements TEXT NOT NULL,
                    common_variations TEXT NOT NULL,
                    optimization_hints TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    usage_count INTEGER DEFAULT 0
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_goal_type ON automation_patterns(goal_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_site_domain ON automation_patterns(site_domain)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_type ON automation_patterns(pattern_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_success_rate ON automation_patterns(success_rate)")
            
            conn.commit()
    
    @contextmanager
    def _get_db_connection(self):
        """Get database connection with proper locking"""
        with self.db_lock:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
    
    async def _load_initial_cache(self):
        """Load frequently used data into cache"""
        try:
            # Load recent patterns
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Load most recent and frequently used patterns
                cursor.execute("""
                    SELECT * FROM automation_patterns 
                    ORDER BY usage_count DESC, updated_at DESC 
                    LIMIT ?
                """, (self.cache_max_size // 2,))
                
                for row in cursor.fetchall():
                    pattern_data = dict(row)
                    pattern_data['action_sequence'] = json.loads(pattern_data['action_sequence'])
                    pattern_data['metadata'] = json.loads(pattern_data['metadata'] or '{}')
                    pattern = AutomationPattern.from_dict(pattern_data)
                    self.patterns_cache[pattern.pattern_id] = pattern
                
                # Load all site knowledge (typically not too many sites)
                cursor.execute("SELECT * FROM site_knowledge")
                for row in cursor.fetchall():
                    site_data = dict(row)
                    for field in ['common_elements', 'timing_patterns', 'success_indicators', 
                                'failure_indicators', 'optimization_rules']:
                        site_data[field] = json.loads(site_data[field])
                    site_knowledge = SiteKnowledge.from_dict(site_data)
                    self.site_knowledge_cache[site_knowledge.domain] = site_knowledge
                
                # Load all goal templates
                cursor.execute("SELECT * FROM goal_templates")
                for row in cursor.fetchall():
                    template_data = dict(row)
                    for field in ['canonical_steps', 'required_elements', 'common_variations', 'optimization_hints']:
                        template_data[field] = json.loads(template_data[field])
                    template = GoalTemplate.from_dict(template_data)
                    self.goal_templates_cache[template.goal_type] = template
            
            logger.info(f"Loaded {len(self.patterns_cache)} patterns, {len(self.site_knowledge_cache)} sites, {len(self.goal_templates_cache)} templates into cache")
            
        except Exception as e:
            logger.error(f"Failed to load initial cache: {e}")
    
    async def store_pattern(self, pattern: AutomationPattern) -> bool:
        """Store automation pattern"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Check if pattern exists
                cursor.execute("SELECT pattern_id FROM automation_patterns WHERE pattern_id = ?", (pattern.pattern_id,))
                exists = cursor.fetchone() is not None
                
                if exists:
                    # Update existing pattern
                    cursor.execute("""
                        UPDATE automation_patterns SET
                            pattern_type = ?, goal_type = ?, site_domain = ?,
                            action_sequence = ?, success_rate = ?, avg_duration = ?,
                            avg_cost = ?, confidence_score = ?, usage_count = ?,
                            last_used = ?, updated_at = ?, metadata = ?
                        WHERE pattern_id = ?
                    """, (
                        pattern.pattern_type, pattern.goal_type, pattern.site_domain,
                        json.dumps(pattern.action_sequence), pattern.success_rate,
                        pattern.avg_duration, pattern.avg_cost, pattern.confidence_score,
                        pattern.usage_count, pattern.last_used.isoformat() if pattern.last_used else None,
                        pattern.updated_at.isoformat(), json.dumps(pattern.metadata),
                        pattern.pattern_id
                    ))
                else:
                    # Insert new pattern
                    cursor.execute("""
                        INSERT INTO automation_patterns (
                            pattern_id, pattern_type, goal_type, site_domain,
                            action_sequence, success_rate, avg_duration, avg_cost,
                            confidence_score, usage_count, last_used, created_at,
                            updated_at, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        pattern.pattern_id, pattern.pattern_type, pattern.goal_type,
                        pattern.site_domain, json.dumps(pattern.action_sequence),
                        pattern.success_rate, pattern.avg_duration, pattern.avg_cost,
                        pattern.confidence_score, pattern.usage_count,
                        pattern.last_used.isoformat() if pattern.last_used else None,
                        pattern.created_at.isoformat(), pattern.updated_at.isoformat(),
                        json.dumps(pattern.metadata)
                    ))
                
                conn.commit()
                
                # Update cache
                self.patterns_cache[pattern.pattern_id] = pattern
                await self._cleanup_cache_if_needed()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to store pattern {pattern.pattern_id}: {e}")
            return False
    
    async def get_pattern(self, pattern_id: str) -> Optional[AutomationPattern]:
        """Retrieve automation pattern by ID"""
        
        # Check cache first
        if pattern_id in self.patterns_cache:
            return self.patterns_cache[pattern_id]
        
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM automation_patterns WHERE pattern_id = ?", (pattern_id,))
                row = cursor.fetchone()
                
                if row:
                    pattern_data = dict(row)
                    pattern_data['action_sequence'] = json.loads(pattern_data['action_sequence'])
                    pattern_data['metadata'] = json.loads(pattern_data['metadata'] or '{}')
                    pattern = AutomationPattern.from_dict(pattern_data)
                    
                    # Add to cache
                    self.patterns_cache[pattern_id] = pattern
                    return pattern
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get pattern {pattern_id}: {e}")
            return None
    
    async def find_patterns(self, 
                          goal_type: Optional[str] = None,
                          site_domain: Optional[str] = None,
                          pattern_type: Optional[str] = None,
                          min_success_rate: float = 0.0,
                          limit: int = 100) -> List[AutomationPattern]:
        """Find patterns matching criteria"""
        
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Build query
                conditions = ["success_rate >= ?"]
                params = [min_success_rate]
                
                if goal_type:
                    conditions.append("goal_type = ?")
                    params.append(goal_type)
                
                if site_domain:
                    conditions.append("site_domain = ?")
                    params.append(site_domain)
                
                if pattern_type:
                    conditions.append("pattern_type = ?")
                    params.append(pattern_type)
                
                query = f"""
                    SELECT * FROM automation_patterns 
                    WHERE {' AND '.join(conditions)}
                    ORDER BY success_rate DESC, usage_count DESC
                    LIMIT ?
                """
                params.append(limit)
                
                cursor.execute(query, params)
                patterns = []
                
                for row in cursor.fetchall():
                    pattern_data = dict(row)
                    pattern_data['action_sequence'] = json.loads(pattern_data['action_sequence'])
                    pattern_data['metadata'] = json.loads(pattern_data['metadata'] or '{}')
                    pattern = AutomationPattern.from_dict(pattern_data)
                    patterns.append(pattern)
                    
                    # Add to cache if not already there
                    if pattern.pattern_id not in self.patterns_cache:
                        self.patterns_cache[pattern.pattern_id] = pattern
                
                return patterns
                
        except Exception as e:
            logger.error(f"Failed to find patterns: {e}")
            return []
    
    async def store_site_knowledge(self, site_knowledge: SiteKnowledge) -> bool:
        """Store site-specific knowledge"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Check if exists
                cursor.execute("SELECT domain FROM site_knowledge WHERE domain = ?", (site_knowledge.domain,))
                exists = cursor.fetchone() is not None
                
                if exists:
                    # Update existing
                    cursor.execute("""
                        UPDATE site_knowledge SET
                            common_elements = ?, timing_patterns = ?, success_indicators = ?,
                            failure_indicators = ?, optimization_rules = ?, complexity_score = ?,
                            reliability_score = ?, last_updated = ?, session_count = ?
                        WHERE domain = ?
                    """, (
                        json.dumps(site_knowledge.common_elements),
                        json.dumps(site_knowledge.timing_patterns),
                        json.dumps(site_knowledge.success_indicators),
                        json.dumps(site_knowledge.failure_indicators),
                        json.dumps(site_knowledge.optimization_rules),
                        site_knowledge.complexity_score,
                        site_knowledge.reliability_score,
                        site_knowledge.last_updated.isoformat(),
                        site_knowledge.session_count,
                        site_knowledge.domain
                    ))
                else:
                    # Insert new
                    cursor.execute("""
                        INSERT INTO site_knowledge (
                            domain, common_elements, timing_patterns, success_indicators,
                            failure_indicators, optimization_rules, complexity_score,
                            reliability_score, last_updated, session_count
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        site_knowledge.domain,
                        json.dumps(site_knowledge.common_elements),
                        json.dumps(site_knowledge.timing_patterns),
                        json.dumps(site_knowledge.success_indicators),
                        json.dumps(site_knowledge.failure_indicators),
                        json.dumps(site_knowledge.optimization_rules),
                        site_knowledge.complexity_score,
                        site_knowledge.reliability_score,
                        site_knowledge.last_updated.isoformat(),
                        site_knowledge.session_count
                    ))
                
                conn.commit()
                
                # Update cache
                self.site_knowledge_cache[site_knowledge.domain] = site_knowledge
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to store site knowledge for {site_knowledge.domain}: {e}")
            return False
    
    async def get_site_knowledge(self, domain: str) -> Optional[SiteKnowledge]:
        """Get site-specific knowledge"""
        
        # Check cache first
        if domain in self.site_knowledge_cache:
            return self.site_knowledge_cache[domain]
        
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM site_knowledge WHERE domain = ?", (domain,))
                row = cursor.fetchone()
                
                if row:
                    site_data = dict(row)
                    for field in ['common_elements', 'timing_patterns', 'success_indicators', 
                                'failure_indicators', 'optimization_rules']:
                        site_data[field] = json.loads(site_data[field])
                    site_knowledge = SiteKnowledge.from_dict(site_data)
                    
                    # Add to cache
                    self.site_knowledge_cache[domain] = site_knowledge
                    return site_knowledge
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get site knowledge for {domain}: {e}")
            return None
    
    async def store_goal_template(self, template: GoalTemplate) -> bool:
        """Store goal template"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Check if exists
                cursor.execute("SELECT goal_type FROM goal_templates WHERE goal_type = ?", (template.goal_type,))
                exists = cursor.fetchone() is not None
                
                if exists:
                    # Update existing
                    cursor.execute("""
                        UPDATE goal_templates SET
                            canonical_steps = ?, success_rate = ?, avg_duration = ?,
                            complexity_score = ?, required_elements = ?, common_variations = ?,
                            optimization_hints = ?, last_updated = ?, usage_count = ?
                        WHERE goal_type = ?
                    """, (
                        json.dumps(template.canonical_steps),
                        template.success_rate,
                        template.avg_duration,
                        template.complexity_score,
                        json.dumps(template.required_elements),
                        json.dumps(template.common_variations),
                        json.dumps(template.optimization_hints),
                        template.last_updated.isoformat(),
                        template.usage_count,
                        template.goal_type
                    ))
                else:
                    # Insert new
                    cursor.execute("""
                        INSERT INTO goal_templates (
                            goal_type, canonical_steps, success_rate, avg_duration,
                            complexity_score, required_elements, common_variations,
                            optimization_hints, last_updated, usage_count
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        template.goal_type,
                        json.dumps(template.canonical_steps),
                        template.success_rate,
                        template.avg_duration,
                        template.complexity_score,
                        json.dumps(template.required_elements),
                        json.dumps(template.common_variations),
                        json.dumps(template.optimization_hints),
                        template.last_updated.isoformat(),
                        template.usage_count
                    ))
                
                conn.commit()
                
                # Update cache
                self.goal_templates_cache[template.goal_type] = template
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to store goal template {template.goal_type}: {e}")
            return False
    
    async def get_goal_template(self, goal_type: str) -> Optional[GoalTemplate]:
        """Get goal template"""
        
        # Check cache first
        if goal_type in self.goal_templates_cache:
            return self.goal_templates_cache[goal_type]
        
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM goal_templates WHERE goal_type = ?", (goal_type,))
                row = cursor.fetchone()
                
                if row:
                    template_data = dict(row)
                    for field in ['canonical_steps', 'required_elements', 'common_variations', 'optimization_hints']:
                        template_data[field] = json.loads(template_data[field])
                    template = GoalTemplate.from_dict(template_data)
                    
                    # Add to cache
                    self.goal_templates_cache[goal_type] = template
                    return template
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get goal template {goal_type}: {e}")
            return None
    
    async def update_pattern_usage(self, pattern_id: str) -> bool:
        """Update pattern usage statistics"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE automation_patterns 
                    SET usage_count = usage_count + 1, last_used = ?
                    WHERE pattern_id = ?
                """, (datetime.now().isoformat(), pattern_id))
                conn.commit()
                
                # Update cache if present
                if pattern_id in self.patterns_cache:
                    pattern = self.patterns_cache[pattern_id]
                    pattern.usage_count += 1
                    pattern.last_used = datetime.now()
                
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Failed to update pattern usage {pattern_id}: {e}")
            return False
    
    async def cleanup_old_patterns(self, days_old: int = 90, min_usage: int = 1) -> int:
        """Clean up old, unused patterns"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Find patterns to delete
                cursor.execute("""
                    SELECT pattern_id FROM automation_patterns 
                    WHERE (last_used IS NULL OR last_used < ?) 
                    AND usage_count < ?
                """, (cutoff_date.isoformat(), min_usage))
                
                pattern_ids = [row[0] for row in cursor.fetchall()]
                
                if pattern_ids:
                    # Delete patterns
                    placeholders = ','.join(['?'] * len(pattern_ids))
                    cursor.execute(f"DELETE FROM automation_patterns WHERE pattern_id IN ({placeholders})", pattern_ids)
                    conn.commit()
                    
                    # Remove from cache
                    for pattern_id in pattern_ids:
                        self.patterns_cache.pop(pattern_id, None)
                    
                    logger.info(f"Cleaned up {len(pattern_ids)} old patterns")
                    return len(pattern_ids)
                
                return 0
                
        except Exception as e:
            logger.error(f"Failed to cleanup old patterns: {e}")
            return 0
    
    async def _cleanup_cache_if_needed(self):
        """Clean up cache if it's getting too large or old"""
        now = datetime.now()
        
        # Check if cleanup is needed
        if (len(self.patterns_cache) > self.cache_max_size or 
            now - self.last_cache_cleanup > self.cache_ttl):
            
            # Remove least recently used patterns
            if len(self.patterns_cache) > self.cache_max_size:
                # Sort by last used time (None values go first)
                sorted_patterns = sorted(
                    self.patterns_cache.items(),
                    key=lambda x: x[1].last_used or datetime.min
                )
                
                # Remove oldest patterns
                patterns_to_remove = len(self.patterns_cache) - self.cache_max_size + 100  # Remove extra for buffer
                for pattern_id, _ in sorted_patterns[:patterns_to_remove]:
                    del self.patterns_cache[pattern_id]
            
            self.last_cache_cleanup = now
    
    async def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Pattern statistics
                cursor.execute("SELECT COUNT(*) FROM automation_patterns")
                total_patterns = cursor.fetchone()[0]
                
                cursor.execute("SELECT pattern_type, COUNT(*) FROM automation_patterns GROUP BY pattern_type")
                patterns_by_type = dict(cursor.fetchall())
                
                cursor.execute("SELECT AVG(success_rate) FROM automation_patterns WHERE pattern_type = 'success'")
                avg_success_rate = cursor.fetchone()[0] or 0.0
                
                # Site statistics
                cursor.execute("SELECT COUNT(*) FROM site_knowledge")
                total_sites = cursor.fetchone()[0]
                
                cursor.execute("SELECT AVG(reliability_score) FROM site_knowledge")
                avg_site_reliability = cursor.fetchone()[0] or 0.0
                
                # Goal template statistics
                cursor.execute("SELECT COUNT(*) FROM goal_templates")
                total_templates = cursor.fetchone()[0]
                
                cursor.execute("SELECT AVG(success_rate) FROM goal_templates")
                avg_template_success = cursor.fetchone()[0] or 0.0
                
                return {
                    "patterns": {
                        "total": total_patterns,
                        "by_type": patterns_by_type,
                        "avg_success_rate": avg_success_rate,
                        "cached": len(self.patterns_cache)
                    },
                    "sites": {
                        "total": total_sites,
                        "avg_reliability": avg_site_reliability,
                        "cached": len(self.site_knowledge_cache)
                    },
                    "templates": {
                        "total": total_templates,
                        "avg_success_rate": avg_template_success,
                        "cached": len(self.goal_templates_cache)
                    },
                    "cache_stats": {
                        "last_cleanup": self.last_cache_cleanup.isoformat(),
                        "max_size": self.cache_max_size
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get knowledge stats: {e}")
            return {}

class KnowledgeBaseManager:
    """High-level manager for knowledge base operations"""
    
    def __init__(self, db_path: str = "webiq_knowledge.db"):
        self.knowledge_base = KnowledgeBase(db_path)
        
        # Pattern generation
        self.pattern_id_counter = 0
        
        # Learning configuration
        self.min_pattern_confidence = 0.6
        self.min_pattern_usage = 2
        self.pattern_similarity_threshold = 0.8
    
    async def learn_from_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn patterns from a completed automation session"""
        
        learning_results = {
            "patterns_created": 0,
            "patterns_updated": 0,
            "site_knowledge_updated": False,
            "goal_template_updated": False,
            "insights": []
        }
        
        try:
            goal = session_data.get("goal", "")
            url = session_data.get("url", "")
            success = session_data.get("success", False)
            action_history = session_data.get("action_history", [])
            duration = session_data.get("duration", 0.0)
            cost = session_data.get("cost", 0.0)
            errors = session_data.get("errors", [])
            
            domain = self._extract_domain(url)
            goal_type = self._classify_goal_type(goal)
            
            # Learn action patterns
            if action_history:
                patterns_learned = await self._learn_action_patterns(
                    goal_type, domain, action_history, success, duration, cost
                )
                learning_results["patterns_created"] += patterns_learned["created"]
                learning_results["patterns_updated"] += patterns_learned["updated"]
                learning_results["insights"].extend(patterns_learned["insights"])
            
            # Update site knowledge
            site_updated = await self._update_site_knowledge(
                domain, action_history, success, duration, errors
            )
            learning_results["site_knowledge_updated"] = site_updated
            
            # Update goal template
            template_updated = await self._update_goal_template(
                goal_type, action_history, success, duration
            )
            learning_results["goal_template_updated"] = template_updated
            
            # Learn from failures
            if not success and errors:
                failure_insights = await self._learn_from_failures(
                    goal_type, domain, action_history, errors
                )
                learning_results["insights"].extend(failure_insights)
            
            logger.info(f"Learning completed: {learning_results}")
            
        except Exception as e:
            logger.error(f"Learning from session failed: {e}")
            learning_results["error"] = str(e)
        
        return learning_results
    
    async def _learn_action_patterns(self, goal_type: str, domain: str, 
                                   action_history: List[Dict[str, Any]], 
                                   success: bool, duration: float, cost: float) -> Dict[str, Any]:
        """Learn action patterns from session"""
        
        results = {"created": 0, "updated": 0, "insights": []}
        
        try:
            # Extract meaningful action sequences
            action_sequences = self._extract_action_sequences(action_history)
            
            for sequence in action_sequences:
                # Generate pattern ID
                pattern_id = self._generate_pattern_id(goal_type, domain, sequence)
                
                # Check if pattern already exists
                existing_pattern = await self.knowledge_base.get_pattern(pattern_id)
                
                if existing_pattern:
                    # Update existing pattern
                    updated_pattern = self._update_pattern_statistics(
                        existing_pattern, success, duration, cost
                    )
                    await self.knowledge_base.store_pattern(updated_pattern)
                    results["updated"] += 1
                    results["insights"].append(f"Updated pattern {pattern_id} with new data")
                else:
                    # Create new pattern
                    new_pattern = AutomationPattern(
                        pattern_id=pattern_id,
                        pattern_type="success" if success else "failure",
                        goal_type=goal_type,
                        site_domain=domain,
                        action_sequence=sequence,
                        success_rate=1.0 if success else 0.0,
                        avg_duration=duration,
                        avg_cost=cost,
                        confidence_score=self._calculate_pattern_confidence(sequence, success),
                        usage_count=1,
                        last_used=datetime.now()
                    )
                    
                    await self.knowledge_base.store_pattern(new_pattern)
                    results["created"] += 1
                    results["insights"].append(f"Created new pattern {pattern_id}")
            
        except Exception as e:
            logger.error(f"Failed to learn action patterns: {e}")
            results["error"] = str(e)
        
        return results
    
    async def _update_site_knowledge(self, domain: str, action_history: List[Dict[str, Any]], 
                                   success: bool, duration: float, errors: List[str]) -> bool:
        """Update site-specific knowledge"""
        
        try:
            # Get existing site knowledge or create new
            site_knowledge = await self.knowledge_base.get_site_knowledge(domain)
            
            if not site_knowledge:
                site_knowledge = SiteKnowledge(
                    domain=domain,
                    common_elements={},
                    timing_patterns={},
                    success_indicators=[],
                    failure_indicators=[],
                    optimization_rules=[],
                    complexity_score=5.0,
                    reliability_score=0.5,
                    session_count=0
                )
            
            # Update with new session data
            site_knowledge.session_count += 1
            
            # Update reliability score
            current_reliability = site_knowledge.reliability_score
            session_success = 1.0 if success else 0.0
            site_knowledge.reliability_score = (
                (current_reliability * (site_knowledge.session_count - 1) + session_success) / 
                site_knowledge.session_count
            )
            
            # Extract and update common elements
            elements = self._extract_elements_from_actions(action_history)
            for element_type, selectors in elements.items():
                if element_type not in site_knowledge.common_elements:
                    site_knowledge.common_elements[element_type] = []
                
                for selector in selectors:
                    if selector not in site_knowledge.common_elements[element_type]:
                        site_knowledge.common_elements[element_type].append(selector)
            
            # Update timing patterns
            timing_data = self._extract_timing_patterns(action_history)
            for action_type, wait_time in timing_data.items():
                if action_type in site_knowledge.timing_patterns:
                    # Average with existing timing
                    existing_time = site_knowledge.timing_patterns[action_type]
                    site_knowledge.timing_patterns[action_type] = (existing_time + wait_time) / 2
                else:
                    site_knowledge.timing_patterns[action_type] = wait_time
            
            # Update complexity score based on session difficulty
            session_complexity = len(action_history) / 10.0  # Normalize by typical action count
            site_knowledge.complexity_score = (
                (site_knowledge.complexity_score * 0.8) + (session_complexity * 0.2)
            )
            
            site_knowledge.last_updated = datetime.now()
            
            # Store updated knowledge
            return await self.knowledge_base.store_site_knowledge(site_knowledge)
            
        except Exception as e:
            logger.error(f"Failed to update site knowledge for {domain}: {e}")
            return False
    
    async def _update_goal_template(self, goal_type: str, action_history: List[Dict[str, Any]], 
                                  success: bool, duration: float) -> bool:
        """Update goal template with new session data"""
        
        try:
            # Get existing template or create new
            template = await self.knowledge_base.get_goal_template(goal_type)
            
            if not template:
                template = GoalTemplate(
                    goal_type=goal_type,
                    canonical_steps=self._extract_canonical_steps(action_history),
                    success_rate=0.5,
                    avg_duration=30.0,
                    complexity_score=5.0,
                    required_elements=[],
                    common_variations=[],
                    optimization_hints=[],
                    usage_count=0
                )
            
            # Update statistics
            template.usage_count += 1
            
            # Update success rate
            current_success_rate = template.success_rate
            session_success = 1.0 if success else 0.0
            template.success_rate = (
                (current_success_rate * (template.usage_count - 1) + session_success) / 
                template.usage_count
            )
            
            # Update average duration
            template.avg_duration = (
                (template.avg_duration * (template.usage_count - 1) + duration) / 
                template.usage_count
            )
            
            # Update canonical steps if this was a successful session
            if success:
                new_steps = self._extract_canonical_steps(action_history)
                template.canonical_steps = self._merge_canonical_steps(
                    template.canonical_steps, new_steps
                )
            
            # Update complexity score
            session_complexity = len(action_history) / 10.0
            template.complexity_score = (
                (template.complexity_score * 0.9) + (session_complexity * 0.1)
            )
            
            template.last_updated = datetime.now()
            
            # Store updated template
            return await self.knowledge_base.store_goal_template(template)
            
        except Exception as e:
            logger.error(f"Failed to update goal template for {goal_type}: {e}")
            return False
    
    async def _learn_from_failures(self, goal_type: str, domain: str, 
                                 action_history: List[Dict[str, Any]], 
                                 errors: List[str]) -> List[str]:
        """Learn from failed automation sessions"""
        
        insights = []
        
        try:
            # Create failure pattern
            failure_sequence = self._extract_failure_sequence(action_history, errors)
            
            if failure_sequence:
                pattern_id = self._generate_pattern_id(goal_type, domain, failure_sequence, "failure")
                
                failure_pattern = AutomationPattern(
                    pattern_id=pattern_id,
                    pattern_type="failure",
                    goal_type=goal_type,
                    site_domain=domain,
                    action_sequence=failure_sequence,
                    success_rate=0.0,
                    avg_duration=0.0,
                    avg_cost=0.0,
                    confidence_score=0.8,  # High confidence in failure patterns
                    usage_count=1,
                    metadata={"errors": errors}
                )
                
                await self.knowledge_base.store_pattern(failure_pattern)
                insights.append(f"Learned failure pattern: {pattern_id}")
            
            # Analyze error patterns
            error_insights = self._analyze_error_patterns(errors)
            insights.extend(error_insights)
            
        except Exception as e:
            logger.error(f"Failed to learn from failures: {e}")
            insights.append(f"Error learning from failures: {str(e)}")
        
        return insights
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return "unknown"
    
    def _classify_goal_type(self, goal: str) -> str:
        """Classify goal into a type category"""
        goal_lower = goal.lower()
        
        if any(word in goal_lower for word in ["login", "sign in", "authenticate"]):
            return "authentication"
        elif any(word in goal_lower for word in ["search", "find", "look for"]):
            return "search"
        elif any(word in goal_lower for word in ["buy", "purchase", "order", "checkout"]):
            return "ecommerce"
        elif any(word in goal_lower for word in ["form", "submit", "fill", "register"]):
            return "form_submission"
        elif any(word in goal_lower for word in ["navigate", "go to", "visit"]):
            return "navigation"
        else:
            return "general"
    
    def _extract_action_sequences(self, action_history: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Extract meaningful action sequences from history"""
        
        sequences = []
        current_sequence = []
        
        for action in action_history:
            current_sequence.append(action)
            
            # Break sequence on navigation or significant delays
            if (action.get("action_type") == "navigate" or 
                action.get("wait_time", 0) > 5.0):
                
                if len(current_sequence) >= 2:  # Minimum sequence length
                    sequences.append(current_sequence.copy())
                current_sequence = []
        
        # Add final sequence if it exists
        if len(current_sequence) >= 2:
            sequences.append(current_sequence)
        
        return sequences
    
    def _generate_pattern_id(self, goal_type: str, domain: str, 
                           sequence: List[Dict[str, Any]], 
                           pattern_type: str = "general") -> str:
        """Generate unique pattern ID"""
        
        # Create hash from sequence characteristics
        sequence_str = "".join([
            action.get("action_type", "") + action.get("selector", "")
            for action in sequence
        ])
        
        hash_input = f"{goal_type}:{domain}:{pattern_type}:{sequence_str}"
        pattern_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        
        return f"{goal_type}_{domain}_{pattern_type}_{pattern_hash}"
    
    def _update_pattern_statistics(self, pattern: AutomationPattern, 
                                 success: bool, duration: float, cost: float) -> AutomationPattern:
        """Update pattern statistics with new session data"""
        
        # Update success rate
        total_sessions = pattern.usage_count + 1
        current_successes = pattern.success_rate * pattern.usage_count
        new_successes = current_successes + (1 if success else 0)
        pattern.success_rate = new_successes / total_sessions
        
        # Update average duration
        pattern.avg_duration = (
            (pattern.avg_duration * pattern.usage_count + duration) / total_sessions
        )
        
        # Update average cost
        pattern.avg_cost = (
            (pattern.avg_cost * pattern.usage_count + cost) / total_sessions
        )
        
        # Update confidence score
        pattern.confidence_score = min(1.0, pattern.confidence_score + 0.1)
        
        # Update usage statistics
        pattern.usage_count = total_sessions
        pattern.last_used = datetime.now()
        pattern.updated_at = datetime.now()
        
        return pattern
    
    def _calculate_pattern_confidence(self, sequence: List[Dict[str, Any]], success: bool) -> float:
        """Calculate confidence score for a pattern"""
        
        base_confidence = 0.7 if success else 0.5
        
        # Adjust based on sequence characteristics
        sequence_length_factor = min(1.0, len(sequence) / 5.0)  # Longer sequences are more specific
        selector_quality_factor = self._assess_selector_quality(sequence)
        
        confidence = base_confidence * sequence_length_factor * selector_quality_factor
        
        return min(1.0, max(0.1, confidence))
    
    def _assess_selector_quality(self, sequence: List[Dict[str, Any]]) -> float:
        """Assess the quality of selectors in a sequence"""
        
        if not sequence:
            return 0.5
        
        quality_scores = []
        
        for action in sequence:
            selector = action.get("selector", "")
            
            # Prefer ID selectors
            if "#" in selector:
                quality_scores.append(1.0)
            # Class selectors are okay
            elif "." in selector:
                quality_scores.append(0.8)
            # Attribute selectors are decent
            elif "[" in selector:
                quality_scores.append(0.7)
            # Tag selectors are less reliable
            else:
                quality_scores.append(0.5)
        
        return sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
    
    def _extract_elements_from_actions(self, action_history: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Extract element information from action history"""
        
        elements = defaultdict(list)
        
        for action in action_history:
            action_type = action.get("action_type", "")
            selector = action.get("selector", "")
            
            if selector:
                elements[action_type].append(selector)
        
        return dict(elements)
    
    def _extract_timing_patterns(self, action_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extract timing patterns from action history"""
        
        timing_data = defaultdict(list)
        
        for action in action_history:
            action_type = action.get("action_type", "")
            wait_time = action.get("wait_time", 0.0)
            
            if wait_time > 0:
                timing_data[action_type].append(wait_time)
        
        # Calculate averages
        return {
            action_type: sum(times) / len(times)
            for action_type, times in timing_data.items()
            if times
        }
    
    def _extract_canonical_steps(self, action_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract canonical steps from action history"""
        
        canonical_steps = []
        
        for action in action_history:
            canonical_step = {
                "action_type": action.get("action_type", ""),
                "element_type": self._infer_element_type(action.get("selector", "")),
                "purpose": action.get("purpose", ""),
                "required": True
            }
            canonical_steps.append(canonical_step)
        
        return canonical_steps
    
    def _merge_canonical_steps(self, existing_steps: List[Dict[str, Any]], 
                             new_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge new steps with existing canonical steps"""
        
        # Simple merge strategy - could be more sophisticated
        if not existing_steps:
            return new_steps
        
        # For now, just return the longer sequence
        return existing_steps if len(existing_steps) >= len(new_steps) else new_steps
    
    def _extract_failure_sequence(self, action_history: List[Dict[str, Any]], 
                                errors: List[str]) -> List[Dict[str, Any]]:
        """Extract the sequence leading to failure"""
        
        # Return the last few actions before failure
        failure_window = min(5, len(action_history))
        failure_sequence = action_history[-failure_window:] if action_history else []
        
        # Add error context
        if failure_sequence:
            failure_sequence[-1]["errors"] = errors
        
        return failure_sequence
    
    def _analyze_error_patterns(self, errors: List[str]) -> List[str]:
        """Analyze error patterns to generate insights"""
        
        insights = []
        
        for error in errors:
            error_lower = error.lower()
            
            if "timeout" in error_lower:
                insights.append("Consider increasing timeout values for this site")
            elif "element not found" in error_lower:
                insights.append("Element selectors may need updating")
            elif "network" in error_lower:
                insights.append("Network issues detected - consider retry logic")
            elif "permission" in error_lower or "access" in error_lower:
                insights.append("Access permissions may be required")
        
        return insights
    
    def _infer_element_type(self, selector: str) -> str:
        """Infer element type from selector"""
        
        if "button" in selector.lower() or "btn" in selector.lower():
            return "button"
        elif "input" in selector.lower():
            return "input"
        elif "form" in selector.lower():
            return "form"
        elif "link" in selector.lower() or "a[" in selector:
            return "link"
        else:
            return "element"
    
    async def get_recommendations(self, goal: str, url: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get recommendations based on learned knowledge"""
        
        domain = self._extract_domain(url)
        goal_type = self._classify_goal_type(goal)
        
        recommendations = {
            "patterns": [],
            "site_insights": {},
            "goal_template": {},
            "optimization_suggestions": []
        }
        
        try:
            # Find relevant patterns
            patterns = await self.knowledge_base.find_patterns(
                goal_type=goal_type,
                site_domain=domain,
                min_success_rate=0.6,
                limit=5
            )
            
            recommendations["patterns"] = [
                {
                    "pattern_id": p.pattern_id,
                    "success_rate": p.success_rate,
                    "avg_duration": p.avg_duration,
                    "confidence_score": p.confidence_score,
                    "usage_count": p.usage_count
                }
                for p in patterns
            ]
            
            # Get site knowledge
            site_knowledge = await self.knowledge_base.get_site_knowledge(domain)
            if site_knowledge:
                recommendations["site_insights"] = {
                    "reliability_score": site_knowledge.reliability_score,
                    "complexity_score": site_knowledge.complexity_score,
                    "common_elements": site_knowledge.common_elements,
                    "timing_patterns": site_knowledge.timing_patterns
                }
            
            # Get goal template
            goal_template = await self.knowledge_base.get_goal_template(goal_type)
            if goal_template:
                recommendations["goal_template"] = {
                    "success_rate": goal_template.success_rate,
                    "avg_duration": goal_template.avg_duration,
                    "complexity_score": goal_template.complexity_score,
                    "canonical_steps": goal_template.canonical_steps,
                    "optimization_hints": goal_template.optimization_hints
                }
            
            # Generate optimization suggestions
            recommendations["optimization_suggestions"] = await self._generate_optimization_suggestions(
                goal_type, domain, site_knowledge, goal_template, patterns
            )
            
        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            recommendations["error"] = str(e)
        
        return recommendations
    
    async def _generate_optimization_suggestions(self, goal_type: str, domain: str,
                                               site_knowledge: Optional[SiteKnowledge],
                                               goal_template: Optional[GoalTemplate],
                                               patterns: List[AutomationPattern]) -> List[str]:
        """Generate optimization suggestions based on knowledge"""
        
        suggestions = []
        
        # Site-based suggestions
        if site_knowledge:
            if site_knowledge.reliability_score < 0.7:
                suggestions.append(f"Site {domain} has low reliability ({site_knowledge.reliability_score:.1%}) - use conservative strategy")
            
            if site_knowledge.complexity_score > 7:
                suggestions.append(f"Site {domain} is complex - increase timeouts and use explicit waits")
            
            # Timing suggestions
            if site_knowledge.timing_patterns:
                avg_wait = sum(site_knowledge.timing_patterns.values()) / len(site_knowledge.timing_patterns)
                if avg_wait > 3.0:
                    suggestions.append(f"Site {domain} requires longer waits (avg: {avg_wait:.1f}s)")
        
        # Goal-based suggestions
        if goal_template:
            if goal_template.success_rate < 0.8:
                suggestions.append(f"Goal type '{goal_type}' has moderate success rate ({goal_template.success_rate:.1%}) - consider fallback strategies")
            
            if goal_template.avg_duration > 60:
                suggestions.append(f"Goal type '{goal_type}' typically takes {goal_template.avg_duration:.0f}s - plan accordingly")
        
        # Pattern-based suggestions
        if patterns:
            best_pattern = max(patterns, key=lambda p: p.success_rate * p.confidence_score)
            suggestions.append(f"Use pattern {best_pattern.pattern_id} (success rate: {best_pattern.success_rate:.1%})")
            
            if len(patterns) > 1:
                suggestions.append(f"Multiple patterns available - consider A/B testing")
        
        return suggestions