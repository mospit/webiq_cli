import re
import logging
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse

# Mock spacy import
try:
    import spacy
except ImportError:
    class MockSpacy:
        def load(self, model):
            return MockNLP()
    
    class MockNLP:
        def __call__(self, text):
            return MockDoc(text)
    
    class MockDoc:
        def __init__(self, text):
            self.text = text
            self.ents = [MockEntity("test@email.com", "EMAIL")]
    
    class MockEntity:
        def __init__(self, text, label):
            self.text = text
            self.label_ = label
    
    spacy = MockSpacy()

logger = logging.getLogger(__name__)

class RequirementExtractor:
    """Extract explicit and implicit requirements from goals"""
    
    def __init__(self):
        self.requirement_patterns = self._initialize_requirement_patterns()
        self.implicit_rules = self._initialize_implicit_rules()
        self.nlp_model = spacy.load("en_core_web_sm")
    
    def extract_explicit(self, goal: str) -> List[str]:
        """Extract explicitly stated requirements"""
        requirements = []
        
        # Pattern-based extraction
        for pattern, req_type in self.requirement_patterns.items():
            matches = re.findall(pattern, goal.lower())
            for match in matches:
                if isinstance(match, tuple):
                    requirements.append(f"{req_type}: {' '.join(match)}")
                else:
                    requirements.append(f"{req_type}: {match}")
        
        # Entity-based extraction using NLP
        doc = self.nlp_model(goal)
        for ent in doc.ents:
            if ent.label_ in ["EMAIL", "PHONE", "PERSON", "ORG"]:
                requirements.append(f"data_input: {ent.text}")
        
        # Quantity and measurement extraction
        quantity_patterns = [
            r"(\d+)\s+(items?|products?|results?)",
            r"(\d+)\s+(minutes?|hours?|seconds?)",
            r"(\d+)\s+(times?|attempts?)"
        ]
        
        for pattern in quantity_patterns:
            matches = re.findall(pattern, goal.lower())
            for match in matches:
                requirements.append(f"quantity_requirement: {' '.join(match)}")
        
        # Conditional requirements
        conditional_patterns = [
            r"if\s+([^,]+),?\s+then\s+([^.]+)",
            r"when\s+([^,]+),?\s+([^.]+)",
            r"unless\s+([^,]+),?\s+([^.]+)"
        ]
        
        for pattern in conditional_patterns:
            matches = re.findall(pattern, goal.lower())
            for match in matches:
                requirements.append(f"conditional_requirement: {match[0]} -> {match[1]}")
        
        return requirements
    
    async def infer_implicit(self, goal: str, url: str, context: Dict[str, Any]) -> List[str]:
        """Infer implicit requirements based on goal and context"""
        implicit_requirements = []
        goal_lower = goal.lower()
        domain = self._extract_domain(url)
        
        # Authentication-related implicit requirements
        if any(word in goal_lower for word in ["signup", "register", "create account"]):
            implicit_requirements.extend([
                "email_verification_may_be_required",
                "password_requirements_compliance",
                "terms_of_service_acceptance",
                "captcha_solving_capability",
                "unique_username_validation",
                "age_verification_may_be_required"
            ])
        
        if any(word in goal_lower for word in ["login", "sign in", "authenticate"]):
            implicit_requirements.extend([
                "credential_validation",
                "session_management",
                "two_factor_auth_handling",
                "remember_me_option_handling",
                "password_reset_capability",
                "account_lockout_handling"
            ])
        
        # Transaction-related implicit requirements
        if any(word in goal_lower for word in ["purchase", "buy", "checkout", "order"]):
            implicit_requirements.extend([
                "payment_information_entry",
                "address_verification",
                "order_confirmation_handling",
                "fraud_protection_compliance",
                "tax_calculation_handling",
                "shipping_option_selection",
                "inventory_availability_check",
                "price_change_handling"
            ])
        
        # Form-related implicit requirements
        if any(word in goal_lower for word in ["fill", "submit", "complete", "form"]):
            implicit_requirements.extend([
                "field_validation_handling",
                "required_field_completion",
                "format_validation",
                "error_message_handling",
                "progress_saving_capability",
                "auto_save_functionality"
            ])
        
        # Search-related implicit requirements
        if any(word in goal_lower for word in ["search", "find", "look for", "locate"]):
            implicit_requirements.extend([
                "search_result_pagination",
                "filter_application",
                "sort_option_handling",
                "no_results_handling",
                "search_suggestion_handling",
                "advanced_search_capability"
            ])
        
        # File handling implicit requirements
        if any(word in goal_lower for word in ["upload", "download", "attach", "file"]):
            implicit_requirements.extend([
                "file_type_validation",
                "file_size_limit_compliance",
                "upload_progress_monitoring",
                "file_preview_capability",
                "virus_scanning_compliance",
                "multiple_file_handling"
            ])
        
        # Domain-specific implicit requirements
        if "bank" in domain or "financial" in domain:
            implicit_requirements.extend([
                "enhanced_security_compliance",
                "session_timeout_handling",
                "multi_factor_authentication",
                "transaction_limit_compliance",
                "regulatory_compliance_checks",
                "audit_trail_generation"
            ])
        
        if "gov" in domain or "government" in domain:
            implicit_requirements.extend([
                "accessibility_compliance",
                "security_clearance_may_be_required",
                "formal_validation_processes",
                "document_verification",
                "citizen_identification_required",
                "privacy_act_compliance"
            ])
        
        if "healthcare" in domain or "medical" in domain:
            implicit_requirements.extend([
                "hipaa_compliance",
                "patient_consent_handling",
                "medical_record_security",
                "appointment_scheduling_rules",
                "insurance_verification"
            ])
        
        if "education" in domain or "school" in domain or "university" in domain:
            implicit_requirements.extend([
                "student_verification",
                "academic_calendar_compliance",
                "grade_privacy_protection",
                "enrollment_status_validation",
                "ferpa_compliance"
            ])
        
        # E-commerce specific requirements
        if any(indicator in domain for indicator in ["shop", "store", "buy", "cart", "commerce"]):
            implicit_requirements.extend([
                "cart_persistence",
                "wishlist_functionality",
                "product_comparison",
                "review_and_rating_handling",
                "recommendation_system_interaction",
                "loyalty_program_integration"
            ])
        
        # Social media requirements
        if any(indicator in domain for indicator in ["facebook", "twitter", "linkedin", "instagram", "social"]):
            implicit_requirements.extend([
                "privacy_settings_compliance",
                "content_moderation_awareness",
                "friend_connection_handling",
                "notification_management",
                "content_sharing_permissions"
            ])
        
        # Context-based implicit requirements
        if context.get("mobile_device", False):
            implicit_requirements.extend([
                "touch_interface_optimization",
                "responsive_design_handling",
                "mobile_keyboard_adaptation",
                "gesture_recognition",
                "orientation_change_handling"
            ])
        
        if context.get("slow_connection", False):
            implicit_requirements.extend([
                "progressive_loading_handling",
                "offline_capability_awareness",
                "data_usage_optimization",
                "timeout_extension_handling"
            ])
        
        if context.get("accessibility_required", False):
            implicit_requirements.extend([
                "screen_reader_compatibility",
                "keyboard_navigation_support",
                "high_contrast_mode_handling",
                "text_size_adaptation",
                "alt_text_utilization"
            ])
        
        # Time-sensitive requirements
        if any(word in goal_lower for word in ["urgent", "asap", "immediately", "quickly"]):
            implicit_requirements.extend([
                "performance_optimization_priority",
                "minimal_user_interaction",
                "automated_decision_making",
                "error_recovery_acceleration"
            ])
        
        # Data handling requirements
        if any(word in goal_lower for word in ["extract", "scrape", "collect", "gather", "data"]):
            implicit_requirements.extend([
                "data_format_standardization",
                "duplicate_detection",
                "data_quality_validation",
                "rate_limiting_compliance",
                "robots_txt_compliance",
                "copyright_respect"
            ])
        
        return implicit_requirements
    
    def extract_constraints(self, goal: str, context: Dict[str, Any]) -> List[str]:
        """Extract constraints from goal and context"""
        constraints = []
        goal_lower = goal.lower()
        
        # Time constraints
        time_patterns = [
            r"within (\d+) (minutes?|hours?|seconds?|days?)",
            r"in (\d+) (minutes?|hours?|seconds?|days?)",
            r"before (\d+):(\d+)",
            r"by (\w+day)"
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, goal_lower)
            for match in matches:
                if isinstance(match, tuple):
                    constraints.append(f"time_constraint: {' '.join(match)}")
                else:
                    constraints.append(f"time_constraint: {match}")
        
        # Quality constraints
        if any(word in goal_lower for word in ["carefully", "accurate", "precise", "exact"]):
            constraints.append("accuracy_prioritized")
        
        if any(word in goal_lower for word in ["quickly", "fast", "rapid", "speed"]):
            constraints.append("speed_prioritized")
        
        if any(word in goal_lower for word in ["secure", "safe", "protected", "private"]):
            constraints.append("security_prioritized")
        
        # Budget constraints
        budget_patterns = [
            r"under \$?(\d+)",
            r"less than \$?(\d+)",
            r"maximum \$?(\d+)",
            r"budget of \$?(\d+)"
        ]
        
        for pattern in budget_patterns:
            matches = re.findall(pattern, goal_lower)
            for match in matches:
                constraints.append(f"budget_constraint: ${match}")
        
        # Quantity constraints
        quantity_patterns = [
            r"at least (\d+)",
            r"minimum (\d+)",
            r"maximum (\d+)",
            r"exactly (\d+)",
            r"no more than (\d+)"
        ]
        
        for pattern in quantity_patterns:
            matches = re.findall(pattern, goal_lower)
            for match in matches:
                constraints.append(f"quantity_constraint: {match}")
        
        # Context-based constraints
        if context.get("headless_mode", False):
            constraints.append("headless_execution_required")
        
        if context.get("mobile_device", False):
            constraints.append("mobile_optimized_interaction")
        
        if context.get("slow_connection", False):
            constraints.append("bandwidth_limited_execution")
        
        if context.get("restricted_permissions", False):
            constraints.append("limited_permission_execution")
        
        if context.get("read_only_mode", False):
            constraints.append("no_data_modification_allowed")
        
        # Browser constraints
        browser_type = context.get("browser_type", "")
        if browser_type:
            constraints.append(f"browser_specific: {browser_type}")
        
        # Language constraints
        if any(word in goal_lower for word in ["english", "spanish", "french", "german", "chinese"]):
            language_match = re.search(r"(english|spanish|french|german|chinese)", goal_lower)
            if language_match:
                constraints.append(f"language_constraint: {language_match.group(1)}")
        
        return constraints
    
    def analyze_requirement_complexity(self, requirements: List[str]) -> Dict[str, Any]:
        """Analyze the complexity of extracted requirements"""
        complexity_analysis = {
            "total_requirements": len(requirements),
            "explicit_count": 0,
            "implicit_count": 0,
            "constraint_count": 0,
            "complexity_score": 0.0,
            "risk_factors": [],
            "categories": {}
        }
        
        # Categorize requirements
        for req in requirements:
            if ":" in req:
                category = req.split(":")[0].strip()
                complexity_analysis["categories"][category] = complexity_analysis["categories"].get(category, 0) + 1
                
                # Determine requirement type
                if category in ["specific_email", "specific_password", "data_input"]:
                    complexity_analysis["explicit_count"] += 1
                elif category.endswith("_constraint"):
                    complexity_analysis["constraint_count"] += 1
                else:
                    complexity_analysis["implicit_count"] += 1
        
        # Calculate complexity score
        base_score = len(requirements) * 0.1
        explicit_weight = complexity_analysis["explicit_count"] * 0.2
        implicit_weight = complexity_analysis["implicit_count"] * 0.15
        constraint_weight = complexity_analysis["constraint_count"] * 0.25
        
        complexity_analysis["complexity_score"] = min(base_score + explicit_weight + implicit_weight + constraint_weight, 1.0)
        
        # Identify risk factors
        high_risk_categories = ["payment_information_entry", "security_clearance_may_be_required", "financial_data"]
        for category in complexity_analysis["categories"]:
            if any(risk in category for risk in high_risk_categories):
                complexity_analysis["risk_factors"].append(category)
        
        return complexity_analysis
    
    def generate_requirement_recommendations(self, goal: str, requirements: List[str], 
                                           context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on requirement analysis"""
        recommendations = []
        
        # Check for missing common requirements
        goal_lower = goal.lower()
        
        if "login" in goal_lower and not any("credential" in req for req in requirements):
            recommendations.append({
                "type": "missing_requirement",
                "description": "Consider adding credential validation requirements",
                "priority": "medium",
                "suggested_requirement": "credential_validation"
            })
        
        if "form" in goal_lower and not any("validation" in req for req in requirements):
            recommendations.append({
                "type": "missing_requirement",
                "description": "Consider adding form validation requirements",
                "priority": "high",
                "suggested_requirement": "field_validation_handling"
            })
        
        if "purchase" in goal_lower and not any("payment" in req for req in requirements):
            recommendations.append({
                "type": "missing_requirement",
                "description": "Consider adding payment processing requirements",
                "priority": "high",
                "suggested_requirement": "payment_information_entry"
            })
        
        # Check for conflicting requirements
        speed_priority = any("speed_prioritized" in req for req in requirements)
        accuracy_priority = any("accuracy_prioritized" in req for req in requirements)
        
        if speed_priority and accuracy_priority:
            recommendations.append({
                "type": "conflicting_requirements",
                "description": "Speed and accuracy priorities may conflict",
                "priority": "medium",
                "suggestion": "Consider balancing speed and accuracy requirements"
            })
        
        # Check for security considerations
        has_sensitive_data = any(word in " ".join(requirements) for word in ["payment", "financial", "personal", "credential"])
        has_security_req = any("security" in req for req in requirements)
        
        if has_sensitive_data and not has_security_req:
            recommendations.append({
                "type": "security_recommendation",
                "description": "Sensitive data detected - consider adding security requirements",
                "priority": "high",
                "suggested_requirement": "enhanced_security_compliance"
            })
        
        return recommendations
    
    def _initialize_requirement_patterns(self) -> Dict[str, str]:
        """Initialize regex patterns for requirement extraction"""
        return {
            r"with email ([\w\.-]+@[\w\.-]+)": "specific_email",
            r"using password (\S+)": "specific_password",
            r"select (\w+) option": "specific_selection",
            r"upload ([\w\s]+) file": "file_upload",
            r"enter (\d+) digit": "numeric_input",
            r"choose (\w+) from": "dropdown_selection",
            r"within (\d+) (\w+)": "time_constraint",
            r"using (\w+) browser": "browser_requirement",
            r"in (\w+) language": "language_requirement",
            r"with (\w+) permissions": "permission_requirement",
            r"on (mobile|desktop|tablet)": "device_requirement",
            r"via (\w+) method": "method_requirement"
        }
    
    def _initialize_implicit_rules(self) -> Dict[str, List[str]]:
        """Initialize rules for implicit requirement inference"""
        return {
            "authentication_goals": [
                "session_management",
                "credential_validation",
                "redirect_handling",
                "two_factor_auth_handling"
            ],
            "form_submission_goals": [
                "field_validation",
                "error_handling",
                "submission_confirmation",
                "progress_saving"
            ],
            "transaction_goals": [
                "cart_management",
                "payment_processing",
                "order_tracking",
                "receipt_generation"
            ],
            "search_goals": [
                "result_pagination",
                "filter_application",
                "sort_handling",
                "no_results_handling"
            ],
            "data_extraction_goals": [
                "rate_limiting_compliance",
                "data_format_standardization",
                "duplicate_detection",
                "quality_validation"
            ]
        }
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            return url.lower()
    
    def get_requirement_statistics(self, requirements: List[str]) -> Dict[str, Any]:
        """Get statistical analysis of requirements"""
        stats = {
            "total_count": len(requirements),
            "category_distribution": {},
            "complexity_indicators": {
                "high_complexity_count": 0,
                "medium_complexity_count": 0,
                "low_complexity_count": 0
            },
            "security_related_count": 0,
            "performance_related_count": 0,
            "user_interaction_count": 0
        }
        
        # Analyze each requirement
        for req in requirements:
            # Category distribution
            if ": " in req:
                category = req.split(": ")[0]
                stats["category_distribution"][category] = stats["category_distribution"].get(category, 0) + 1
            
            # Complexity analysis
            if any(word in req.lower() for word in ["multi", "complex", "advanced", "comprehensive"]):
                stats["complexity_indicators"]["high_complexity_count"] += 1
            elif any(word in req.lower() for word in ["validation", "handling", "processing"]):
                stats["complexity_indicators"]["medium_complexity_count"] += 1
            else:
                stats["complexity_indicators"]["low_complexity_count"] += 1
            
            # Security analysis
            if any(word in req.lower() for word in ["security", "auth", "credential", "encryption", "privacy"]):
                stats["security_related_count"] += 1
            
            # Performance analysis
            if any(word in req.lower() for word in ["speed", "performance", "optimization", "timeout"]):
                stats["performance_related_count"] += 1
            
            # User interaction analysis
            if any(word in req.lower() for word in ["input", "click", "selection", "interaction", "interface"]):
                stats["user_interaction_count"] += 1
        
        return stats