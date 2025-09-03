"""
Study Designer Tool

Provides intelligent guidance for clinical research study design:
- Research question analysis
- Appropriate study type recommendations  
- Variable selection guidance
- Analysis plan generation
"""

from typing import Dict, List, Optional, Any
import re


class StudyDesigner:
    """
    Intelligent study design assistant for clinical research.
    
    Analyzes research questions and provides methodological guidance
    following best practices in clinical research.
    """
    
    def __init__(self):
        self.study_types = {
            'descriptive': {
                'description': 'Describe characteristics of a population or phenomenon',
                'methods': ['cross_sectional', 'case_series', 'ecological'],
                'reporting_standard': 'STROBE'
            },
            'analytical': {
                'description': 'Examine associations between variables',
                'methods': ['cohort', 'case_control', 'cross_sectional_analytical'],
                'reporting_standard': 'STROBE'
            },
            'predictive': {
                'description': 'Develop or validate prediction models',
                'methods': ['prediction_model_development', 'prediction_model_validation'],
                'reporting_standard': 'TRIPOD+AI'
            }
        }
        
        # Common clinical variables by category
        self.variable_categories = {
            'demographics': ['age', 'sex', 'race', 'ethnicity'],
            'comorbidities': ['diabetes', 'hypertension', 'coronary_artery_disease', 'copd', 'heart_failure'],
            'severity': ['apache_score', 'sofa_score', 'glasgow_coma_scale'],
            'outcomes': ['mortality', 'icu_mortality', 'hospital_mortality', 'los_days', 'ventilator_days'],
            'laboratory': ['creatinine', 'bun', 'lactate', 'hemoglobin', 'white_blood_cell'],
            'medications': ['vasopressors', 'antibiotics', 'steroids', 'sedatives']
        }
    
    def create_study_plan(self, 
                         research_question: str,
                         study_type: Optional[str] = None,
                         outcome_type: Optional[str] = None,
                         available_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive study design plan.
        
        Args:
            research_question: Primary research question
            study_type: Specified study type or None for auto-detection
            outcome_type: Type of primary outcome
            available_data: Information about available datasets
            
        Returns:
            Comprehensive study design plan
        """
        # Analyze research question
        question_analysis = self._analyze_research_question(research_question)
        
        # Determine study type
        if study_type is None:
            study_type = self._recommend_study_type(question_analysis, outcome_type)
        
        # Generate variable recommendations
        variable_recommendations = self._recommend_variables(
            question_analysis, study_type, available_data
        )
        
        # Create analysis plan
        analysis_plan = self._create_analysis_plan(study_type, outcome_type, question_analysis)
        
        # Sample size guidance
        sample_size_guidance = self._provide_sample_size_guidance(study_type, outcome_type)
        
        return {
            'research_question': research_question,
            'question_analysis': question_analysis,
            'study_type': study_type,
            'reporting_standard': self.study_types[study_type]['reporting_standard'],
            'suggested_outcome': variable_recommendations['primary_outcome'],
            'suggested_predictors': variable_recommendations['predictors'],
            'potential_confounders': variable_recommendations['confounders'],
            'analysis_steps': analysis_plan,
            'sample_size_guidance': sample_size_guidance,
            'primary_analysis': self._determine_primary_analysis(study_type, outcome_type)
        }
    
    def _analyze_research_question(self, question: str) -> Dict[str, Any]:
        """Analyze research question to extract key components."""
        question_lower = question.lower()
        
        # Detect key question types
        question_patterns = {
            'association': ['associat', 'correlat', 'relationship', 'related'],
            'causation': ['cause', 'effect', 'impact', 'influence'],
            'prediction': ['predict', 'forecast', 'risk', 'probability'],
            'comparison': ['compare', 'difference', 'versus', 'vs'],
            'description': ['describe', 'characterize', 'prevalence', 'incidence']
        }
        
        detected_types = []
        for q_type, patterns in question_patterns.items():
            if any(pattern in question_lower for pattern in patterns):
                detected_types.append(q_type)
        
        # Extract potential variables mentioned
        clinical_terms = self._extract_clinical_terms(question)
        
        # Detect population
        population_terms = ['patient', 'adult', 'pediatric', 'elderly', 'icu', 'critical']
        detected_population = [term for term in population_terms if term in question_lower]
        
        return {
            'question_types': detected_types,
            'clinical_terms': clinical_terms,
            'population': detected_population,
            'complexity': self._assess_question_complexity(question)
        }
    
    def _extract_clinical_terms(self, question: str) -> List[str]:
        """Extract clinical terms from research question."""
        question_lower = question.lower()
        
        clinical_terms = []
        
        # Check all variable categories
        for category, variables in self.variable_categories.items():
            for variable in variables:
                if variable.replace('_', ' ') in question_lower:
                    clinical_terms.append(variable)
        
        # Common clinical abbreviations
        abbreviations = {
            'aki': 'acute kidney injury',
            'ards': 'acute respiratory distress syndrome', 
            'sepsis': 'sepsis',
            'shock': 'shock',
            'mortality': 'mortality',
            'los': 'length of stay'
        }
        
        for abbrev, full_term in abbreviations.items():
            if abbrev in question_lower or full_term in question_lower:
                clinical_terms.append(abbrev)
        
        return list(set(clinical_terms))  # Remove duplicates
    
    def _recommend_study_type(self, question_analysis: Dict[str, Any], 
                             outcome_type: Optional[str]) -> str:
        """Recommend appropriate study type based on question analysis."""
        question_types = question_analysis['question_types']
        
        # Prediction questions → predictive study
        if 'prediction' in question_types:
            return 'predictive'
        
        # Association/causation questions → analytical study  
        if any(q_type in question_types for q_type in ['association', 'causation', 'comparison']):
            return 'analytical'
        
        # Description questions → descriptive study
        if 'description' in question_types:
            return 'descriptive'
        
        # Default based on outcome type
        if outcome_type == 'binary' and len(question_analysis['clinical_terms']) > 3:
            return 'predictive'
        elif question_analysis['complexity'] > 3:
            return 'analytical'
        else:
            return 'descriptive'
    
    def _recommend_variables(self, question_analysis: Dict[str, Any],
                           study_type: str,
                           available_data: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Recommend variables based on study design."""
        
        # Start with terms mentioned in research question
        mentioned_terms = question_analysis['clinical_terms']
        
        # Determine primary outcome
        outcome_candidates = [term for term in mentioned_terms 
                            if term in self.variable_categories['outcomes']]
        
        if outcome_candidates:
            primary_outcome = outcome_candidates[0]
        else:
            # Default outcomes by study type
            if study_type == 'predictive':
                primary_outcome = 'mortality'
            elif study_type == 'analytical':
                primary_outcome = 'icu_mortality' 
            else:
                primary_outcome = 'los_days'
        
        # Predictor variables
        predictors = []
        
        # Always include demographics
        predictors.extend(self.variable_categories['demographics'])
        
        # Add clinical variables based on study focus
        if any(term in mentioned_terms for term in ['sepsis', 'infection']):
            predictors.extend(['lactate', 'white_blood_cell', 'antibiotics'])
        
        if any(term in mentioned_terms for term in ['aki', 'kidney', 'renal']):
            predictors.extend(['creatinine', 'bun'])
        
        if study_type == 'predictive':
            # Include more predictors for prediction models
            predictors.extend(self.variable_categories['severity'])
            predictors.extend(self.variable_categories['laboratory'][:3])
        
        # Potential confounders
        confounders = self.variable_categories['demographics'] + self.variable_categories['comorbidities'][:3]
        
        # Filter based on available data
        if available_data:
            available_vars = []
            for dataset in available_data.values():
                if hasattr(dataset, 'columns'):
                    available_vars.extend(dataset.columns.tolist())
            
            # Keep only available variables
            predictors = [var for var in predictors if var in available_vars]
            confounders = [var for var in confounders if var in available_vars]
        
        return {
            'primary_outcome': primary_outcome,
            'predictors': list(set(predictors))[:10],  # Limit to 10
            'confounders': list(set(confounders))[:5]   # Limit to 5
        }
    
    def _create_analysis_plan(self, study_type: str, outcome_type: Optional[str],
                            question_analysis: Dict[str, Any]) -> List[str]:
        """Create step-by-step analysis plan."""
        
        base_steps = [
            "Explore data and assess quality, including missing data patterns",
            "Generate Table 1 with baseline characteristics",
        ]
        
        if study_type == 'descriptive':
            analysis_steps = base_steps + [
                "Calculate descriptive statistics for key variables",
                "Create visualizations of distributions and patterns",
                "Assess clinical significance of findings"
            ]
        
        elif study_type == 'analytical':
            analysis_steps = base_steps + [
                "Test bivariate associations between predictors and outcome",
                "Fit multivariable regression model with appropriate type",
                "Check model assumptions and perform diagnostics",
                "Conduct sensitivity analyses to test robustness"
            ]
        
        elif study_type == 'predictive':
            analysis_steps = base_steps + [
                "Build prediction model using appropriate algorithm",
                "Validate model performance with cross-validation",
                "Assess model calibration and discrimination",
                "Generate model interpretability analysis",
                "Test model in validation dataset if available"
            ]
        
        else:
            analysis_steps = base_steps + [
                "Perform appropriate statistical analyses",
                "Interpret results in clinical context"
            ]
        
        return analysis_steps
    
    def _provide_sample_size_guidance(self, study_type: str, 
                                    outcome_type: Optional[str]) -> str:
        """Provide sample size guidance based on study type."""
        
        if study_type == 'predictive':
            return ("For prediction models, aim for at least 10-20 events per predictor variable. "
                   "Consider Events Per Variable (EPV) ratio and total sample size requirements.")
        
        elif study_type == 'analytical':
            if outcome_type == 'binary':
                return ("For logistic regression, target at least 10 events per predictor variable. "
                       "Consider power analysis for detecting clinically meaningful effect sizes.")
            else:
                return ("For linear regression, aim for at least 10-15 observations per predictor. "
                       "Power analysis should consider expected effect sizes and variance.")
        
        else:
            return ("For descriptive studies, sample size should provide adequate precision "
                   "for key estimates. Consider confidence interval width for main measures.")
    
    def _determine_primary_analysis(self, study_type: str, outcome_type: Optional[str]) -> str:
        """Determine primary analysis method."""
        
        if study_type == 'predictive':
            return 'machine_learning_model'
        elif study_type == 'analytical':
            if outcome_type == 'binary':
                return 'logistic_regression'
            elif outcome_type == 'time_to_event':
                return 'cox_regression'
            else:
                return 'linear_regression'
        else:
            return 'descriptive_statistics'
    
    def _assess_question_complexity(self, question: str) -> int:
        """Assess complexity of research question (0-5 scale)."""
        complexity_indicators = [
            len(re.findall(r'\b(and|or|versus|vs|compared|relationship)\b', question.lower())),
            len(re.findall(r'\b(predict|model|risk|factor)\b', question.lower())),
            len(re.findall(r'\b(adjust|control|confound)\b', question.lower())),
            int('interaction' in question.lower()),
            int(len(question.split()) > 20)
        ]
        
        return sum(complexity_indicators)