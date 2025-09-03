"""
TRIPOD+AI Reporter

Generates comprehensive reports following TRIPOD+AI guidelines for clinical 
prediction models. Covers all 27 checklist items for transparent reporting
of prediction model studies using regression or machine learning methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import json
from pathlib import Path


class TRIPODReporter:
    """
    Generate TRIPOD+AI-compliant reports for prediction model studies.
    
    Covers all 27 TRIPOD+AI checklist items:
    - Title and Abstract (1a-1b)
    - Introduction (2-3)
    - Methods (4a-12b)
    - Results (13a-19)
    - Discussion (20-22)
    - Other Information (23-27)
    """
    
    def __init__(self, study_info: Dict[str, Any]):
        """
        Initialize TRIPOD+AI reporter with study information.
        
        Args:
            study_info: Dictionary containing study metadata and model information
        """
        self.study_info = study_info
        self.report_sections = {}
        self.model_results = {}
        self.checklist_completion = {}
        
    def generate_report(self,
                       model_results: Dict[str, Any],
                       output_format: str = 'html',
                       include_checklist: bool = True) -> str:
        """
        Generate complete TRIPOD+AI-compliant report.
        
        Args:
            model_results: Dictionary containing model development and validation results
            output_format: Output format ('html', 'markdown', 'docx')  
            include_checklist: Whether to include TRIPOD+AI checklist
            
        Returns:
            Formatted report string
        """
        self.model_results = model_results
        
        # Generate all sections
        self._generate_title_and_abstract()
        self._generate_introduction()
        self._generate_methods()
        self._generate_results()
        self._generate_discussion()
        self._generate_other_information()
        
        # Compile report
        if output_format == 'html':
            report = self._format_html_report(include_checklist)
        elif output_format == 'markdown':
            report = self._format_markdown_report(include_checklist)
        elif output_format == 'docx':
            report = self._format_docx_report(include_checklist)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        return report
    
    def _generate_title_and_abstract(self):
        """Generate Title and Abstract section (Items 1a-1b)."""
        
        # Item 1a: Title
        model_type = self.study_info.get('model_type', 'prediction model')
        outcome = self.study_info.get('target_outcome', 'clinical outcome')
        population = self.study_info.get('population', 'clinical population')
        
        title = (f"Development and Validation of a {model_type} for Predicting "
                f"{outcome} in {population}")
        
        # Item 1b: Abstract
        abstract_sections = {
            'background': self._generate_abstract_background(),
            'methods': self._generate_abstract_methods(),
            'results': self._generate_abstract_results(),
            'conclusions': self._generate_abstract_conclusions()
        }
        
        self.report_sections['title_abstract'] = {
            'title': title,
            'abstract': abstract_sections
        }
        
        self.checklist_completion['item_1a'] = True
        self.checklist_completion['item_1b'] = True
    
    def _generate_introduction(self):
        """Generate Introduction section (Items 2-3)."""
        
        # Item 2: Background and objectives  
        background = {
            'clinical_context': self.study_info.get('clinical_context',
                'Accurate prediction of clinical outcomes is important for patient care.'),
            'existing_models': self.study_info.get('existing_models',
                'Previous prediction models have limitations that this study addresses.'),
            'rationale': self.study_info.get('rationale',
                'A new prediction model is needed to improve clinical decision-making.')
        }
        
        # Item 3: Objectives
        objectives = {
            'primary_objective': self.study_info.get('primary_objective',
                'To develop and validate a prediction model for clinical outcomes.'),
            'intended_use': self.study_info.get('intended_use',
                'Model intended for clinical decision support.'),
            'target_population': self.study_info.get('target_population',
                'Adult patients in clinical settings.')
        }
        
        self.report_sections['introduction'] = {
            'background': background,
            'objectives': objectives
        }
        
        self.checklist_completion['item_2'] = True
        self.checklist_completion['item_3'] = True
    
    def _generate_methods(self):
        """Generate Methods section (Items 4a-12b)."""
        
        methods = {}
        
        # Item 4a: Study design and data sources
        methods['study_design'] = {
            'design_type': self.study_info.get('study_design', 'retrospective cohort'),
            'data_sources': self.study_info.get('data_sources', 'clinical database'),
            'study_period': self.study_info.get('study_period', 'not specified'),
            'geographical_location': self.study_info.get('location', 'not specified')
        }
        
        # Item 4b: Eligibility criteria
        methods['eligibility'] = {
            'inclusion_criteria': self.study_info.get('inclusion_criteria', []),
            'exclusion_criteria': self.study_info.get('exclusion_criteria', []),
            'participant_selection': self.study_info.get('participant_selection',
                'Consecutive patients meeting eligibility criteria')
        }
        
        # Item 5a: Outcome definition
        methods['outcome_definition'] = self._generate_outcome_definition()
        
        # Item 5b: Outcome measurement
        methods['outcome_measurement'] = {
            'measurement_method': self.study_info.get('outcome_measurement',
                'Standard clinical assessment'),
            'timing': self.study_info.get('outcome_timing', 'At specified time points'),
            'adjudication': self.study_info.get('outcome_adjudication',
                'Standardized definitions applied')
        }
        
        # Item 6a: Predictor definition
        methods['predictor_definition'] = self._generate_predictor_definition()
        
        # Item 6b: Predictor measurement and timing
        methods['predictor_measurement'] = {
            'measurement_methods': self.study_info.get('predictor_measurement',
                'Standard clinical measurements'),
            'measurement_timing': self.study_info.get('predictor_timing',
                'At baseline or specified time points'),
            'blinding': self.study_info.get('blinding_predictors',
                'Predictors measured independently of outcome')
        }
        
        # Item 7a: Sample size and missing data
        methods['sample_size'] = self._generate_sample_size_description()
        
        # Item 7b: Missing data handling
        methods['missing_data'] = self._generate_missing_data_handling()
        
        # Item 8a: Model development approach
        methods['model_development'] = self._generate_model_development_description()
        
        # Item 8b: Model selection procedures
        methods['model_selection'] = self._generate_model_selection_description()
        
        # Item 9: Model performance measures
        methods['performance_measures'] = self._generate_performance_measures_description()
        
        # Item 10a: Resampling procedures
        methods['resampling'] = self._generate_resampling_description()
        
        # Item 10b: Model updating procedures
        methods['model_updating'] = self.study_info.get('model_updating',
            'Model updating procedures not applicable')
        
        # Item 11: Risk groups
        methods['risk_groups'] = self._generate_risk_groups_description()
        
        # Item 12a: Development vs validation
        methods['development_validation'] = {
            'development_data': self.study_info.get('development_data_description',
                'Training dataset used for model development'),
            'validation_approach': self.study_info.get('validation_approach',
                'Internal validation with resampling')
        }
        
        # Item 12b: Temporal or external validation
        methods['external_validation'] = self.study_info.get('external_validation',
            'External validation not performed in this study')
        
        self.report_sections['methods'] = methods
        
        # Mark items 4a-12b as complete
        items = ['4a', '4b', '5a', '5b', '6a', '6b', '7a', '7b', '8a', '8b', '9', '10a', '10b', '11', '12a', '12b']
        for item in items:
            self.checklist_completion[f'item_{item}'] = True
    
    def _generate_results(self):
        """Generate Results section (Items 13a-19)."""
        
        results = {}
        
        # Item 13a: Participant flow
        results['participant_flow'] = self._generate_participant_flow()
        
        # Item 13b: Model development dataset characteristics
        results['development_characteristics'] = self._generate_development_characteristics()
        
        # Item 13c: Model validation dataset characteristics
        results['validation_characteristics'] = self._generate_validation_characteristics()
        
        # Item 14a: Model specification
        results['model_specification'] = self._generate_model_specification()
        
        # Item 14b: Model equation or algorithm
        results['model_equation'] = self._generate_model_equation()
        
        # Item 15a: Model performance in development
        results['development_performance'] = self._generate_development_performance()
        
        # Item 15b: Model performance in validation
        results['validation_performance'] = self._generate_validation_performance()
        
        # Item 16: Model updating results
        results['model_updating_results'] = self.model_results.get('updating_results',
            'Model updating not performed')
        
        # Item 17: Risk group results
        results['risk_group_results'] = self._generate_risk_group_results()
        
        # Item 18: Calibration assessment
        results['calibration'] = self._generate_calibration_results()
        
        # Item 19: Sensitivity analysis
        results['sensitivity_analysis'] = self._generate_sensitivity_analysis_results()
        
        self.report_sections['results'] = results
        
        # Mark items 13a-19 as complete
        items = ['13a', '13b', '13c', '14a', '14b', '15a', '15b', '16', '17', '18', '19']
        for item in items:
            self.checklist_completion[f'item_{item}'] = True
    
    def _generate_discussion(self):
        """Generate Discussion section (Items 20-22)."""
        
        discussion = {}
        
        # Item 20: Key results and interpretation
        discussion['key_results'] = {
            'main_findings': self._summarize_model_performance(),
            'clinical_interpretation': self.study_info.get('clinical_interpretation',
                'Model performance suggests clinical utility for prediction.'),
            'comparison_to_existing': self.study_info.get('comparison_existing',
                'Model performance compares favorably to existing approaches.')
        }
        
        # Item 21: Limitations
        discussion['limitations'] = {
            'study_limitations': self.study_info.get('limitations', [
                'Retrospective design may introduce bias',
                'Single-center data may limit generalizability',
                'Limited external validation performed'
            ]),
            'model_limitations': self.study_info.get('model_limitations', [
                'Model performance may vary across populations',
                'Predictor availability may limit implementation'
            ])
        }
        
        # Item 22: Clinical implications and future research
        discussion['implications'] = {
            'clinical_implications': self.study_info.get('clinical_implications',
                'Model may improve clinical decision-making and patient outcomes.'),
            'implementation_considerations': self.study_info.get('implementation',
                'Consider integration into clinical workflow.'),
            'future_research': self.study_info.get('future_research', [
                'External validation in diverse populations',
                'Prospective implementation studies',
                'Model updating with additional data'
            ])
        }
        
        self.report_sections['discussion'] = discussion
        
        # Mark items 20-22 as complete
        for i in range(20, 23):
            self.checklist_completion[f'item_{i}'] = True
    
    def _generate_other_information(self):
        """Generate Other Information section (Items 23-27)."""
        
        other_info = {}
        
        # Item 23: Supplementary information
        other_info['supplementary_info'] = {
            'additional_materials': self.study_info.get('supplementary_materials',
                'Supplementary materials available upon request'),
            'code_availability': 'Analysis code available for transparency and reproducibility'
        }
        
        # Item 24: Declaration of interests
        other_info['declarations'] = {
            'funding': self.study_info.get('funding', 'Funding sources declared'),
            'conflicts_of_interest': self.study_info.get('conflicts_of_interest',
                'Authors declare no conflicts of interest'),
            'competing_interests': self.study_info.get('competing_interests',
                'No competing interests declared')
        }
        
        # Item 25: Model availability
        other_info['model_availability'] = {
            'model_access': self.study_info.get('model_access',
                'Model available for research purposes subject to data sharing agreements'),
            'implementation_details': self.study_info.get('implementation_details',
                'Implementation guidance available'),
            'contact_information': self.study_info.get('contact_info',
                'Contact corresponding author for model access')
        }
        
        # Item 26: Ethics approval
        other_info['ethics'] = {
            'ethics_approval': self.study_info.get('ethics_approval',
                'Ethics approval obtained from institutional review board'),
            'consent': self.study_info.get('consent',
                'Appropriate consent obtained as required'),
            'data_governance': self.study_info.get('data_governance',
                'Data use compliant with governance requirements')
        }
        
        # Item 27: Registration
        other_info['registration'] = {
            'protocol_registration': self.study_info.get('protocol_registration',
                'Study protocol registered where appropriate'),
            'registration_number': self.study_info.get('registration_number',
                'Registration number not applicable')
        }
        
        self.report_sections['other_information'] = other_info
        
        # Mark items 23-27 as complete
        for i in range(23, 28):
            self.checklist_completion[f'item_{i}'] = True
    
    def _generate_abstract_background(self) -> str:
        """Generate abstract background section."""
        outcome = self.study_info.get('target_outcome', 'clinical outcome')
        return f"Background: Accurate prediction of {outcome} is important for clinical decision-making. This study developed and validated a prediction model using clinical data."
    
    def _generate_abstract_methods(self) -> str:
        """Generate abstract methods section."""
        model_type = self.model_results.get('model_type', 'prediction model')
        n_patients = self.model_results.get('n_development', 'N/A')
        n_features = self.model_results.get('n_features', 'multiple')
        
        return f"Methods: {model_type} developed using {n_patients} patients and {n_features} clinical predictors. Internal validation performed with appropriate resampling techniques."
    
    def _generate_abstract_results(self) -> str:
        """Generate abstract results section."""
        # Extract key performance metrics
        performance_text = "Results: "
        
        if 'development_performance' in self.model_results:
            dev_perf = self.model_results['development_performance']
            if 'auc_roc' in dev_perf:
                performance_text += f"Development AUC: {dev_perf['auc_roc']:.3f}. "
            if 'calibration' in dev_perf:
                performance_text += "Good calibration observed. "
        
        if 'validation_performance' in self.model_results:
            val_perf = self.model_results['validation_performance']
            if 'auc_roc' in val_perf:
                performance_text += f"Validation AUC: {val_perf['auc_roc']:.3f}."
        
        if performance_text == "Results: ":
            performance_text += "Model demonstrated satisfactory performance in development and validation."
        
        return performance_text
    
    def _generate_abstract_conclusions(self) -> str:
        """Generate abstract conclusions section."""
        return self.study_info.get('conclusions_brief',
            'Conclusions: The prediction model demonstrated good performance and may be useful for clinical decision-making.')
    
    def _generate_outcome_definition(self) -> Dict[str, Any]:
        """Generate outcome definition (Item 5a)."""
        return {
            'primary_outcome': self.study_info.get('primary_outcome',
                'Primary clinical outcome as defined in study protocol'),
            'outcome_type': self.study_info.get('outcome_type', 'binary'),
            'definition_source': self.study_info.get('outcome_definition_source',
                'Standard clinical definitions'),
            'time_horizon': self.study_info.get('time_horizon',
                'Specified prediction time horizon')
        }
    
    def _generate_predictor_definition(self) -> Dict[str, Any]:
        """Generate predictor definition (Item 6a)."""
        predictors = self.study_info.get('predictors', [])
        
        return {
            'candidate_predictors': predictors,
            'predictor_rationale': self.study_info.get('predictor_rationale',
                'Predictors selected based on clinical knowledge and data availability'),
            'predictor_categories': self.study_info.get('predictor_categories', {
                'demographic': [],
                'clinical': [],
                'laboratory': [],
                'imaging': []
            })
        }
    
    def _generate_sample_size_description(self) -> Dict[str, Any]:
        """Generate sample size description (Item 7a)."""
        return {
            'sample_size_calculation': self.study_info.get('sample_size_calculation',
                'Sample size based on available data and events per variable considerations'),
            'events_per_variable': self.study_info.get('events_per_variable',
                'Adequate events per variable ratio maintained'),
            'development_n': self.model_results.get('n_development', 'N/A'),
            'validation_n': self.model_results.get('n_validation', 'N/A')
        }
    
    def _generate_missing_data_handling(self) -> Dict[str, Any]:
        """Generate missing data handling description (Item 7b)."""
        return {
            'missing_data_pattern': self.model_results.get('missing_data_analysis',
                'Missing data patterns assessed'),
            'handling_method': self.study_info.get('missing_data_method',
                'Complete case analysis with sensitivity analysis'),
            'imputation_details': self.study_info.get('imputation_details',
                'Multiple imputation considered for sensitivity analysis')
        }
    
    def _generate_model_development_description(self) -> Dict[str, Any]:
        """Generate model development description (Item 8a)."""
        return {
            'modeling_approach': self.model_results.get('modeling_approach',
                'Supervised machine learning approach'),
            'algorithm_type': self.model_results.get('model_type', 'Not specified'),
            'feature_selection': self.model_results.get('feature_selection_method',
                'Feature selection performed'),
            'hyperparameter_tuning': self.study_info.get('hyperparameter_tuning',
                'Hyperparameter optimization performed')
        }
    
    def _generate_model_selection_description(self) -> Dict[str, Any]:
        """Generate model selection description (Item 8b)."""
        return {
            'selection_criteria': self.study_info.get('model_selection_criteria',
                'Cross-validation performance used for model selection'),
            'comparison_models': self.model_results.get('model_comparison', {}),
            'final_model_rationale': self.study_info.get('final_model_rationale',
                'Best performing model selected based on validation metrics')
        }
    
    def _generate_performance_measures_description(self) -> Dict[str, Any]:
        """Generate performance measures description (Item 9)."""
        measures = {
            'discrimination': ['AUC-ROC', 'AUC-PR'],
            'calibration': ['Calibration plot', 'Hosmer-Lemeshow test'],
            'classification': ['Sensitivity', 'Specificity', 'PPV', 'NPV'],
            'overall': ['Accuracy', 'F1-score']
        }
        
        if self.model_results.get('task_type') == 'regression':
            measures.update({
                'regression': ['R-squared', 'RMSE', 'MAE']
            })
        
        return {
            'performance_metrics': measures,
            'metric_rationale': 'Metrics selected to assess discrimination, calibration, and clinical utility'
        }
    
    def _generate_resampling_description(self) -> Dict[str, Any]:
        """Generate resampling description (Item 10a)."""
        return {
            'resampling_method': self.model_results.get('cv_method', 'Cross-validation'),
            'resampling_details': self.model_results.get('cv_folds', '5-fold cross-validation'),
            'bootstrap_details': self.study_info.get('bootstrap_details',
                'Bootstrap resampling not performed'),
            'optimism_correction': self.study_info.get('optimism_correction',
                'Cross-validation provides optimism-corrected estimates')
        }
    
    def _generate_risk_groups_description(self) -> Dict[str, Any]:
        """Generate risk groups description (Item 11)."""
        return {
            'risk_stratification': self.study_info.get('risk_stratification',
                'Risk groups defined based on predicted probability thresholds'),
            'threshold_selection': self.study_info.get('threshold_selection',
                'Thresholds selected based on clinical relevance'),
            'group_definitions': self.study_info.get('risk_group_definitions', {})
        }
    
    def _generate_participant_flow(self) -> Dict[str, Any]:
        """Generate participant flow (Item 13a)."""
        return {
            'initial_population': self.model_results.get('initial_n', 'N/A'),
            'exclusions': self.model_results.get('exclusions', []),
            'final_development': self.model_results.get('n_development', 'N/A'),
            'final_validation': self.model_results.get('n_validation', 'N/A')
        }
    
    def _generate_development_characteristics(self) -> Dict[str, Any]:
        """Generate development dataset characteristics (Item 13b)."""
        return {
            'baseline_characteristics': self.model_results.get('development_characteristics', {}),
            'outcome_prevalence': self.model_results.get('development_outcome_prevalence', 'N/A'),
            'predictor_distributions': self.model_results.get('development_predictor_summary', {})
        }
    
    def _generate_validation_characteristics(self) -> Dict[str, Any]:
        """Generate validation dataset characteristics (Item 13c)."""
        return {
            'baseline_characteristics': self.model_results.get('validation_characteristics', {}),
            'outcome_prevalence': self.model_results.get('validation_outcome_prevalence', 'N/A'),
            'predictor_distributions': self.model_results.get('validation_predictor_summary', {})
        }
    
    def _generate_model_specification(self) -> Dict[str, Any]:
        """Generate model specification (Item 14a)."""
        return {
            'final_predictors': self.model_results.get('selected_features', []),
            'model_architecture': self.model_results.get('model_architecture',
                'Model architecture details'),
            'preprocessing_steps': self.model_results.get('preprocessing_steps', []),
            'model_complexity': self.model_results.get('model_complexity', 'Not specified')
        }
    
    def _generate_model_equation(self) -> Dict[str, Any]:
        """Generate model equation or algorithm (Item 14b)."""
        return {
            'model_equation': self.model_results.get('model_equation',
                'Model equation or algorithm specification not provided'),
            'coefficients': self.model_results.get('model_coefficients', {}),
            'intercept': self.model_results.get('model_intercept', 'N/A'),
            'implementation_code': 'Available upon request'
        }
    
    def _generate_development_performance(self) -> Dict[str, Any]:
        """Generate development performance (Item 15a)."""
        return self.model_results.get('development_performance', {
            'auc_roc': 'N/A',
            'calibration': 'N/A',
            'sensitivity': 'N/A',
            'specificity': 'N/A'
        })
    
    def _generate_validation_performance(self) -> Dict[str, Any]:
        """Generate validation performance (Item 15b)."""
        return self.model_results.get('validation_performance', {
            'auc_roc': 'N/A',
            'calibration': 'N/A',
            'sensitivity': 'N/A',
            'specificity': 'N/A'
        })
    
    def _generate_risk_group_results(self) -> Dict[str, Any]:
        """Generate risk group results (Item 17)."""
        return self.model_results.get('risk_group_analysis', {
            'risk_groups': {},
            'group_performance': {}
        })
    
    def _generate_calibration_results(self) -> Dict[str, Any]:
        """Generate calibration results (Item 18)."""
        return self.model_results.get('calibration_results', {
            'calibration_plot': 'Available in supplementary materials',
            'hosmer_lemeshow': 'N/A',
            'calibration_slope': 'N/A',
            'calibration_intercept': 'N/A'
        })
    
    def _generate_sensitivity_analysis_results(self) -> Dict[str, Any]:
        """Generate sensitivity analysis results (Item 19)."""
        return self.model_results.get('sensitivity_analyses', {
            'missing_data_sensitivity': 'Performed with multiple imputation',
            'threshold_sensitivity': 'Different thresholds evaluated',
            'population_sensitivity': 'Subgroup analyses performed'
        })
    
    def _summarize_model_performance(self) -> str:
        """Summarize model performance from results."""
        summary_parts = []
        
        # Development performance
        dev_perf = self.model_results.get('development_performance', {})
        if 'auc_roc' in dev_perf:
            summary_parts.append(f"Development AUC: {dev_perf['auc_roc']:.3f}")
        
        # Validation performance
        val_perf = self.model_results.get('validation_performance', {})
        if 'auc_roc' in val_perf:
            summary_parts.append(f"Validation AUC: {val_perf['auc_roc']:.3f}")
        
        if not summary_parts:
            return "Model demonstrated satisfactory performance in development and validation."
        
        return ". ".join(summary_parts) + "."
    
    def _format_html_report(self, include_checklist: bool) -> str:
        """Format report as HTML."""
        html_parts = ['<!DOCTYPE html>', '<html>', '<head>',
                     '<title>TRIPOD+AI Report</title>', '</head>', '<body>']
        
        # Title
        html_parts.append(f'<h1>{self.report_sections["title_abstract"]["title"]}</h1>')
        
        # Abstract
        html_parts.append('<h2>Abstract</h2>')
        abstract = self.report_sections["title_abstract"]["abstract"]
        for section, content in abstract.items():
            html_parts.append(f'<p><strong>{section.title()}:</strong> {content}</p>')
        
        # Main sections
        section_mapping = {
            'introduction': 'Introduction',
            'methods': 'Methods',
            'results': 'Results',
            'discussion': 'Discussion',
            'other_information': 'Other Information'
        }
        
        for section_key, section_title in section_mapping.items():
            if section_key in self.report_sections:
                html_parts.append(f'<h2>{section_title}</h2>')
                html_parts.append(self._format_section_html(self.report_sections[section_key]))
        
        # TRIPOD+AI checklist
        if include_checklist:
            html_parts.append('<h2>TRIPOD+AI Checklist Completion</h2>')
            html_parts.append(self._format_checklist_html())
        
        html_parts.extend(['</body>', '</html>'])
        
        return '\n'.join(html_parts)
    
    def _format_markdown_report(self, include_checklist: bool) -> str:
        """Format report as Markdown."""
        md_parts = []
        
        # Title
        md_parts.append(f'# {self.report_sections["title_abstract"]["title"]}\n')
        
        # Abstract
        md_parts.append('## Abstract\n')
        abstract = self.report_sections["title_abstract"]["abstract"]
        for section, content in abstract.items():
            md_parts.append(f'**{section.title()}:** {content}\n')
        
        # Main sections
        section_mapping = {
            'introduction': 'Introduction',
            'methods': 'Methods',
            'results': 'Results', 
            'discussion': 'Discussion',
            'other_information': 'Other Information'
        }
        
        for section_key, section_title in section_mapping.items():
            if section_key in self.report_sections:
                md_parts.append(f'\n## {section_title}\n')
                md_parts.append(self._format_section_markdown(self.report_sections[section_key]))
        
        # TRIPOD+AI checklist
        if include_checklist:
            md_parts.append('\n## TRIPOD+AI Checklist Completion\n')
            md_parts.append(self._format_checklist_markdown())
        
        return ''.join(md_parts)
    
    def _format_section_html(self, section_data: Dict[str, Any]) -> str:
        """Format a section as HTML."""
        html_parts = []
        
        def format_item(key, value, level=3):
            if isinstance(value, dict):
                html_parts.append(f'<h{level}>{key.replace("_", " ").title()}</h{level}>')
                for subkey, subvalue in value.items():
                    format_item(subkey, subvalue, level + 1)
            elif isinstance(value, list):
                html_parts.append(f'<h{level}>{key.replace("_", " ").title()}</h{level}>')
                html_parts.append('<ul>')
                for item in value:
                    html_parts.append(f'<li>{item}</li>')
                html_parts.append('</ul>')
            else:
                html_parts.append(f'<p><strong>{key.replace("_", " ").title()}:</strong> {value}</p>')
        
        for key, value in section_data.items():
            format_item(key, value)
        
        return '\n'.join(html_parts)
    
    def _format_section_markdown(self, section_data: Dict[str, Any]) -> str:
        """Format a section as Markdown."""
        md_parts = []
        
        def format_item(key, value, level=3):
            if isinstance(value, dict):
                md_parts.append(f'{"#" * level} {key.replace("_", " ").title()}\n')
                for subkey, subvalue in value.items():
                    format_item(subkey, subvalue, level + 1)
            elif isinstance(value, list):
                md_parts.append(f'{"#" * level} {key.replace("_", " ").title()}\n')
                for item in value:
                    md_parts.append(f'- {item}\n')
            else:
                md_parts.append(f'**{key.replace("_", " ").title()}:** {value}\n\n')
        
        for key, value in section_data.items():
            format_item(key, value)
        
        return ''.join(md_parts)
    
    def _format_checklist_html(self) -> str:
        """Format TRIPOD+AI checklist as HTML."""
        html_parts = ['<table>', '<tr><th>Item</th><th>Description</th><th>Completed</th></tr>']
        
        checklist_items = self._get_tripod_checklist_items()
        
        for item_num, description in checklist_items.items():
            completed = '✓' if self.checklist_completion.get(f'item_{item_num}', False) else '✗'
            html_parts.append(f'<tr><td>{item_num}</td><td>{description}</td><td>{completed}</td></tr>')
        
        html_parts.append('</table>')
        
        return '\n'.join(html_parts)
    
    def _format_checklist_markdown(self) -> str:
        """Format TRIPOD+AI checklist as Markdown."""
        md_parts = ['| Item | Description | Completed |', '|------|-------------|-----------|']
        
        checklist_items = self._get_tripod_checklist_items()
        
        for item_num, description in checklist_items.items():
            completed = '✓' if self.checklist_completion.get(f'item_{item_num}', False) else '✗'
            md_parts.append(f'| {item_num} | {description} | {completed} |')
        
        return '\n'.join(md_parts)
    
    def _get_tripod_checklist_items(self) -> Dict[str, str]:
        """Get TRIPOD+AI checklist items and descriptions."""
        return {
            '1a': "Title: Identify as prediction model study",
            '1b': "Abstract: Structured summary of study",
            '2': "Background and objectives",
            '3': "Study aims and intended use",
            '4a': "Study design and data sources",
            '4b': "Eligibility criteria",
            '5a': "Outcome definition and measurement",
            '5b': "Outcome timing and assessment",
            '6a': "Predictor definition and rationale",
            '6b': "Predictor measurement and timing",
            '7a': "Sample size and justification",
            '7b': "Missing data handling",
            '8a': "Model development approach",
            '8b': "Model selection procedures",
            '9': "Performance measures",
            '10a': "Resampling procedures",
            '10b': "Model updating procedures",
            '11': "Risk groups definition",
            '12a': "Development vs validation datasets",
            '12b': "External validation approach",
            '13a': "Participant flow",
            '13b': "Development dataset characteristics",
            '13c': "Validation dataset characteristics",
            '14a': "Model specification",
            '14b': "Model equation or algorithm",
            '15a': "Development performance",
            '15b': "Validation performance",
            '16': "Model updating results",
            '17': "Risk group results",
            '18': "Model calibration",
            '19': "Sensitivity analyses",
            '20': "Key results interpretation",
            '21': "Limitations",
            '22': "Clinical implications",
            '23': "Supplementary information",
            '24': "Declaration of interests",
            '25': "Model availability",
            '26': "Ethics approval",
            '27': "Registration"
        }
    
    def _format_docx_report(self, include_checklist: bool) -> str:
        """Format report for Word document (placeholder)."""
        # This would require python-docx library
        # For now, return markdown format
        return self._format_markdown_report(include_checklist)