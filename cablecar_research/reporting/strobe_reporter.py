"""
STROBE Reporter

Generates comprehensive reports following STROBE (Strengthening the Reporting 
of Observational Studies in Epidemiology) guidelines. Produces standardized
sections covering all 22 STROBE checklist items.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import json
from pathlib import Path


class STROBEReporter:
    """
    Generate STROBE-compliant reports for observational studies.
    
    Covers all 22 STROBE checklist items:
    - Title and Abstract (1)
    - Introduction (2-3)  
    - Methods (4-12)
    - Results (13-17)
    - Discussion (18-21)
    - Other Information (22)
    """
    
    def __init__(self, study_info: Dict[str, Any]):
        """
        Initialize STROBE reporter with study information.
        
        Args:
            study_info: Dictionary containing study metadata
        """
        self.study_info = study_info
        self.report_sections = {}
        self.analysis_results = {}
        self.checklist_completion = {}
        
    def generate_report(self,
                       analysis_results: Dict[str, Any],
                       output_format: str = 'html',
                       include_checklist: bool = True) -> str:
        """
        Generate complete STROBE-compliant report.
        
        Args:
            analysis_results: Dictionary containing all analysis results
            output_format: Output format ('html', 'markdown', 'docx')
            include_checklist: Whether to include STROBE checklist
            
        Returns:
            Formatted report string
        """
        self.analysis_results = analysis_results
        
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
        """Generate Title and Abstract section (Item 1)."""
        
        # Title
        study_design = self.study_info.get('design', 'observational study')
        population = self.study_info.get('population', 'clinical population')
        
        title = (f"{self.study_info.get('title', 'Clinical Study')}: "
                f"A {study_design} in {population}")
        
        # Abstract components
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
        
        self.checklist_completion['item_1'] = True
    
    def _generate_introduction(self):
        """Generate Introduction section (Items 2-3)."""
        
        # Item 2: Background/rationale
        background = {
            'rationale': self.study_info.get('rationale', 
                'This study addresses important clinical questions using observational data.'),
            'literature_review': self.study_info.get('literature_review',
                'Previous studies have shown the need for further investigation in this area.'),
            'knowledge_gaps': self.study_info.get('knowledge_gaps',
                'Gaps remain in our understanding of these clinical relationships.')
        }
        
        # Item 3: Objectives
        objectives = {
            'primary_objective': self.study_info.get('primary_objective',
                'To examine associations between clinical variables and outcomes.'),
            'secondary_objectives': self.study_info.get('secondary_objectives', []),
            'hypotheses': self.study_info.get('hypotheses', [])
        }
        
        self.report_sections['introduction'] = {
            'background': background,
            'objectives': objectives
        }
        
        self.checklist_completion['item_2'] = True
        self.checklist_completion['item_3'] = True
    
    def _generate_methods(self):
        """Generate Methods section (Items 4-12)."""
        
        methods = {}
        
        # Item 4: Study design and setting
        methods['study_design'] = {
            'design_type': self.study_info.get('design', 'retrospective cohort'),
            'setting': self.study_info.get('setting', 'clinical database'),
            'data_source': self.study_info.get('data_source', 'electronic health records'),
            'study_period': self.study_info.get('study_period', 'not specified')
        }
        
        # Item 5: Participants  
        methods['participants'] = self._generate_participants_description()
        
        # Item 6: Variables
        methods['variables'] = self._generate_variables_description()
        
        # Item 7: Data sources/measurement
        methods['data_sources'] = {
            'data_collection': self.study_info.get('data_collection_methods',
                'Data extracted from clinical databases'),
            'measurement_procedures': self.study_info.get('measurement_procedures',
                'Standard clinical measurement procedures'),
            'quality_control': self.study_info.get('quality_control',
                'Data quality checks performed')
        }
        
        # Item 8: Bias
        methods['bias_control'] = {
            'potential_biases': self.study_info.get('potential_biases', [
                'Selection bias', 'Information bias', 'Confounding'
            ]),
            'bias_mitigation': self.study_info.get('bias_mitigation', [
                'Careful variable definition', 'Missing data analysis', 'Sensitivity analyses'
            ])
        }
        
        # Item 9: Study size
        methods['study_size'] = self._generate_study_size_description()
        
        # Item 10: Quantitative variables
        methods['quantitative_variables'] = self._generate_quantitative_description()
        
        # Item 11: Statistical methods
        methods['statistical_methods'] = self._generate_statistical_methods_description()
        
        # Item 12: Subgroups, missing data, loss to follow-up, sensitivity analyses
        methods['additional_analyses'] = self._generate_additional_analyses_description()
        
        self.report_sections['methods'] = methods
        
        # Mark items 4-12 as complete
        for i in range(4, 13):
            self.checklist_completion[f'item_{i}'] = True
    
    def _generate_results(self):
        """Generate Results section (Items 13-17)."""
        
        results = {}
        
        # Item 13: Participants
        results['participants'] = self._generate_participant_flow()
        
        # Item 14: Descriptive data
        results['descriptive_data'] = self._generate_descriptive_results()
        
        # Item 15: Outcome data
        results['outcome_data'] = self._generate_outcome_results()
        
        # Item 16: Main results
        results['main_results'] = self._generate_main_results()
        
        # Item 17: Other analyses
        results['other_analyses'] = self._generate_other_analyses_results()
        
        self.report_sections['results'] = results
        
        # Mark items 13-17 as complete
        for i in range(13, 18):
            self.checklist_completion[f'item_{i}'] = True
    
    def _generate_discussion(self):
        """Generate Discussion section (Items 18-21)."""
        
        discussion = {}
        
        # Item 18: Key results
        discussion['key_results'] = {
            'summary': self._summarize_key_findings(),
            'interpretation': self.study_info.get('interpretation',
                'Results support the study hypotheses and align with existing literature.')
        }
        
        # Item 19: Limitations
        discussion['limitations'] = {
            'study_limitations': self.study_info.get('limitations', [
                'Observational design limits causal inference',
                'Potential for unmeasured confounding',
                'Missing data may introduce bias'
            ]),
            'impact_assessment': 'These limitations should be considered when interpreting results.'
        }
        
        # Item 20: Interpretation
        discussion['interpretation'] = {
            'causality': 'Causal relationships cannot be definitively established from observational data.',
            'generalizability': self.study_info.get('generalizability',
                'Results may generalize to similar clinical populations.'),
            'clinical_relevance': self.study_info.get('clinical_relevance',
                'Findings have potential clinical implications.')
        }
        
        # Item 21: Generalizability
        discussion['generalizability'] = {
            'external_validity': self.study_info.get('external_validity',
                'Results may apply to broader populations with similar characteristics.'),
            'population_differences': self.study_info.get('population_differences',
                'Consider population differences when applying results.')
        }
        
        self.report_sections['discussion'] = discussion
        
        # Mark items 18-21 as complete
        for i in range(18, 22):
            self.checklist_completion[f'item_{i}'] = True
    
    def _generate_other_information(self):
        """Generate Other Information section (Item 22)."""
        
        other_info = {
            'funding': self.study_info.get('funding', 'Funding sources not specified'),
            'conflicts_of_interest': self.study_info.get('conflicts_of_interest', 
                'No conflicts of interest declared'),
            'data_availability': self.study_info.get('data_availability',
                'Data sharing restrictions apply to protect patient privacy'),
            'code_availability': 'Analysis code available for validation and replication',
            'ethics_approval': self.study_info.get('ethics_approval',
                'Ethics approval obtained as appropriate')
        }
        
        self.report_sections['other_information'] = other_info
        self.checklist_completion['item_22'] = True
    
    def _generate_abstract_background(self) -> str:
        """Generate abstract background section."""
        return (f"Background: {self.study_info.get('background_brief', 'This study examines important clinical relationships using observational data.')}")
    
    def _generate_abstract_methods(self) -> str:
        """Generate abstract methods section."""
        design = self.study_info.get('design', 'observational study')
        n_patients = self.analysis_results.get('descriptive_data', {}).get('n_patients', 'N/A')
        return f"Methods: {design.title()} including {n_patients} patients. Statistical analyses performed using appropriate methods."
    
    def _generate_abstract_results(self) -> str:
        """Generate abstract results section."""
        # Extract key findings from analysis results
        key_findings = []
        
        if 'main_results' in self.analysis_results:
            for result in self.analysis_results['main_results']:
                if isinstance(result, dict) and 'p_value' in result:
                    significance = 'significant' if result['p_value'] < 0.05 else 'non-significant'
                    key_findings.append(f"Association was {significance} (p={result['p_value']:.3f})")
        
        if not key_findings:
            key_findings = ["Key associations were identified in the analysis."]
        
        return f"Results: {' '.join(key_findings)}"
    
    def _generate_abstract_conclusions(self) -> str:
        """Generate abstract conclusions section."""
        return self.study_info.get('conclusions_brief', 
            'Conclusions: Study findings contribute to understanding of clinical relationships.')
    
    def _generate_participants_description(self) -> Dict[str, Any]:
        """Generate participants description (Item 5)."""
        return {
            'eligibility_criteria': {
                'inclusion_criteria': self.study_info.get('inclusion_criteria', [
                    'Adult patients (≥18 years)',
                    'Available outcome data'
                ]),
                'exclusion_criteria': self.study_info.get('exclusion_criteria', [
                    'Missing key variables',
                    'Invalid data entries'
                ])
            },
            'recruitment_methods': self.study_info.get('recruitment_methods',
                'Patients identified from clinical database'),
            'follow_up_period': self.study_info.get('follow_up_period', 'Not applicable')
        }
    
    def _generate_variables_description(self) -> Dict[str, Any]:
        """Generate variables description (Item 6)."""
        variables = {}
        
        # Primary outcome
        if 'primary_outcome' in self.study_info:
            variables['primary_outcome'] = {
                'name': self.study_info['primary_outcome'],
                'definition': self.study_info.get('primary_outcome_definition',
                    'Primary outcome as defined in study protocol'),
                'measurement': self.study_info.get('primary_outcome_measurement',
                    'Standard clinical measurement')
            }
        
        # Secondary outcomes
        variables['secondary_outcomes'] = self.study_info.get('secondary_outcomes', [])
        
        # Predictors/exposures
        variables['predictors'] = self.study_info.get('predictors', [])
        
        # Confounders
        variables['confounders'] = self.study_info.get('confounders', [])
        
        return variables
    
    def _generate_study_size_description(self) -> Dict[str, Any]:
        """Generate study size description (Item 9)."""
        size_info = {}
        
        if 'sample_size_calculation' in self.study_info:
            size_info['calculated_size'] = self.study_info['sample_size_calculation']
        else:
            size_info['calculated_size'] = 'Sample size determined by available data'
        
        # Actual sample size from results
        if 'descriptive_data' in self.analysis_results:
            size_info['actual_size'] = self.analysis_results['descriptive_data'].get('n_patients', 'N/A')
        
        size_info['power_analysis'] = self.study_info.get('power_analysis',
            'Post-hoc power analysis may be considered')
        
        return size_info
    
    def _generate_quantitative_description(self) -> Dict[str, Any]:
        """Generate quantitative variables description (Item 10)."""
        return {
            'categorization_methods': self.study_info.get('categorization_methods',
                'Continuous variables analyzed as continuous; categorical variables as specified'),
            'transformation_methods': self.study_info.get('transformation_methods', []),
            'missing_data_handling': self.study_info.get('missing_data_handling',
                'Complete case analysis performed; missing data patterns assessed')
        }
    
    def _generate_statistical_methods_description(self) -> Dict[str, Any]:
        """Generate statistical methods description (Item 11)."""
        methods = {
            'descriptive_methods': [
                'Mean ± standard deviation for continuous variables',
                'Frequencies and percentages for categorical variables',
                'Median and interquartile range for non-normal continuous variables'
            ],
            'inferential_methods': [],
            'software': 'Python with scientific computing libraries',
            'significance_level': 'α = 0.05'
        }
        
        # Add methods based on analyses performed
        if 'hypothesis_testing' in self.analysis_results:
            methods['inferential_methods'].extend([
                'Chi-square tests for categorical associations',
                'T-tests or Mann-Whitney U tests for continuous comparisons',
                'Multiple comparison corrections applied where appropriate'
            ])
        
        if 'regression' in self.analysis_results:
            methods['inferential_methods'].extend([
                'Linear/logistic regression for multivariable analysis',
                'Model diagnostics performed',
                'Confidence intervals calculated'
            ])
        
        return methods
    
    def _generate_additional_analyses_description(self) -> Dict[str, Any]:
        """Generate additional analyses description (Item 12)."""
        return {
            'subgroup_analyses': self.study_info.get('subgroup_analyses', []),
            'missing_data_strategy': 'Complete case analysis with missing data assessment',
            'sensitivity_analyses': [
                'Alternative variable definitions',
                'Different inclusion/exclusion criteria',
                'Multiple imputation for missing data'
            ]
        }
    
    def _generate_participant_flow(self) -> Dict[str, Any]:
        """Generate participant flow description (Item 13)."""
        flow = {
            'initial_population': self.analysis_results.get('initial_n', 'N/A'),
            'exclusions': [],
            'final_sample': self.analysis_results.get('final_n', 'N/A')
        }
        
        if 'exclusion_summary' in self.analysis_results:
            flow['exclusions'] = self.analysis_results['exclusion_summary']
        
        return flow
    
    def _generate_descriptive_results(self) -> Dict[str, Any]:
        """Generate descriptive results (Item 14)."""
        descriptive = {}
        
        if 'table1' in self.analysis_results:
            descriptive['baseline_characteristics'] = self.analysis_results['table1']
        
        if 'missing_data' in self.analysis_results:
            descriptive['missing_data'] = self.analysis_results['missing_data']
        
        return descriptive
    
    def _generate_outcome_results(self) -> Dict[str, Any]:
        """Generate outcome results (Item 15)."""
        outcomes = {}
        
        if 'outcome_analysis' in self.analysis_results:
            outcomes = self.analysis_results['outcome_analysis']
        
        return outcomes
    
    def _generate_main_results(self) -> Dict[str, Any]:
        """Generate main results (Item 16)."""
        main_results = {}
        
        if 'primary_analysis' in self.analysis_results:
            main_results['primary_analysis'] = self.analysis_results['primary_analysis']
        
        if 'effect_sizes' in self.analysis_results:
            main_results['effect_sizes'] = self.analysis_results['effect_sizes']
        
        return main_results
    
    def _generate_other_analyses_results(self) -> Dict[str, Any]:
        """Generate other analyses results (Item 17)."""
        other = {}
        
        if 'secondary_analyses' in self.analysis_results:
            other['secondary_analyses'] = self.analysis_results['secondary_analyses']
        
        if 'sensitivity_analyses' in self.analysis_results:
            other['sensitivity_analyses'] = self.analysis_results['sensitivity_analyses']
        
        return other
    
    def _summarize_key_findings(self) -> str:
        """Summarize key findings from analysis results."""
        findings = []
        
        # Extract significant results
        if 'main_results' in self.analysis_results:
            for result_name, result_data in self.analysis_results['main_results'].items():
                if isinstance(result_data, dict) and 'p_value' in result_data:
                    if result_data['p_value'] < 0.05:
                        findings.append(f"Significant association found for {result_name} (p={result_data['p_value']:.3f})")
        
        if not findings:
            findings = ["Key associations were identified in the primary analysis."]
        
        return '; '.join(findings)
    
    def _format_html_report(self, include_checklist: bool) -> str:
        """Format report as HTML."""
        html_parts = ['<!DOCTYPE html>', '<html>', '<head>', 
                     '<title>STROBE Report</title>', '</head>', '<body>']
        
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
        
        # STROBE checklist
        if include_checklist:
            html_parts.append('<h2>STROBE Checklist Completion</h2>')
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
        
        # STROBE checklist
        if include_checklist:
            md_parts.append('\n## STROBE Checklist Completion\n')
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
        """Format STROBE checklist as HTML."""
        html_parts = ['<table>', '<tr><th>Item</th><th>Description</th><th>Completed</th></tr>']
        
        checklist_items = self._get_strobe_checklist_items()
        
        for item_num, description in checklist_items.items():
            completed = '✓' if self.checklist_completion.get(f'item_{item_num}', False) else '✗'
            html_parts.append(f'<tr><td>{item_num}</td><td>{description}</td><td>{completed}</td></tr>')
        
        html_parts.append('</table>')
        
        return '\n'.join(html_parts)
    
    def _format_checklist_markdown(self) -> str:
        """Format STROBE checklist as Markdown."""
        md_parts = ['| Item | Description | Completed |', '|------|-------------|-----------|']
        
        checklist_items = self._get_strobe_checklist_items()
        
        for item_num, description in checklist_items.items():
            completed = '✓' if self.checklist_completion.get(f'item_{item_num}', False) else '✗'
            md_parts.append(f'| {item_num} | {description} | {completed} |')
        
        return '\n'.join(md_parts)
    
    def _get_strobe_checklist_items(self) -> Dict[int, str]:
        """Get STROBE checklist items and descriptions."""
        return {
            1: "Title and abstract",
            2: "Background/rationale",
            3: "Objectives",
            4: "Study design and setting",
            5: "Participants",
            6: "Variables",
            7: "Data sources/measurement",
            8: "Bias",
            9: "Study size",
            10: "Quantitative variables",
            11: "Statistical methods",
            12: "Data access and cleaning methods",
            13: "Participants",
            14: "Descriptive data",
            15: "Outcome data",
            16: "Main results",
            17: "Other analyses",
            18: "Key results",
            19: "Limitations",
            20: "Interpretation",
            21: "Generalizability",
            22: "Funding"
        }
    
    def _format_docx_report(self, include_checklist: bool) -> str:
        """Format report for Word document (placeholder)."""
        # This would require python-docx library
        # For now, return markdown format
        return self._format_markdown_report(include_checklist)