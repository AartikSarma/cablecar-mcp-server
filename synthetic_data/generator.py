"""
Enhanced Synthetic CLIF Dataset Generator

Generates comprehensive synthetic clinical datasets with:
- Realistic clinical relationships and patterns
- Multiple outcome types for diverse analyses
- Complex temporal patterns
- Missing data patterns that mirror real clinical data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Set random seeds for reproducibility
fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)


class CLIFSyntheticGenerator:
    """
    Enhanced synthetic CLIF dataset generator for clinical research testing.
    
    Features:
    - Realistic clinical patterns and relationships
    - Multiple outcome types (mortality, LOS, readmissions)
    - Complex temporal data (vitals, labs, medications)
    - Missing data patterns
    - Configurable complexity and sample size
    """
    
    def __init__(self, 
                 n_patients: int = 1000,
                 outcome_prevalence: Dict[str, float] = None,
                 missing_data_rate: float = 0.15,
                 output_dir: str = "./data/synthetic"):
        """
        Initialize synthetic data generator.
        
        Args:
            n_patients: Number of patients to generate
            outcome_prevalence: Prevalence rates for different outcomes
            missing_data_rate: Overall missing data rate (0.0 to 1.0)
            output_dir: Output directory for generated files
        """
        self.n_patients = n_patients
        self.missing_data_rate = missing_data_rate
        self.output_dir = Path(output_dir)
        
        # Default outcome prevalences
        if outcome_prevalence is None:
            self.outcome_prevalence = {
                'mortality': 0.12,
                'icu_mortality': 0.08,
                'readmission_30d': 0.15,
                'aki': 0.25,
                'sepsis': 0.18
            }
        else:
            self.outcome_prevalence = outcome_prevalence
        
        # Clinical parameters
        self.age_ranges = {
            'young_adult': (18, 35),
            'adult': (36, 55), 
            'middle_aged': (56, 70),
            'elderly': (71, 90)
        }
        
        self.comorbidities = [
            'diabetes', 'hypertension', 'coronary_artery_disease',
            'chronic_kidney_disease', 'copd', 'heart_failure',
            'stroke', 'cancer', 'liver_disease'
        ]
        
        # Initialize containers for generated data
        self.patient_data = {}
        self.hospitalization_data = []
        self.adt_data = []
        self.vitals_data = []
        self.labs_data = []
        self.respiratory_support_data = []
        self.medication_admin_data = []
        self.patient_assessments_data = []
        
    def generate_complete_dataset(self) -> Dict[str, int]:
        """
        Generate complete synthetic CLIF dataset.
        
        Returns:
            Dictionary with record counts for each generated table
        """
        print("Generating comprehensive synthetic CLIF dataset...")
        print(f"Parameters: {self.n_patients} patients, {self.missing_data_rate:.1%} missing data rate")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate core data with relationships
        self._generate_patients_with_risk_factors()
        self._generate_hospitalizations_with_outcomes()
        self._generate_adt_with_patterns()
        
        # Generate time-series clinical data
        self._generate_vitals_with_trends()
        self._generate_labs_with_patterns()
        self._generate_respiratory_support()
        self._generate_medications_with_protocols()
        self._generate_assessments()
        
        # Apply realistic missing data patterns
        self._apply_missing_data_patterns()
        
        # Save all tables
        record_counts = self._save_all_tables()
        
        # Generate data summary
        self._generate_data_summary()
        
        print(f"✓ Synthetic dataset generated in {self.output_dir}")
        return record_counts
    
    def _generate_patients_with_risk_factors(self):
        """Generate patients with realistic demographic and risk factor patterns."""
        races = ['White', 'Black or African American', 'Asian', 'Hispanic or Latino', 'Other', 'Unknown']
        ethnicities = ['Not Hispanic or Latino', 'Hispanic or Latino', 'Unknown']
        sexes = ['Male', 'Female']
        
        # Age-based risk stratification
        age_distribution = [0.15, 0.25, 0.35, 0.25]  # young_adult, adult, middle_aged, elderly
        age_categories = list(self.age_ranges.keys())
        
        for i in range(self.n_patients):
            patient_id = f"PT{i+1:06d}"
            
            # Select age category and generate age
            age_category = np.random.choice(age_categories, p=age_distribution)
            age_min, age_max = self.age_ranges[age_category]
            age = np.random.randint(age_min, age_max + 1)
            
            # Birth date calculation
            days_old = age * 365 + np.random.randint(-180, 180)  # Add some variation
            birth_date = datetime.now() - timedelta(days=days_old)
            
            # Demographics with some correlation patterns
            sex = np.random.choice(sexes, p=[0.52, 0.48])
            race = np.random.choice(races, p=[0.6, 0.15, 0.1, 0.08, 0.05, 0.02])
            ethnicity = np.random.choice(ethnicities, p=[0.75, 0.22, 0.03])
            
            # Risk factors correlated with age
            risk_multiplier = 1 + (age - 18) * 0.02  # Increased risk with age
            
            # Generate comorbidities with age correlation
            comorbidity_probs = self._get_age_adjusted_comorbidity_probs(age)
            patient_comorbidities = []
            
            for comorbidity, base_prob in comorbidity_probs.items():
                if np.random.random() < base_prob:
                    patient_comorbidities.append(comorbidity)
            
            self.patient_data[patient_id] = {
                'patient_id': patient_id,
                'date_of_birth': birth_date.date(),
                'age': age,
                'race': race,
                'ethnicity': ethnicity,
                'sex': sex,
                'comorbidities': patient_comorbidities,
                'risk_score': self._calculate_patient_risk_score(age, patient_comorbidities)
            }
    
    def _generate_hospitalizations_with_outcomes(self):
        """Generate hospitalizations with realistic outcome patterns."""
        admission_sources = ['Emergency Department', 'Direct Admission', 
                           'Transfer from Another Hospital', 'Operating Room']
        discharge_dispositions = ['Home', 'Skilled Nursing Facility', 
                                'Expired', 'Hospice', 'Transfer to Another Hospital']
        
        hosp_id = 1
        for patient_id, patient_info in self.patient_data.items():
            # Number of hospitalizations (most patients have 1, some have more)
            n_hosps = np.random.choice([1, 2, 3], p=[0.75, 0.20, 0.05])
            
            for hosp_num in range(n_hosps):
                # Admission date (within last 2 years)
                days_ago = np.random.randint(30, 730)
                admission_date = datetime.now() - timedelta(days=days_ago)
                
                # Determine outcomes based on patient risk factors
                risk_score = patient_info['risk_score']
                age = patient_info['age']
                
                # Mortality risk increases with age and comorbidities
                mortality_risk = self.outcome_prevalence['mortality'] * risk_score
                is_mortality = np.random.random() < mortality_risk
                
                # AKI risk
                aki_risk = self.outcome_prevalence['aki'] * (1 + 0.3 * len(patient_info['comorbidities']) / 5)
                has_aki = np.random.random() < aki_risk
                
                # Sepsis risk
                sepsis_risk = self.outcome_prevalence['sepsis'] * risk_score
                has_sepsis = np.random.random() < sepsis_risk
                
                # ICU LOS influenced by severity and outcomes
                base_icu_los = max(1, int(np.random.gamma(2, 2)))
                if has_sepsis:
                    base_icu_los *= 1.5
                if has_aki:
                    base_icu_los *= 1.3
                if is_mortality:
                    base_icu_los *= 0.8  # May die sooner
                
                icu_los = max(1, min(int(base_icu_los), 30))
                
                # Total hospital LOS
                total_los = icu_los + np.random.poisson(2)
                discharge_date = admission_date + timedelta(days=total_los)
                
                # Determine discharge disposition
                if is_mortality:
                    disposition = 'Expired'
                elif age > 75 and np.random.random() < 0.3:
                    disposition = 'Skilled Nursing Facility'
                else:
                    disposition = np.random.choice(['Home', 'Transfer to Another Hospital'], p=[0.85, 0.15])
                
                # Mechanical ventilation based on severity
                vent_risk = 0.3 * risk_score
                if has_sepsis:
                    vent_risk *= 2
                requires_vent = np.random.random() < vent_risk
                
                hosp_data = {
                    'hospitalization_id': f"H{hosp_id:08d}",
                    'patient_id': patient_id,
                    'admission_dttm': admission_date,
                    'discharge_dttm': discharge_date,
                    'age_at_admission': age,
                    'admission_source': np.random.choice(admission_sources),
                    'discharge_disposition': disposition,
                    'icu_los_days': icu_los,
                    'hospital_los_days': total_los,
                    'mortality': int(is_mortality),
                    'icu_mortality': int(is_mortality and icu_los >= total_los * 0.8),
                    'aki': int(has_aki),
                    'sepsis': int(has_sepsis),
                    'mechanical_ventilation': int(requires_vent),
                    'readmission_30d': 0  # Will be calculated later
                }
                
                self.hospitalization_data.append(hosp_data)
                hosp_id += 1
        
        # Calculate readmissions
        self._calculate_readmissions()
    
    def _calculate_readmissions(self):
        """Calculate 30-day readmission outcomes."""
        # Group by patient and sort by admission date
        patient_hosps = {}
        for hosp in self.hospitalization_data:
            patient_id = hosp['patient_id']
            if patient_id not in patient_hosps:
                patient_hosps[patient_id] = []
            patient_hosps[patient_id].append(hosp)
        
        # Sort hospitalizations by admission date for each patient
        for patient_id in patient_hosps:
            patient_hosps[patient_id].sort(key=lambda x: x['admission_dttm'])
        
        # Mark readmissions
        for patient_id, hosps in patient_hosps.items():
            for i in range(len(hosps) - 1):
                current_hosp = hosps[i]
                next_hosp = hosps[i + 1]
                
                # Skip if current hospitalization ended in death
                if current_hosp['discharge_disposition'] == 'Expired':
                    continue
                
                # Check if next admission is within 30 days
                days_between = (next_hosp['admission_dttm'] - current_hosp['discharge_dttm']).days
                
                if 0 <= days_between <= 30:
                    # Update the original hospitalization record
                    for hosp in self.hospitalization_data:
                        if hosp['hospitalization_id'] == current_hosp['hospitalization_id']:
                            hosp['readmission_30d'] = 1
                            break
    
    def _generate_adt_with_patterns(self):
        """Generate ADT data with realistic location patterns."""
        icu_locations = ['ICU', 'MICU', 'SICU', 'CVICU', 'NICU']
        ward_locations = ['Medical Ward', 'Surgical Ward', 'Step-Down Unit', 'Telemetry']
        
        for hosp in self.hospitalization_data:
            hosp_id = hosp['hospitalization_id']
            admission_time = hosp['admission_dttm']
            discharge_time = hosp['discharge_dttm']
            icu_los = hosp['icu_los_days']
            
            current_time = admission_time
            
            # ICU stay
            icu_location = np.random.choice(icu_locations)
            icu_end_time = admission_time + timedelta(days=icu_los)
            
            self.adt_data.append({
                'hospitalization_id': hosp_id,
                'location_category': 'ICU',
                'location_name': icu_location,
                'in_dttm': current_time,
                'out_dttm': icu_end_time
            })
            
            current_time = icu_end_time
            
            # Post-ICU ward stay (if patient didn't die in ICU)
            if current_time < discharge_time:
                ward_location = np.random.choice(ward_locations)
                
                self.adt_data.append({
                    'hospitalization_id': hosp_id,
                    'location_category': 'Ward',
                    'location_name': ward_location,
                    'in_dttm': current_time,
                    'out_dttm': discharge_time
                })
    
    def _generate_vitals_with_trends(self):
        """Generate vitals with realistic temporal patterns and clinical relationships."""
        vital_types = {
            'Heart Rate': {'normal_range': (60, 100), 'unit': 'bpm'},
            'Blood Pressure Systolic': {'normal_range': (90, 140), 'unit': 'mmHg'},
            'Blood Pressure Diastolic': {'normal_range': (60, 90), 'unit': 'mmHg'},
            'Respiratory Rate': {'normal_range': (12, 20), 'unit': '/min'},
            'Temperature': {'normal_range': (36.1, 37.2), 'unit': '°C'},
            'SpO2': {'normal_range': (95, 100), 'unit': '%'}
        }
        
        for hosp in self.hospitalization_data:
            hosp_id = hosp['hospitalization_id']
            admission_time = hosp['admission_dttm']
            discharge_time = hosp['discharge_dttm']
            has_sepsis = hosp['sepsis']
            mortality = hosp['mortality']
            
            # Vitals frequency (every 1-4 hours in ICU)
            total_hours = int((discharge_time - admission_time).total_seconds() / 3600)
            
            for hour in range(0, total_hours, np.random.randint(1, 5)):
                record_time = admission_time + timedelta(hours=hour)
                
                # Skip some records to create realistic missing patterns
                if np.random.random() < 0.1:  # 10% missing
                    continue
                
                for vital_name, vital_info in vital_types.items():
                    # Base vital signs with some individual variation
                    min_normal, max_normal = vital_info['normal_range']
                    base_value = np.random.uniform(min_normal, max_normal)
                    
                    # Apply clinical condition modifiers
                    if has_sepsis:
                        if vital_name == 'Heart Rate':
                            base_value *= 1.3  # Tachycardia
                        elif vital_name == 'Temperature':
                            base_value += np.random.uniform(1, 3)  # Fever
                        elif vital_name == 'Blood Pressure Systolic':
                            base_value *= 0.8  # Hypotension
                    
                    # Deterioration pattern if mortality
                    if mortality:
                        hours_to_death = (discharge_time - record_time).total_seconds() / 3600
                        if hours_to_death < 24:  # Last 24 hours
                            deterioration_factor = 1 - (hours_to_death / 24) * 0.3
                            if vital_name in ['Blood Pressure Systolic', 'SpO2']:
                                base_value *= deterioration_factor
                            elif vital_name == 'Heart Rate':
                                base_value *= (2 - deterioration_factor)
                    
                    # Add measurement noise
                    noise_factor = 0.1
                    value = base_value * (1 + np.random.normal(0, noise_factor))
                    
                    # Apply physiological limits
                    value = max(0, value)
                    if vital_name == 'SpO2':
                        value = min(100, value)
                    
                    self.vitals_data.append({
                        'hospitalization_id': hosp_id,
                        'recorded_dttm': record_time,
                        'vital_category': vital_name,
                        'vital_value': round(value, 1),
                        'vital_unit': vital_info['unit']
                    })
    
    def _generate_labs_with_patterns(self):
        """Generate lab data with clinical patterns and AKI progression."""
        lab_types = {
            'Creatinine': {'normal_range': (0.6, 1.2), 'unit': 'mg/dL'},
            'BUN': {'normal_range': (7, 20), 'unit': 'mg/dL'},
            'Sodium': {'normal_range': (136, 145), 'unit': 'mEq/L'},
            'Potassium': {'normal_range': (3.5, 5.1), 'unit': 'mEq/L'},
            'Chloride': {'normal_range': (98, 107), 'unit': 'mEq/L'},
            'Glucose': {'normal_range': (70, 100), 'unit': 'mg/dL'},
            'Hemoglobin': {'normal_range': (12, 16), 'unit': 'g/dL'},
            'White Blood Cell': {'normal_range': (4.5, 11.0), 'unit': '×10³/μL'},
            'Platelet': {'normal_range': (150, 400), 'unit': '×10³/μL'},
            'Lactate': {'normal_range': (0.5, 2.2), 'unit': 'mmol/L'}
        }
        
        for hosp in self.hospitalization_data:
            hosp_id = hosp['hospitalization_id']
            admission_time = hosp['admission_dttm']
            discharge_time = hosp['discharge_dttm']
            has_aki = hosp['aki']
            has_sepsis = hosp['sepsis']
            
            # Labs typically drawn daily or twice daily in ICU
            total_days = (discharge_time - admission_time).days + 1
            
            for day in range(total_days):
                # 1-2 lab draws per day
                n_draws = np.random.choice([1, 2], p=[0.4, 0.6])
                
                for draw in range(n_draws):
                    hour = 6 + draw * 12 + np.random.randint(-2, 3)  # 6am and 6pm ±2hrs
                    lab_time = admission_time + timedelta(days=day, hours=hour)
                    
                    if lab_time >= discharge_time:
                        break
                    
                    # Skip some lab draws
                    if np.random.random() < 0.05:  # 5% missing
                        continue
                    
                    for lab_name, lab_info in lab_types.items():
                        min_normal, max_normal = lab_info['normal_range']
                        base_value = np.random.uniform(min_normal, max_normal)
                        
                        # Apply clinical condition patterns
                        if has_aki and lab_name in ['Creatinine', 'BUN']:
                            # AKI progression pattern
                            days_since_admission = day
                            if lab_name == 'Creatinine':
                                # Creatinine rises over first few days then may plateau
                                if days_since_admission <= 3:
                                    aki_multiplier = 1 + (days_since_admission * 0.5)
                                else:
                                    aki_multiplier = 2.5  # Peak level
                                base_value *= aki_multiplier
                            elif lab_name == 'BUN':
                                # BUN rises with creatinine but more variably
                                base_value *= np.random.uniform(1.5, 3.0)
                        
                        if has_sepsis:
                            if lab_name == 'White Blood Cell':
                                # Can be high or low in sepsis
                                sepsis_modifier = np.random.choice([0.3, 2.5], p=[0.2, 0.8])
                                base_value *= sepsis_modifier
                            elif lab_name == 'Lactate':
                                # Elevated lactate in sepsis
                                base_value *= np.random.uniform(1.5, 4.0)
                            elif lab_name == 'Platelet':
                                # Thrombocytopenia in sepsis
                                base_value *= np.random.uniform(0.3, 0.8)
                        
                        # Add measurement variation
                        noise_factor = 0.08
                        value = base_value * (1 + np.random.normal(0, noise_factor))
                        value = max(0, value)  # No negative lab values
                        
                        self.labs_data.append({
                            'hospitalization_id': hosp_id,
                            'lab_collected_dttm': lab_time,
                            'lab_category': lab_name,
                            'lab_value': round(value, 2),
                            'lab_unit': lab_info['unit']
                        })
    
    def _generate_respiratory_support(self):
        """Generate respiratory support data for ventilated patients."""
        vent_modes = ['Volume Control', 'Pressure Control', 'SIMV', 'PSV', 'CPAP']
        
        for hosp in self.hospitalization_data:
            if not hosp['mechanical_ventilation']:
                continue
                
            hosp_id = hosp['hospitalization_id']
            admission_time = hosp['admission_dttm']
            discharge_time = hosp['discharge_dttm']
            
            # Ventilation typically starts within hours of admission
            vent_start = admission_time + timedelta(hours=np.random.randint(1, 12))
            
            # Ventilation duration (subset of ICU stay)
            icu_hours = hosp['icu_los_days'] * 24
            vent_duration_hours = np.random.randint(12, min(icu_hours, 240))  # Max 10 days
            vent_end = vent_start + timedelta(hours=vent_duration_hours)
            
            # Ensure ventilation doesn't exceed discharge
            vent_end = min(vent_end, discharge_time)
            
            self.respiratory_support_data.append({
                'hospitalization_id': hosp_id,
                'device_category': 'Mechanical Ventilator',
                'device_name': 'Ventilator',
                'mode': np.random.choice(vent_modes),
                'start_dttm': vent_start,
                'end_dttm': vent_end,
                'duration_hours': (vent_end - vent_start).total_seconds() / 3600
            })
    
    def _generate_medications_with_protocols(self):
        """Generate medication data following clinical protocols."""
        medications = {
            'sepsis_protocol': [
                'Norepinephrine', 'Vancomycin', 'Piperacillin-Tazobactam',
                'Hydrocortisone', 'Lactated Ringers'
            ],
            'standard_icu': [
                'Propofol', 'Fentanyl', 'Midazolam', 'Heparin',
                'Pantoprazole', 'Insulin', 'Furosemide'
            ],
            'cardiac': [
                'Metoprolol', 'Lisinopril', 'Atorvastatin', 'Aspirin'
            ]
        }
        
        for hosp in self.hospitalization_data:
            hosp_id = hosp['hospitalization_id']
            admission_time = hosp['admission_dttm']
            discharge_time = hosp['discharge_dttm']
            has_sepsis = hosp['sepsis']
            
            selected_meds = []
            
            # Sepsis protocol
            if has_sepsis:
                selected_meds.extend(
                    np.random.choice(medications['sepsis_protocol'], 
                                   size=np.random.randint(3, 6), 
                                   replace=False)
                )
            
            # Standard ICU medications
            selected_meds.extend(
                np.random.choice(medications['standard_icu'],
                               size=np.random.randint(2, 5),
                               replace=False)
            )
            
            # Remove duplicates
            selected_meds = list(set(selected_meds))
            
            for med_name in selected_meds:
                # Medication timing varies
                start_delay_hours = np.random.randint(0, 24)
                med_start = admission_time + timedelta(hours=start_delay_hours)
                
                # Duration varies by medication type
                if med_name in medications['sepsis_protocol']:
                    duration_hours = np.random.randint(48, 120)  # 2-5 days
                else:
                    duration_hours = np.random.randint(24, 72)   # 1-3 days
                
                med_end = med_start + timedelta(hours=duration_hours)
                med_end = min(med_end, discharge_time)
                
                # Generate administration records (every 4-12 hours)
                current_time = med_start
                while current_time < med_end:
                    self.medication_admin_data.append({
                        'hospitalization_id': hosp_id,
                        'medication_name': med_name,
                        'administration_dttm': current_time,
                        'dose': f"{np.random.randint(1, 10)} mg",  # Simplified dosing
                        'route': np.random.choice(['IV', 'PO', 'SQ'])
                    })
                    
                    # Next administration
                    interval_hours = np.random.randint(4, 12)
                    current_time += timedelta(hours=interval_hours)
    
    def _generate_assessments(self):
        """Generate patient assessment data."""
        assessment_types = [
            'Glasgow Coma Scale', 'Richmond Agitation-Sedation Scale',
            'Confusion Assessment Method', 'Braden Scale', 'Pain Scale'
        ]
        
        for hosp in self.hospitalization_data:
            hosp_id = hosp['hospitalization_id']
            admission_time = hosp['admission_dttm']
            discharge_time = hosp['discharge_dttm']
            
            # Assessments typically done daily
            total_days = (discharge_time - admission_time).days + 1
            
            for day in range(total_days):
                assessment_time = admission_time + timedelta(days=day, hours=np.random.randint(6, 18))
                
                if assessment_time >= discharge_time:
                    break
                
                for assessment_type in assessment_types:
                    # Not all assessments done every day
                    if np.random.random() < 0.7:
                        
                        # Generate realistic scores
                        if assessment_type == 'Glasgow Coma Scale':
                            score = np.random.randint(8, 16)
                        elif assessment_type == 'Richmond Agitation-Sedation Scale':
                            score = np.random.randint(-5, 5)
                        elif assessment_type == 'Pain Scale':
                            score = np.random.randint(0, 11)
                        else:
                            score = np.random.randint(0, 10)
                        
                        self.patient_assessments_data.append({
                            'hospitalization_id': hosp_id,
                            'assessment_dttm': assessment_time,
                            'assessment_type': assessment_type,
                            'assessment_value': score
                        })
    
    def _apply_missing_data_patterns(self):
        """Apply realistic missing data patterns to all datasets."""
        # Apply missing data to patient comorbidity information
        for patient_id, patient_data in self.patient_data.items():
            if np.random.random() < self.missing_data_rate * 0.5:  # Less missing for demographics
                if np.random.random() < 0.3:
                    patient_data['race'] = None
                if np.random.random() < 0.2:
                    patient_data['ethnicity'] = None
        
        # Missing patterns in time-series data tend to be clustered
        # (e.g., lab draws missed for entire shifts)
        self._apply_clustered_missing_pattern(self.vitals_data, 'recorded_dttm')
        self._apply_clustered_missing_pattern(self.labs_data, 'lab_collected_dttm')
    
    def _apply_clustered_missing_pattern(self, data_list: List[Dict], time_col: str):
        """Apply clustered missing data patterns to time-series data."""
        # Group by hospitalization
        hosp_groups = {}
        for record in data_list:
            hosp_id = record['hospitalization_id']
            if hosp_id not in hosp_groups:
                hosp_groups[hosp_id] = []
            hosp_groups[hosp_id].append(record)
        
        # Apply missing patterns by hospitalization
        for hosp_id, records in hosp_groups.items():
            # Sort by time
            records.sort(key=lambda x: x[time_col])
            
            # Create missing clusters (6-12 hour periods)
            total_records = len(records)
            n_missing_clusters = int(total_records * self.missing_data_rate * 0.3)
            
            for _ in range(n_missing_clusters):
                # Select random start point
                start_idx = np.random.randint(0, max(1, total_records - 5))
                cluster_size = np.random.randint(2, 6)
                
                # Mark records for removal
                for idx in range(start_idx, min(start_idx + cluster_size, total_records)):
                    if idx < len(records) and np.random.random() < 0.8:
                        records[idx]['_to_remove'] = True
        
        # Remove marked records
        indices_to_remove = []
        for i, record in enumerate(data_list):
            if record.get('_to_remove', False):
                indices_to_remove.append(i)
        
        # Remove in reverse order to maintain indices
        for idx in sorted(indices_to_remove, reverse=True):
            data_list.pop(idx)
    
    def _save_all_tables(self) -> Dict[str, int]:
        """Save all generated tables to CSV files."""
        record_counts = {}
        
        # Convert patient data to list of dictionaries
        patient_list = []
        for patient_id, patient_info in self.patient_data.items():
            # Flatten comorbidities into separate columns
            patient_record = {
                'patient_id': patient_info['patient_id'],
                'date_of_birth': patient_info['date_of_birth'],
                'race': patient_info['race'],
                'ethnicity': patient_info['ethnicity'],
                'sex': patient_info['sex']
            }
            
            # Add comorbidity flags
            for comorbidity in self.comorbidities:
                patient_record[f'comorbid_{comorbidity}'] = int(comorbidity in patient_info['comorbidities'])
            
            patient_list.append(patient_record)
        
        # Save all tables
        tables = {
            'patient': patient_list,
            'hospitalization': self.hospitalization_data,
            'adt': self.adt_data,
            'vitals': self.vitals_data,
            'labs': self.labs_data,
            'respiratory_support': self.respiratory_support_data,
            'medication_administration': self.medication_admin_data,
            'patient_assessments': self.patient_assessments_data
        }
        
        for table_name, table_data in tables.items():
            if table_data:
                df = pd.DataFrame(table_data)
                
                # Clean up any temporary columns
                if '_to_remove' in df.columns:
                    df = df.drop('_to_remove', axis=1)
                
                filepath = self.output_dir / f"{table_name}.csv"
                df.to_csv(filepath, index=False)
                record_counts[table_name] = len(df)
                print(f"  ✓ {table_name}: {len(df)} records")
        
        return record_counts
    
    def _generate_data_summary(self):
        """Generate summary statistics for the synthetic dataset."""
        summary = {
            'generation_info': {
                'timestamp': datetime.now().isoformat(),
                'n_patients': self.n_patients,
                'missing_data_rate': self.missing_data_rate,
                'outcome_prevalences': self.outcome_prevalence
            },
            'patient_demographics': self._summarize_demographics(),
            'clinical_outcomes': self._summarize_outcomes(),
            'data_quality': self._assess_data_quality()
        }
        
        # Save summary
        summary_path = self.output_dir / "dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"  ✓ Dataset summary saved to {summary_path}")
    
    def _summarize_demographics(self) -> Dict[str, Any]:
        """Summarize patient demographics."""
        ages = [p['age'] for p in self.patient_data.values()]
        sexes = [p['sex'] for p in self.patient_data.values() if p['sex']]
        races = [p['race'] for p in self.patient_data.values() if p['race']]
        
        return {
            'age_mean': np.mean(ages),
            'age_std': np.std(ages),
            'sex_distribution': pd.Series(sexes).value_counts().to_dict() if sexes else {},
            'race_distribution': pd.Series(races).value_counts().to_dict() if races else {}
        }
    
    def _summarize_outcomes(self) -> Dict[str, Any]:
        """Summarize clinical outcomes."""
        if not self.hospitalization_data:
            return {}
        
        outcomes = {}
        for outcome in ['mortality', 'icu_mortality', 'aki', 'sepsis', 'mechanical_ventilation', 'readmission_30d']:
            values = [h[outcome] for h in self.hospitalization_data]
            outcomes[outcome] = {
                'count': sum(values),
                'rate': np.mean(values),
                'total': len(values)
            }
        
        return outcomes
    
    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess data quality metrics."""
        quality_metrics = {}
        
        # Calculate actual missing data rates
        all_tables = {
            'vitals': self.vitals_data,
            'labs': self.labs_data,
            'medications': self.medication_admin_data,
            'assessments': self.patient_assessments_data
        }
        
        for table_name, table_data in all_tables.items():
            if table_data:
                # Calculate completeness
                total_possible = len(table_data)
                quality_metrics[table_name] = {
                    'records_generated': total_possible,
                    'completeness': 1.0  # Simplified for now
                }
        
        return quality_metrics
    
    def _get_age_adjusted_comorbidity_probs(self, age: int) -> Dict[str, float]:
        """Get age-adjusted comorbidity probabilities."""
        base_probs = {
            'diabetes': 0.15,
            'hypertension': 0.25,
            'coronary_artery_disease': 0.12,
            'chronic_kidney_disease': 0.08,
            'copd': 0.10,
            'heart_failure': 0.06,
            'stroke': 0.05,
            'cancer': 0.08,
            'liver_disease': 0.04
        }
        
        # Increase probabilities with age
        age_multiplier = 1 + (age - 50) * 0.02 if age > 50 else 1
        
        return {k: min(0.8, v * age_multiplier) for k, v in base_probs.items()}
    
    def _calculate_patient_risk_score(self, age: int, comorbidities: List[str]) -> float:
        """Calculate patient risk score based on age and comorbidities."""
        age_score = (age - 18) / 72  # Normalized age score (0-1)
        comorbidity_score = len(comorbidities) / 5  # Normalized comorbidity burden
        
        return 1 + (age_score * 0.5) + (comorbidity_score * 0.8)


def main():
    """Main function to generate synthetic dataset."""
    generator = CLIFSyntheticGenerator(
        n_patients=1000,
        missing_data_rate=0.15,
        output_dir="./data/synthetic"
    )
    
    record_counts = generator.generate_complete_dataset()
    
    print(f"\n🎉 Synthetic dataset generation complete!")
    print(f"Total patients: {generator.n_patients}")
    print(f"Total hospitalizations: {record_counts.get('hospitalization', 0)}")
    print(f"Tables generated: {list(record_counts.keys())}")


if __name__ == "__main__":
    main()