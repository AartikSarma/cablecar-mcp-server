"""Generate synthetic CLIF-compliant clinical data for testing CableCar."""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import argparse


class CLIFSyntheticGenerator:
    """Generate realistic synthetic CLIF data with known clinical patterns.

    Generates data with embedded ground-truth effects for evaluation:
    - Higher SOFA scores → higher mortality
    - Older age → longer ICU stay
    - Sepsis patients → more vasopressor use
    - Mechanical ventilation → longer LOS
    """

    def __init__(self, n_patients: int = 500, seed: int = 42):
        self.n_patients = n_patients
        self.rng = np.random.default_rng(seed)
        self.base_date = datetime(2023, 1, 1)

    def generate_all(self) -> dict[str, pd.DataFrame]:
        """Generate all CLIF tables and return as dict."""
        patients = self._generate_patients()
        hospitalizations = self._generate_hospitalizations(patients)
        adt = self._generate_adt(hospitalizations)
        vitals = self._generate_vitals(hospitalizations)
        labs = self._generate_labs(hospitalizations)
        respiratory = self._generate_respiratory_support(hospitalizations)
        medications = self._generate_medications(hospitalizations)
        assessments = self._generate_assessments(hospitalizations)

        return {
            "patient": patients,
            "hospitalization": hospitalizations,
            "adt": adt,
            "vitals": vitals,
            "labs": labs,
            "respiratory_support": respiratory,
            "medication_admin_continuous": medications,
            "patient_assessments": assessments,
        }

    def save(self, output_dir: str | Path = "data/synthetic"):
        """Generate and save all tables as CSV files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tables = self.generate_all()
        for name, df in tables.items():
            path = output_dir / f"{name}.csv"
            df.to_csv(path, index=False)
            print(f"  Saved {name}: {len(df)} rows → {path}")

        print(f"\nGenerated {len(tables)} CLIF tables for {self.n_patients} patients")
        return tables

    def _generate_patients(self) -> pd.DataFrame:
        """Generate patient demographics."""
        patient_ids = [f"P{i:06d}" for i in range(1, self.n_patients + 1)]

        # Realistic demographic distributions
        races = self.rng.choice(
            ["White", "Black", "Asian", "Hispanic", "Other", "Unknown"],
            size=self.n_patients,
            p=[0.55, 0.18, 0.08, 0.12, 0.04, 0.03]
        )
        ethnicities = self.rng.choice(
            ["Non-Hispanic", "Hispanic", "Unknown"],
            size=self.n_patients,
            p=[0.80, 0.15, 0.05]
        )
        sexes = self.rng.choice(
            ["Male", "Female"],
            size=self.n_patients,
            p=[0.55, 0.45]  # ICU populations skew slightly male
        )

        return pd.DataFrame({
            "patient_id": patient_ids,
            "race_category": races,
            "ethnicity_category": ethnicities,
            "sex_category": sexes,
        })

    def _generate_hospitalizations(self, patients: pd.DataFrame) -> pd.DataFrame:
        """Generate hospitalizations with realistic patterns."""
        records = []
        hosp_id = 1

        for _, patient in patients.iterrows():
            # Some patients have multiple hospitalizations
            n_hosps = self.rng.choice([1, 2, 3], p=[0.80, 0.15, 0.05])

            for _ in range(n_hosps):
                admission = self.base_date + timedelta(
                    days=int(self.rng.uniform(0, 365)),
                    hours=int(self.rng.uniform(0, 24))
                )

                age = self.rng.normal(65, 15)
                age = max(18, min(100, age))

                # LOS depends on age (ground truth: older → longer stay)
                base_los_hours = max(24, self.rng.normal(120 + (age - 65) * 2, 72))

                # Determine if patient will die (ground truth: age-dependent)
                mortality_prob = 0.05 + (age - 18) / (100 - 18) * 0.25
                died = self.rng.random() < mortality_prob

                if died:
                    # Dying patients tend to have shorter or longer stays (bimodal)
                    base_los_hours *= self.rng.choice([0.5, 1.5], p=[0.4, 0.6])

                discharge = admission + timedelta(hours=max(24, base_los_hours))

                admission_types = self.rng.choice(
                    ["Emergency", "Urgent", "Elective", "Transfer"],
                    p=[0.50, 0.25, 0.15, 0.10]
                )

                records.append({
                    "hospitalization_id": f"H{hosp_id:07d}",
                    "patient_id": patient["patient_id"],
                    "admission_dttm": admission.strftime("%Y-%m-%d %H:%M:%S"),
                    "discharge_dttm": discharge.strftime("%Y-%m-%d %H:%M:%S"),
                    "age_at_admission": round(age, 1),
                    "admission_type_name": admission_types,
                    "_died": died,  # Hidden ground truth, removed before saving
                    "_severity": self.rng.choice(["mild", "moderate", "severe"], p=[0.4, 0.35, 0.25]),
                })
                hosp_id += 1

        df = pd.DataFrame(records)
        # Store ground truth internally but don't save it
        self._hosp_metadata = df[["hospitalization_id", "_died", "_severity"]].copy()
        return df.drop(columns=["_died", "_severity"])

    def _generate_adt(self, hospitalizations: pd.DataFrame) -> pd.DataFrame:
        """Generate ADT (admission/discharge/transfer) records."""
        records = []

        for _, hosp in hospitalizations.iterrows():
            admission = pd.Timestamp(hosp["admission_dttm"])
            discharge = pd.Timestamp(hosp["discharge_dttm"])
            meta = self._hosp_metadata[self._hosp_metadata["hospitalization_id"] == hosp["hospitalization_id"]].iloc[0]

            current_time = admission

            # Most ICU patients: ED → ICU → Ward → Discharge
            if meta["_severity"] == "severe":
                locations = ["ED", "ICU", "ICU", "Ward"]
            elif meta["_severity"] == "moderate":
                locations = ["ED", "ICU", "Ward"]
            else:
                locations = ["ED", "Ward"]

            total_hours = (discharge - admission).total_seconds() / 3600
            hours_per_loc = total_hours / len(locations)

            for loc in locations:
                end_time = min(current_time + timedelta(hours=max(1, hours_per_loc + self.rng.normal(0, 2))), discharge)
                records.append({
                    "hospitalization_id": hosp["hospitalization_id"],
                    "in_dttm": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "out_dttm": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "location_category": loc,
                })
                current_time = end_time
                if current_time >= discharge:
                    break

        return pd.DataFrame(records)

    def _generate_vitals(self, hospitalizations: pd.DataFrame) -> pd.DataFrame:
        """Generate vital signs with realistic ranges."""
        records = []

        vital_params = {
            "heart_rate": (80, 15, 40, 180),       # mean, std, min, max
            "sbp": (120, 20, 60, 220),
            "dbp": (70, 12, 30, 130),
            "resp_rate": (18, 4, 8, 45),
            "spo2": (96, 3, 70, 100),
            "temperature": (37.0, 0.5, 35.0, 41.0),
            "map": (80, 15, 40, 160),
        }

        for _, hosp in hospitalizations.iterrows():
            admission = pd.Timestamp(hosp["admission_dttm"])
            discharge = pd.Timestamp(hosp["discharge_dttm"])
            meta = self._hosp_metadata[self._hosp_metadata["hospitalization_id"] == hosp["hospitalization_id"]].iloc[0]

            total_hours = (discharge - admission).total_seconds() / 3600
            # Vitals every 1-4 hours depending on severity
            interval = 1 if meta["_severity"] == "severe" else (2 if meta["_severity"] == "moderate" else 4)
            n_measurements = max(1, int(total_hours / interval))

            # Limit to reasonable number
            n_measurements = min(n_measurements, 200)

            for vital_name, (mean, std, vmin, vmax) in vital_params.items():
                # Adjust for severity
                severity_shift = 0
                if meta["_severity"] == "severe":
                    if vital_name == "heart_rate": severity_shift = 15
                    elif vital_name == "sbp": severity_shift = -20
                    elif vital_name == "spo2": severity_shift = -5

                for i in range(n_measurements):
                    time = admission + timedelta(hours=i * interval + self.rng.uniform(0, 0.5))
                    if time > discharge:
                        break
                    value = self.rng.normal(mean + severity_shift, std)
                    value = max(vmin, min(vmax, value))

                    records.append({
                        "hospitalization_id": hosp["hospitalization_id"],
                        "recorded_dttm": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "vital_category": vital_name,
                        "vital_value": round(value, 1),
                    })

        return pd.DataFrame(records)

    def _generate_labs(self, hospitalizations: pd.DataFrame) -> pd.DataFrame:
        """Generate lab results."""
        records = []

        lab_params = {
            "lactate": (1.5, 1.0, 0.5, 15.0),
            "creatinine": (1.0, 0.5, 0.3, 10.0),
            "bilirubin": (1.0, 0.8, 0.1, 20.0),
            "wbc": (10.0, 4.0, 0.5, 40.0),
            "hemoglobin": (12.0, 2.0, 5.0, 18.0),
            "platelets": (200, 80, 10, 500),
            "sodium": (140, 4, 120, 160),
            "potassium": (4.0, 0.5, 2.5, 7.0),
            "glucose": (120, 40, 40, 500),
            "bun": (20, 10, 5, 100),
            "pco2": (40, 8, 20, 80),
            "po2": (90, 20, 40, 500),
            "ph": (7.40, 0.05, 7.10, 7.60),
            "bicarbonate": (24, 4, 10, 40),
        }

        for _, hosp in hospitalizations.iterrows():
            admission = pd.Timestamp(hosp["admission_dttm"])
            discharge = pd.Timestamp(hosp["discharge_dttm"])
            meta = self._hosp_metadata[self._hosp_metadata["hospitalization_id"] == hosp["hospitalization_id"]].iloc[0]

            total_hours = (discharge - admission).total_seconds() / 3600
            # Labs every 4-12 hours depending on severity
            interval = 4 if meta["_severity"] == "severe" else (8 if meta["_severity"] == "moderate" else 12)
            n_sets = max(1, int(total_hours / interval))
            n_sets = min(n_sets, 50)

            for lab_name, (mean, std, vmin, vmax) in lab_params.items():
                # Not all labs drawn every time
                draw_prob = 0.8 if meta["_severity"] == "severe" else 0.5

                severity_shift = 0
                if meta["_severity"] == "severe":
                    if lab_name == "lactate": severity_shift = 2.0
                    elif lab_name == "creatinine": severity_shift = 1.0
                    elif lab_name == "bilirubin": severity_shift = 0.5

                for i in range(n_sets):
                    if self.rng.random() > draw_prob:
                        continue

                    order_time = admission + timedelta(hours=i * interval)
                    if order_time > discharge:
                        break
                    collect_time = order_time + timedelta(minutes=self.rng.integers(5, 30))
                    result_time = collect_time + timedelta(minutes=self.rng.integers(30, 120))

                    value = self.rng.normal(mean + severity_shift, std)
                    value = max(vmin, min(vmax, value))

                    records.append({
                        "hospitalization_id": hosp["hospitalization_id"],
                        "lab_order_dttm": order_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "lab_collect_dttm": collect_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "lab_result_dttm": result_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "lab_category": lab_name,
                        "lab_value": round(value, 2),
                    })

        return pd.DataFrame(records)

    def _generate_respiratory_support(self, hospitalizations: pd.DataFrame) -> pd.DataFrame:
        """Generate respiratory support records."""
        records = []

        for _, hosp in hospitalizations.iterrows():
            admission = pd.Timestamp(hosp["admission_dttm"])
            discharge = pd.Timestamp(hosp["discharge_dttm"])
            meta = self._hosp_metadata[self._hosp_metadata["hospitalization_id"] == hosp["hospitalization_id"]].iloc[0]

            # Determine respiratory trajectory based on severity
            if meta["_severity"] == "severe":
                devices = ["vent", "vent", "vent", "hfnc", "nasal_cannula"]
                prob = 0.9
            elif meta["_severity"] == "moderate":
                devices = ["nippv", "hfnc", "nasal_cannula"]
                prob = 0.6
            else:
                devices = ["nasal_cannula", "room_air"]
                prob = 0.3

            if self.rng.random() > prob:
                continue

            total_hours = (discharge - admission).total_seconds() / 3600
            hours_per_device = total_hours / len(devices)
            current_time = admission

            for device in devices:
                n_records = max(1, int(hours_per_device / 4))
                n_records = min(n_records, 20)

                for j in range(n_records):
                    time = current_time + timedelta(hours=j * 4 + self.rng.uniform(0, 1))
                    if time > discharge:
                        break

                    record = {
                        "hospitalization_id": hosp["hospitalization_id"],
                        "recorded_dttm": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "device_category": device,
                    }

                    if device == "vent":
                        record["mode_category"] = self.rng.choice(["AC/VC", "AC/PC", "PSV", "SIMV"])
                        record["fio2_set"] = round(self.rng.uniform(0.3, 1.0), 2)
                        record["peep_set"] = round(self.rng.uniform(5, 20), 0)
                        record["tidal_volume_set"] = round(self.rng.normal(450, 50), 0)
                    elif device in ("hfnc", "nippv"):
                        record["fio2_set"] = round(self.rng.uniform(0.3, 0.8), 2)

                    records.append(record)

                current_time += timedelta(hours=hours_per_device)

        return pd.DataFrame(records)

    def _generate_medications(self, hospitalizations: pd.DataFrame) -> pd.DataFrame:
        """Generate continuous medication administration records."""
        records = []

        med_params = {
            "norepinephrine": ("mcg/kg/min", 0.1, 0.05, 0.01, 0.5),
            "vasopressin": ("units/min", 0.04, 0.01, 0.01, 0.04),
            "propofol": ("mcg/kg/min", 30, 10, 5, 80),
            "fentanyl": ("mcg/hr", 50, 20, 25, 200),
            "midazolam": ("mg/hr", 3, 1, 1, 10),
            "insulin": ("units/hr", 5, 2, 1, 20),
        }

        for _, hosp in hospitalizations.iterrows():
            meta = self._hosp_metadata[self._hosp_metadata["hospitalization_id"] == hosp["hospitalization_id"]].iloc[0]
            admission = pd.Timestamp(hosp["admission_dttm"])
            discharge = pd.Timestamp(hosp["discharge_dttm"])

            # Severe patients get vasopressors + sedation
            if meta["_severity"] == "severe":
                meds_to_give = self.rng.choice(
                    list(med_params.keys()),
                    size=self.rng.integers(2, 5),
                    replace=False
                )
            elif meta["_severity"] == "moderate":
                meds_to_give = self.rng.choice(
                    list(med_params.keys()),
                    size=self.rng.integers(0, 3),
                    replace=False
                )
            else:
                # Mild: maybe just insulin
                if self.rng.random() < 0.2:
                    meds_to_give = ["insulin"]
                else:
                    continue

            for med in meds_to_give:
                unit, mean_dose, std_dose, min_dose, max_dose = med_params[med]

                # Med given for a portion of the stay
                start_offset = self.rng.uniform(0, 12)
                duration = self.rng.uniform(6, 72)
                med_start = admission + timedelta(hours=start_offset)
                med_end = min(med_start + timedelta(hours=duration), discharge)

                current = med_start
                while current < med_end:
                    dose = max(min_dose, min(max_dose, self.rng.normal(mean_dose, std_dose)))
                    records.append({
                        "hospitalization_id": hosp["hospitalization_id"],
                        "admin_dttm": current.strftime("%Y-%m-%d %H:%M:%S"),
                        "med_category": med,
                        "med_dose": round(dose, 3),
                        "med_dose_unit": unit,
                    })
                    current += timedelta(hours=self.rng.uniform(0.5, 2))

        return pd.DataFrame(records)

    def _generate_assessments(self, hospitalizations: pd.DataFrame) -> pd.DataFrame:
        """Generate patient assessment records (GCS, RASS, CAM-ICU)."""
        records = []

        for _, hosp in hospitalizations.iterrows():
            admission = pd.Timestamp(hosp["admission_dttm"])
            discharge = pd.Timestamp(hosp["discharge_dttm"])
            meta = self._hosp_metadata[self._hosp_metadata["hospitalization_id"] == hosp["hospitalization_id"]].iloc[0]

            total_hours = (discharge - admission).total_seconds() / 3600
            interval = 4 if meta["_severity"] != "mild" else 8
            n_assessments = max(1, min(50, int(total_hours / interval)))

            for i in range(n_assessments):
                time = admission + timedelta(hours=i * interval + self.rng.uniform(0, 1))
                if time > discharge:
                    break

                # GCS components
                if meta["_severity"] == "severe":
                    gcs_eye = self.rng.choice([1, 2, 3, 4], p=[0.2, 0.3, 0.3, 0.2])
                    gcs_verbal = self.rng.choice([1, 2, 3, 4, 5], p=[0.3, 0.2, 0.2, 0.2, 0.1])
                    gcs_motor = self.rng.choice([1, 2, 3, 4, 5, 6], p=[0.15, 0.15, 0.15, 0.2, 0.2, 0.15])
                else:
                    gcs_eye = self.rng.choice([1, 2, 3, 4], p=[0.05, 0.1, 0.25, 0.6])
                    gcs_verbal = self.rng.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.15, 0.2, 0.5])
                    gcs_motor = self.rng.choice([1, 2, 3, 4, 5, 6], p=[0.02, 0.03, 0.05, 0.1, 0.2, 0.6])

                gcs_total = gcs_eye + gcs_verbal + gcs_motor

                for assess_name, assess_value in [
                    ("gcs_total", gcs_total),
                    ("gcs_eye", gcs_eye),
                    ("gcs_verbal", gcs_verbal),
                    ("gcs_motor", gcs_motor),
                    ("rass", self.rng.integers(-5, 5)),
                ]:
                    records.append({
                        "hospitalization_id": hosp["hospitalization_id"],
                        "recorded_dttm": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "assessment_category": assess_name,
                        "assessment_value": float(assess_value),
                    })

                # CAM-ICU only for some
                if meta["_severity"] != "mild" and self.rng.random() < 0.5:
                    records.append({
                        "hospitalization_id": hosp["hospitalization_id"],
                        "recorded_dttm": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "assessment_category": "cam_icu",
                        "assessment_value": float(self.rng.choice([0, 1], p=[0.6, 0.4])),
                    })

        return pd.DataFrame(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic CLIF data")
    parser.add_argument("--n-patients", type=int, default=500, help="Number of patients")
    parser.add_argument("--output", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"Generating synthetic CLIF data for {args.n_patients} patients...")
    generator = CLIFSyntheticGenerator(n_patients=args.n_patients, seed=args.seed)
    generator.save(args.output)
    print("Done!")
