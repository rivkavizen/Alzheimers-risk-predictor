"""Column definitions and validation rules for data cleaning."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "alzheimers_disease_data.csv"
CLEANED_DATA_PATH = PROJECT_ROOT / "alzheimers_disease_data_cleaned.csv"

# Columns dropped before saving cleaned data (not used for ML)
DROP_COLUMNS = ["PatientID", "DoctorInCharge"]

TARGET_COLUMN = "Diagnosis"

BINARY_COLUMNS = [
    "Gender",
    "Smoking",
    "FamilyHistoryAlzheimers",
    "CardiovascularDisease",
    "Diabetes",
    "Depression",
    "HeadInjury",
    "Hypertension",
    "MemoryComplaints",
    "BehavioralProblems",
    "Confusion",
    "Disorientation",
    "PersonalityChanges",
    "DifficultyCompletingTasks",
    "Forgetfulness",
]

ORDINAL_COLUMNS = {
    "Ethnicity": (0, 3),
    "EducationLevel": (0, 3),
}

CONTINUOUS_RANGES = {
    "Age": (50, 100),
    "BMI": (10, 60),
    "AlcoholConsumption": (0, 30),
    "PhysicalActivity": (0, 15),
    "DietQuality": (0, 10),
    "SleepQuality": (0, 10),
    "SystolicBP": (60, 200),
    "DiastolicBP": (40, 130),
    "CholesterolTotal": (50, 400),
    "CholesterolLDL": (20, 300),
    "CholesterolHDL": (10, 120),
    "CholesterolTriglycerides": (30, 500),
    "MMSE": (0, 30),
    "FunctionalAssessment": (0, 10),
    "ADL": (0, 10),
}
