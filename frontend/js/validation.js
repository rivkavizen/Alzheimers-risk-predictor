/** Field validation rules (aligned with ml/cleaning_config.py). */

export const FIELD_RULES = {
  age: { min: 50, max: 100, label: "Age" },
  gender: { min: 0, max: 1, integer: true, label: "Gender" },
  ethnicity: { min: 0, max: 3, integer: true, label: "Ethnicity" },
  education_level: { min: 0, max: 3, integer: true, label: "Education level" },
  bmi: { min: 10, max: 60, label: "BMI" },
  smoking: { min: 0, max: 1, integer: true, label: "Smoking" },
  alcohol_consumption: { min: 0, max: 30, label: "Alcohol consumption" },
  physical_activity: { min: 0, max: 15, label: "Physical activity" },
  diet_quality: { min: 0, max: 10, label: "Diet quality" },
  sleep_quality: { min: 0, max: 10, label: "Sleep quality" },
  family_history_alzheimers: { min: 0, max: 1, integer: true, label: "Family history" },
  cardiovascular_disease: { min: 0, max: 1, integer: true, label: "Cardiovascular disease" },
  diabetes: { min: 0, max: 1, integer: true, label: "Diabetes" },
  depression: { min: 0, max: 1, integer: true, label: "Depression" },
  head_injury: { min: 0, max: 1, integer: true, label: "Head injury" },
  hypertension: { min: 0, max: 1, integer: true, label: "Hypertension" },
  systolic_bp: { min: 60, max: 200, label: "Systolic BP" },
  diastolic_bp: { min: 40, max: 130, label: "Diastolic BP" },
  cholesterol_total: { min: 50, max: 400, label: "Total cholesterol" },
  cholesterol_ldl: { min: 20, max: 300, label: "LDL cholesterol" },
  cholesterol_hdl: { min: 10, max: 120, label: "HDL cholesterol" },
  cholesterol_triglycerides: { min: 30, max: 500, label: "Triglycerides" },
  mmse: { min: 0, max: 30, label: "MMSE" },
  functional_assessment: { min: 0, max: 10, label: "Functional assessment" },
  memory_complaints: { min: 0, max: 1, integer: true, label: "Memory complaints" },
  behavioral_problems: { min: 0, max: 1, integer: true, label: "Behavioral problems" },
  adl: { min: 0, max: 10, label: "ADL" },
  confusion: { min: 0, max: 1, integer: true, label: "Confusion" },
  disorientation: { min: 0, max: 1, integer: true, label: "Disorientation" },
  personality_changes: { min: 0, max: 1, integer: true, label: "Personality changes" },
  difficulty_completing_tasks: { min: 0, max: 1, integer: true, label: "Difficulty completing tasks" },
  forgetfulness: { min: 0, max: 1, integer: true, label: "Forgetfulness" },
};

export function validateAssessment(data) {
  const errors = [];

  for (const [field, rule] of Object.entries(FIELD_RULES)) {
    const raw = data[field];
    const label = rule.label || field;

    if (raw === undefined || raw === null || raw === "") {
      errors.push(`${label}: value is required`);
      continue;
    }

    const value = Number(raw);
    if (Number.isNaN(value)) {
      errors.push(`${label}: must be a number`);
      continue;
    }

    if (rule.integer && !Number.isInteger(value)) {
      errors.push(`${label}: must be a whole number (${rule.min}–${rule.max})`);
      continue;
    }

    if (value < rule.min || value > rule.max) {
      errors.push(`${label}: must be between ${rule.min} and ${rule.max} (got ${value})`);
    }
  }

  return errors;
}
