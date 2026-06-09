export const FIELD_GROUPS = {
  demographics: [
    ["age", "Age", "number"],
    ["gender", "Gender (0=F, 1=M)", "number"],
    ["ethnicity", "Ethnicity (0-3)", "number"],
    ["education_level", "Education level (0-3)", "number"],
    ["bmi", "BMI", "number"],
    ["smoking", "Smoking (0/1)", "number"],
    ["alcohol_consumption", "Alcohol consumption", "number"],
    ["physical_activity", "Physical activity", "number"],
    ["diet_quality", "Diet quality", "number"],
    ["sleep_quality", "Sleep quality", "number"],
  ],
  medical: [
    ["family_history_alzheimers", "Family history (0/1)", "number"],
    ["cardiovascular_disease", "Cardiovascular disease (0/1)", "number"],
    ["diabetes", "Diabetes (0/1)", "number"],
    ["depression", "Depression (0/1)", "number"],
    ["head_injury", "Head injury (0/1)", "number"],
    ["hypertension", "Hypertension (0/1)", "number"],
  ],
  labs: [
    ["systolic_bp", "Systolic BP", "number"],
    ["diastolic_bp", "Diastolic BP", "number"],
    ["cholesterol_total", "Total cholesterol", "number"],
    ["cholesterol_ldl", "LDL cholesterol", "number"],
    ["cholesterol_hdl", "HDL cholesterol", "number"],
    ["cholesterol_triglycerides", "Triglycerides", "number"],
  ],
  cognitive: [
    ["mmse", "MMSE (0-30)", "number"],
    ["functional_assessment", "Functional assessment", "number"],
    ["memory_complaints", "Memory complaints (0/1)", "number"],
    ["behavioral_problems", "Behavioral problems (0/1)", "number"],
    ["adl", "ADL score", "number"],
    ["confusion", "Confusion (0/1)", "number"],
    ["disorientation", "Disorientation (0/1)", "number"],
    ["personality_changes", "Personality changes (0/1)", "number"],
    ["difficulty_completing_tasks", "Difficulty completing tasks (0/1)", "number"],
    ["forgetfulness", "Forgetfulness (0/1)", "number"],
  ],
};

export const DEFAULTS = {
  age: 73, gender: 0, ethnicity: 0, education_level: 2, bmi: 22.93,
  smoking: 0, alcohol_consumption: 13.3, physical_activity: 6.33,
  diet_quality: 1.35, sleep_quality: 9.03, family_history_alzheimers: 0,
  cardiovascular_disease: 0, diabetes: 1, depression: 1, head_injury: 0,
  hypertension: 0, systolic_bp: 142, diastolic_bp: 72,
  cholesterol_total: 242.37, cholesterol_ldl: 56.15, cholesterol_hdl: 33.68,
  cholesterol_triglycerides: 162.19, mmse: 21.46, functional_assessment: 6.52,
  memory_complaints: 0, behavioral_problems: 0, adl: 1.73, confusion: 0,
  disorientation: 0, personality_changes: 0, difficulty_completing_tasks: 1,
  forgetfulness: 0,
};

export function collectFormData(form) {
  const data = {};
  new FormData(form).forEach((value, key) => {
    data[key] = Number(value);
  });
  return data;
}

export function renderFields(container, fields) {
  container.innerHTML = fields.map(([name, label, type]) => `
    <label for="${name}">${label}</label>
    <input id="${name}" name="${name}" type="${type}" step="any"
      value="${DEFAULTS[name] ?? ""}" required />
  `).join("");
}
