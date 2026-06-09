import { FIELD_RULES, validateAssessment } from "./validation.js";

export const FIELD_GROUPS = {
  demographics: [
    ["age", "Age (50–100)"],
    ["gender", "Gender (0=F, 1=M)"],
    ["ethnicity", "Ethnicity (0–3)"],
    ["education_level", "Education level (0–3)"],
    ["bmi", "BMI (10–60)"],
    ["smoking", "Smoking (0/1)"],
    ["alcohol_consumption", "Alcohol consumption (0–30)"],
    ["physical_activity", "Physical activity (0–15)"],
    ["diet_quality", "Diet quality (0–10)"],
    ["sleep_quality", "Sleep quality (0–10)"],
  ],
  medical: [
    ["family_history_alzheimers", "Family history (0/1)"],
    ["cardiovascular_disease", "Cardiovascular disease (0/1)"],
    ["diabetes", "Diabetes (0/1)"],
    ["depression", "Depression (0/1)"],
    ["head_injury", "Head injury (0/1)"],
    ["hypertension", "Hypertension (0/1)"],
  ],
  labs: [
    ["systolic_bp", "Systolic BP (60–200)"],
    ["diastolic_bp", "Diastolic BP (40–130)"],
    ["cholesterol_total", "Total cholesterol (50–400)"],
    ["cholesterol_ldl", "LDL cholesterol (20–300)"],
    ["cholesterol_hdl", "HDL cholesterol (10–120)"],
    ["cholesterol_triglycerides", "Triglycerides (30–500)"],
  ],
  cognitive: [
    ["mmse", "MMSE (0–30)"],
    ["functional_assessment", "Functional assessment (0–10)"],
    ["memory_complaints", "Memory complaints (0/1)"],
    ["behavioral_problems", "Behavioral problems (0/1)"],
    ["adl", "ADL score (0–10)"],
    ["confusion", "Confusion (0/1)"],
    ["disorientation", "Disorientation (0/1)"],
    ["personality_changes", "Personality changes (0/1)"],
    ["difficulty_completing_tasks", "Difficulty completing tasks (0/1)"],
    ["forgetfulness", "Forgetfulness (0/1)"],
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

export { validateAssessment };

export function renderFields(container, fields) {
  container.innerHTML = fields.map(([name, label]) => {
    const rule = FIELD_RULES[name];
    const step = rule?.integer ? "1" : "any";
    const min = rule?.min ?? "";
    const max = rule?.max ?? "";
    return `
    <div class="field-wrap" data-field="${name}">
      <label for="${name}">${label}</label>
      <input id="${name}" name="${name}" type="number" step="${step}"
        min="${min}" max="${max}" value="${DEFAULTS[name] ?? ""}" required />
      <small class="field-error muted" id="err-${name}"></small>
    </div>`;
  }).join("");
}

export function showValidationErrors(errors) {
  document.querySelectorAll(".field-error").forEach((el) => { el.textContent = ""; });
  document.querySelectorAll(".field-wrap input").forEach((el) => {
    el.style.borderColor = "";
  });

  errors.forEach((msg) => {
    const field = Object.keys(FIELD_RULES).find((f) =>
      msg.toLowerCase().startsWith(FIELD_RULES[f].label.toLowerCase()) || msg.startsWith(f)
    );
    if (field) {
      const errEl = document.getElementById(`err-${field}`);
      const input = document.getElementById(field);
      if (errEl) errEl.textContent = msg;
      if (input) input.style.borderColor = "#dc2626";
    }
  });
}
