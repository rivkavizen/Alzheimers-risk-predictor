from services.crew.agents import (
    data_analyst,
    data_engineer,
    data_scientist,
    health_coach,
    insights_analyst,
    recommendation_advisor,
    risk_analyst,
)


def analyze_risk_task(agent, context: str):
    from crewai import Task
    return Task(
        description=(
            "Analyze this patient's Alzheimer's risk model output:\n{context}\n\n"
            "Summarize the top modifiable risk drivers and protective factors. "
            "Do not diagnose. Keep it under 150 words."
        ),
        expected_output="A concise plain-language risk interpretation.",
        agent=agent,
    )


def generate_recommendations_task(agent, context: str):
    from crewai import Task
    return Task(
        description=(
            "Using this patient context:\n{context}\n\n"
            "Create 4-6 prioritized recommendations as a JSON array. Each item must have: "
            "category, action, evidence, priority (high|medium|low). "
            "Focus on modifiable factors with positive SHAP (risk-increasing). "
            "Do not recommend changing factors that are already protective (negative SHAP). "
            "Return ONLY valid JSON array, no markdown."
        ),
        expected_output="A JSON array of recommendation objects.",
        agent=agent,
    )


def chat_reply_task(agent, context: str, history: str, user_message: str):
    from crewai import Task
    return Task(
        description=(
            "Patient context:\n{context}\n\n"
            "Conversation so far:\n{history}\n\n"
            "User message: {user_message}\n\n"
            "Reply in plain conversational text only (no markdown, no bullets, no headers). "
            "Answer their question in the first sentence. Max ~150 words. Never diagnose."
        ),
        expected_output="A short plain-text chat reply (2-4 paragraphs, no markdown).",
        agent=agent,
    )


def clean_and_contract_task(agent):
    from crewai import Task

    return Task(
        description=(
            "Run the Data Engineer step: ml/data_pipeline.py run_engineer (or ml/clean_data.py). "
            "Deliverables: cleaned CSV and dataset_contract.json with schema, allowed values, "
            "assumptions, and constraints."
        ),
        expected_output=(
            "Confirmation that cleaned data and dataset_contract.json exist with row counts "
            "and cleaning summary."
        ),
        agent=agent,
    )


def produce_eda_report_task(agent):
    from crewai import Task

    return Task(
        description=(
            "Run the Data Analyst step after cleaning. Produce eda_report.html with charts "
            "for class balance, demographics, MMSE, correlations, and lifestyle factors."
        ),
        expected_output="Confirmation that eda_report.html exists and summarizes cohort visuals.",
        agent=agent,
    )


def produce_insights_task(agent):
    from crewai import Task

    return Task(
        description=(
            "Run the Insights Analyst step after EDA. Produce insights.md with executive summary, "
            "key statistical signals, modifiable risk patterns, and recommendations for modeling."
        ),
        expected_output="Confirmation that insights.md exists with business-facing findings.",
        agent=agent,
    )


def train_model_task(agent):
    from crewai import Task

    return Task(
        description=(
            "Read dataset_contract.json before training. Train XGBoost via ml/train.py "
            "using only model_features from the contract. Validate forbidden columns are "
            "excluded and document CV AUC."
        ),
        expected_output="Training metrics JSON with cv_auc and confirmation of contract compliance.",
        agent=agent,
    )


def build_data_pipeline_tasks(llm):
    engineer = data_engineer(llm)
    analyst = data_analyst(llm)
    insights = insights_analyst(llm)
    scientist = data_scientist(llm)

    t1 = clean_and_contract_task(engineer)
    t2 = produce_eda_report_task(analyst)
    t3 = produce_insights_task(insights)
    t4 = train_model_task(scientist)

    t2.context = [t1]
    t3.context = [t1, t2]
    t4.context = [t1, t2, t3]

    return [engineer, analyst, insights, scientist], [t1, t2, t3, t4]


def build_recommendation_tasks(llm, context: str):
    analyst = risk_analyst(llm)
    advisor = recommendation_advisor(llm)
    t1 = analyze_risk_task(analyst, context)
    t2 = generate_recommendations_task(advisor, context)
    t2.context = [t1]
    return [analyst, advisor], [t1, t2]


def build_chat_task(llm, context: str, history: str, user_message: str):
    coach = health_coach(llm)
    task = chat_reply_task(coach, context, history, user_message)
    return [coach], [task]
