from services.crew.agents import health_coach, recommendation_advisor, risk_analyst


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
