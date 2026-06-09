def build_llm(api_key: str):
    from crewai import LLM

    from config import ANTHROPIC_MODEL

    return LLM(model=ANTHROPIC_MODEL, api_key=api_key)


def risk_analyst(llm):
    from crewai import Agent

    return Agent(
        role="Risk Analyst",
        goal="Interpret Alzheimer's risk scores and SHAP factors for a specific patient.",
        backstory=(
            "You are a clinical data analyst who explains model outputs in plain language. "
            "You distinguish modifiable lifestyle factors from fixed demographic factors."
        ),
        llm=llm,
        verbose=False,
    )


def recommendation_advisor(llm):
    from crewai import Agent

    return Agent(
        role="Recommendation Advisor",
        goal="Create prioritized, actionable recommendations to potentially delay cognitive decline.",
        backstory=(
            "You focus on evidence-based lifestyle interventions. "
            "You never diagnose and you avoid generic advice when SHAP data is available."
        ),
        llm=llm,
        verbose=False,
    )


def health_coach(llm):
    from crewai import Agent

    from services.chat_format import CHAT_SYSTEM_PROMPT

    return Agent(
        role="Health Coach",
        goal="Answer follow-up questions in short, plain conversational text.",
        backstory=CHAT_SYSTEM_PROMPT,
        llm=llm,
        verbose=False,
    )
