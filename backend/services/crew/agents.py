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


def data_engineer(llm):
    from crewai import Agent

    return Agent(
        role="Data Engineer",
        goal="Clean raw data and publish dataset_contract.json for downstream agents.",
        backstory=(
            "You own data quality and schema governance. You run ml/clean_data.py (engineer step) "
            "to produce the cleaned CSV and dataset_contract.json with schema, allowed values, "
            "assumptions, and constraints that the Data Scientist must follow."
        ),
        llm=llm,
        verbose=False,
    )


def data_analyst(llm):
    from crewai import Agent

    return Agent(
        role="Data Analyst",
        goal="Produce eda_report.html with visual exploratory analysis of the cleaned dataset.",
        backstory=(
            "You are a healthcare data analyst focused on charts and statistical profiles. "
            "You run the analyst step in ml/data_pipeline.py and deliver eda_report.html only."
        ),
        llm=llm,
        verbose=False,
    )


def insights_analyst(llm):
    from crewai import Agent

    return Agent(
        role="Insights Analyst",
        goal="Translate EDA findings into business-facing insights.md summaries.",
        backstory=(
            "You bridge analytics and stakeholders. You read the cleaned data and EDA context, "
            "then produce insights.md with executive summaries, risk patterns, and modeling guidance."
        ),
        llm=llm,
        verbose=False,
    )


def data_scientist(llm):
    from crewai import Agent

    return Agent(
        role="Data Scientist",
        goal="Train and evaluate models that strictly follow dataset_contract.json.",
        backstory=(
            "You never train on forbidden columns and you validate every feature against "
            "dataset_contract.json before fitting. You use ml/train.py, respect class imbalance, "
            "and produce interpretable models with SHAP support."
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
