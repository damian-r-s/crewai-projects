import warnings
warnings.filterwarnings("ignore")

import os
from crewai import Agent, Task, Crew
from crewai_tools import ScrapeWebsiteTool

os.environ["OPENAI_API_KEY"] = "Your OpenAI API Key"
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"

knowledge_base_tool = ScrapeWebsiteTool(
    website_url="https://docs.crewai.com/en/enterprise/guides/kickoff-crew"
)

technical_support_agent = Agent(
    role="AI Technical Support Engineer",
    goal=(
        "Deliver accurate, practical, and easy-to-understand "
        "technical assistance for platform users"
    ),
    backstory=(
        "You are part of the customer engineering team responsible "
        "for helping enterprise users solve implementation and integration challenges. "
        "Your responsibility is to investigate issues carefully, "
        "reference official documentation when needed, "
        "and provide reliable step-by-step guidance."
    ),
    allow_delegation=False,
    verbose=True
)

review_agent = Agent(
    role="Customer Experience Reviewer",
    goal=(
        "Ensure all support responses are technically correct, "
        "clear, and aligned with professional communication standards"
    ),
    backstory=(
        "You review outgoing technical responses before they are delivered to clients. "
        "Your focus is clarity, completeness, tone consistency, "
        "and ensuring the proposed solution is easy to follow."
    ),
    allow_delegation=False,
    verbose=True
)

support_task = Task(
    description=(
        "A client organization named {customer} submitted the following request:\n\n"
        "{inquiry}\n\n"
        "The request was submitted by {person}. "
        "Analyze the issue carefully and prepare a complete response. "
        "Use available documentation and explain the solution clearly, "
        "including any relevant implementation details or configuration examples."
    ),
    expected_output=(
        "A complete and user-friendly technical response explaining the solution. "
        "The response should include clear guidance, useful references, "
        "and practical implementation recommendations."
    ),
    tools=[knowledge_base_tool],
    agent=technical_support_agent,
)

review_task = Task(
    description=(
        "Review the drafted response prepared for {customer}. "
        "Verify that:\n"
        "- all technical questions are answered\n"
        "- explanations are accurate and understandable\n"
        "- the tone is professional and approachable\n"
        "- the guidance is actionable and well-structured\n\n"
        "Improve weak sections where necessary."
    ),
    expected_output=(
        "A polished final response ready to be shared with the client. "
        "The message should be concise, technically accurate, "
        "and written in a supportive tone."
    ),
    agent=review_agent,
)

crew = Crew(
    agents=[
        technical_support_agent,
        review_agent
    ],
    tasks=[
        support_task,
        review_task
    ],
    verbose=True,
    memory=True
)

inputs = {
    "customer": "DeepLearningAI",
    "person": "Andrew Ng",
    "inquiry": (
        "I would like guidance on initializing a CrewAI workflow "
        "and enabling persistent memory between tasks and agents. "
        "Could you explain the recommended setup?"
    )
}

result = crew.kickoff(inputs=inputs)

from IPython.display import Markdown
Markdown(result)