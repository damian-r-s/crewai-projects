import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew
import os

os.environ["OPENAI_API_KEY"] = "Your OpenAI API Key"
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"

planner = Agent(
    role="Research & Content Strategist",
    goal="Develop a structured and data-informed content strategy around {topic}",
    backstory=(
        "You are responsible for shaping high-quality content direction. "
        "You analyze emerging trends, audience intent, and key insights related to {topic}. "
        "Your output serves as the foundation for writers and editors to build upon, "
        "ensuring clarity, relevance, and informational value."
    ),
    allow_delegation=False,
    verbose=True
)

writer = Agent(
    role="Editorial Content Creator",
    goal="Produce a clear, engaging, and well-reasoned article about {topic}",
    backstory=(
        "You specialize in transforming structured plans into compelling written content. "
        "You rely on strategic input from the research phase while adding narrative flow, "
        "clarity, and human-like storytelling. "
        "Your writing balances insight, readability, and objectivity."
    ),
    allow_delegation=False,
    verbose=True
)

editor = Agent(
    role="Editorial Quality Reviewer",
    goal="Ensure the article is polished, coherent, and aligned with editorial standards",
    backstory=(
        "You act as the final quality gate before publication. "
        "You review content for clarity, tone consistency, factual alignment, "
        "and readability. Your focus is improving structure and ensuring professional standards."
    ),
    allow_delegation=False,
    verbose=True
)

plan = Task(
    description=(
        "Analyze the topic {topic} and build a structured content direction:\n"
        "- Identify current trends and relevant developments\n"
        "- Define the target audience and their needs\n"
        "- Propose a logical article structure\n"
        "- Suggest SEO-friendly keywords and supporting references"
    ),
    expected_output=(
        "A structured content strategy including audience insights, "
        "article outline, and SEO keyword suggestions."
    ),
    agent=planner,
)

write = Task(
    description=(
        "Using the provided strategy, write a complete article on {topic}:\n"
        "- Maintain a clear and engaging flow\n"
        "- Integrate SEO keywords naturally\n"
        "- Organize content into well-labeled sections\n"
        "- Include an introduction, main discussion, and conclusion\n"
        "- Ensure readability and logical structure"
    ),
    expected_output=(
        "A fully written article in markdown format with structured sections "
        "and 2–3 paragraphs per section."
    ),
    agent=writer,
)

edit = Task(
    description=(
        "Review the generated article and improve it by:\n"
        "- Correcting grammar and stylistic issues\n"
        "- Improving clarity and readability\n"
        "- Ensuring consistent tone and structure\n"
        "- Removing unnecessary repetition or weak phrasing"
    ),
    expected_output=(
        "A refined, publication-ready markdown article with improved clarity "
        "and professional tone."
    ),
    agent=editor
)

crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=True
)

result = crew.kickoff(inputs={"topic": "Artificial Intelligence"})

from IPython.display import Markdown
Markdown(result)