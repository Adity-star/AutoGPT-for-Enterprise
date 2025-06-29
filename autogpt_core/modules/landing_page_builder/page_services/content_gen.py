 # GPT-based copy generation
import os
import logging
from typing import Dict

from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

from modules.landing_page_builder.page_services.landing_page import LandingPageContent
from autogpt_core.config.prompts.prompts import content_generation_prompt
from utils.logger import logging

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = PydanticOutputParser(pydantic_object=LandingPageContent)

prompt_template = PromptTemplate.from_template(
    content_generation_prompt,
    input_variables=[
        "idea",
        "recommendation",
        "demand_analysis",
        "competition_analysis",
        "unit_economics",
        "format_instructions",
    ]
)

llm = ChatOpenAI(
    model = "gpt-4",
    temperature=0.7,
    openai_api_key = os.environ.get("OPENAPI_API_KEY")
)

# 4. Async function to call from your app
async def generate_landing_page_content(idea_data: Dict[str, str]) -> LandingPageContent:
    logging.info("Generating landing page content....")
    prompt = prompt_template.format_messages(
        idea=idea_data['idea'],
        recommendation=idea_data['recommendation'],
        demand_analysis=idea_data['demand_analysis'],
        competition_analysis=idea_data['competition_analysis'],
        unit_economics=idea_data['unit_economics'],
        format_instructions=parser.get_format_instructions()
    )

    try:
        response = await llm.ainvoke(prompt)
        return parser.parse(response.content)
    except Exception as e:
        # Log or raise a custom exception
        print(f"Error in generate_landing_page_content: {e}")
        raise