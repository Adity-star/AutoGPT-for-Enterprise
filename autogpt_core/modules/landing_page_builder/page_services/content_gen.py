 # GPT-based copy generation
import os
import logging
from typing import Dict

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

from modules.landing_page_builder.page_services.landing_page import LandingPageContent
from autogpt_core.config.prompts.prompts import content_generation_prompt
from utils.logger import logging
from dotenv import load_dotenv
load_dotenv()


# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = PydanticOutputParser(pydantic_object=LandingPageContent)

# Step 1: Define prompt using ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_template(
    content_generation_prompt + "\n{format_instructions}"
)

# Step 2: Initialize LLM client
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set.")


llm = ChatGroq(
                api_key=groq_api_key,
                model="meta-llama/llama-4-scout-17b-16e-instruct",
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