# Uses GPT to generate email copy

from autogpt_core.modules.email_campaign.schema import CampaignInput, CampaignState
from autogpt_core.utils.idea_memory import load_ideas_from_db
from utils.logger import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from autogpt_core.config.prompts.prompts import email_generation_prompt

load_dotenv()

logger = logging.getLogger(__name__)

# Step 2: Initialize LLM client
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set.")


llm = ChatGroq(
                api_key=groq_api_key,
                model="meta-llama/llama-4-scout-17b-16e-instruct",
            )


def load_campaign_content(state: CampaignState) -> CampaignState:
    """
    Load the most recent validated idea from the SQLite database.
    This will be used to generate email content for the campaign.
    """
    logger.info("Loading idea from database...")

    ideas = load_ideas_from_db(limit=1)
    if not ideas:
        logger.error("No ideas found in DB.")
        raise ValueError("No ideas available to use for the email campaign.")

    idea_data = ideas[0]
    logger.info(f"Idea loaded successfully: {idea_data.get('idea')}")

    # Merge with existing state
    return {
        **state,
        "idea_data": idea_data
    }

email_prompt = ChatPromptTemplate.from_messages(email_generation_prompt)


def generate_email_copy(state: CampaignState) -> CampaignState:
    logger.info("Generating marketing email for campaign...")

    idea_data = state.get("idea_data", {})
    product = idea_data.get("idea", "AI-powered hiring assistant for small businesses")
    target_customer = "early-stage startups and small business owners"
    benefits = idea_data.get("recommendation", "Reduces hiring time, automates candidate engagement, and increases recruiter efficiency.")

    prompt = email_prompt.format_messages(
        product=product,
        target_customer=target_customer,
        benefits=benefits,
    )

    response = llm.invoke(prompt)
    email_text = response.content.strip()

    logger.info("Email content generated successfully.")
    
    return {
        **state,
        "email_content": email_text
    }
