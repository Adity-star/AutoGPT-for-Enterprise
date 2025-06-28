 # DALL·E integration

from typing import Optional
import os
from utils.logger import logging
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAI
from openai import AsyncOpenAI
from .landing_page import LandingPageContent

logger = logging.getLogger(__name__)

openai_client = AsyncOpenAI(api_key =os.environ.get("OPENAI_API_KEY"))

# Fallback prompt
DEFAULT_IMAGE_PROMPT = "A modern, sleek landing page concept for a tech startup."


def create_image_prompt(content: LandingPageContent) -> str:
    """
    Constructs a prompt based on landing page content for DALL·E 3.
    """
    features_formatted = ", ".join(content.features)
    return (
        f"A high-converting landing page for a product called '{content.headline}'. "
        f"Tagline: '{content.subheadline}'. "
        f"Features: {features_formatted}. "
        "The page should be visually modern, clean, and optimized for tech-savvy users."
    )

async def generate_landing_page_images(content: LandingPageContent) -> Optional[dict]:
    """
    Asynchronously generates an image using DALL·E 3 through OpenAI's API.
    """
    prompt = create_image_prompt(content)
    logger.info(f"Generating image with prompt: {prompt}")

    try:
        response = await openai_client.images.generate(
            model = "dall-e-e",
            prompt=prompt,
            size = "1024x1024",
            n = 1
        )
        image_url = response.data[0].url
        return {
            "image_url": image_url,
            "prompt_used": prompt
        }
    except Exception as e:
        logger.warning(f"Image generation failed: {e}. Retrying with fallback prompt.")
        try:
            response = await openai_client.images.generate(
                model="dall-e-3",
                prompt=DEFAULT_IMAGE_PROMPT,
                size="1024x1024",
                n=1
            )
            return {
                "image_url": response.data[0].url,
                "prompt_used": DEFAULT_IMAGE_PROMPT
            }
        except Exception as fallback_error:
            logger.error(f"Fallback image generation also failed: {fallback_error}")
            return None
        
generate_image_runnable = RunnableLambda(generate_landing_page_images)
