 # DALL·E integration

from typing import Optional
import os
from utils.logger import logging

from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAI
from openai import AsyncOpenAI
from autogpt_core.modules.landing_page_builder.page_services.landing_page import LandingPageContent

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

openai_client = AsyncOpenAI(api_key =os.environ.get("OPENAI_API_KEY"))

# Fallback prompt
DEFAULT_IMAGE_PROMPT = "A modern, sleek landing page concept for a tech startup."


def create_image_prompt(content: LandingPageContent) -> str:
    """
    Constructs a detailed prompt based on landing page content for DALL·E 3.
    """
    features_formatted = ", ".join(content.features)
    return (
        f"A high-converting landing page for a product called '{content.headline}'. "
        f"Tagline: '{content.subheadline}'. "
        f"Features include: {features_formatted}. "
        "Design should be visually modern, clean, and optimized for tech-savvy users. "
        "Use bright, appealing colors with clear typography."
    )

async def generate_landing_page_images(content: LandingPageContent) -> Optional[dict]:
    """
    Asynchronously generates a DALL·E 3 image based on LandingPageContent.
    """
    prompt = create_image_prompt(content)
    logger.info(f"Generating image with prompt: {prompt}")

    try:
        response = await openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            n=1
        )
        image_url = response.data[0].url if response.data and hasattr(response.data[0], "url") else None
        if not image_url:
            raise ValueError("No image URL returned from DALL·E 3.")
        
        return {
            "image_url": image_url,
            "prompt_used": prompt
        }

    except Exception as e:
        logger.warning(f"Image generation failed with custom prompt: {e}. Retrying with fallback prompt.")

        try:
            fallback_response = await openai_client.images.generate(
                model="dall-e-3",
                prompt=DEFAULT_IMAGE_PROMPT,
                size="1024x1024",
                n=1
            )
            fallback_url = fallback_response.data[0].url if fallback_response.data and hasattr(fallback_response.data[0], "url") else None
            return {
                "image_url": fallback_url,
                "prompt_used": DEFAULT_IMAGE_PROMPT
            }
        except Exception as fallback_error:
            logger.error(f"Fallback image generation also failed: {fallback_error}", exc_info=True)
            return None
        
generate_image_runnable = RunnableLambda(generate_landing_page_images)
