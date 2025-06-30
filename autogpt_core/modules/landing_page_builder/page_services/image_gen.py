import os
import aiohttp
from utils.logger import logging
from autogpt_core.modules.landing_page_builder.page_services.landing_page import LandingPageContent
import base64


# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Constants
HUGGINGFACE_API_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
HUGGINGFACE_API_TOKEN = os.getenv("HF_TOKEN")
DEFAULT_IMAGE_PROMPT = "A modern, sleek landing page concept for a tech startup."

# Construct proper URL
HUGGINGFACE_API_URL = f"https://api-inference.huggingface.co/models/{HUGGINGFACE_API_MODEL}"

def create_image_prompt(content: LandingPageContent) -> str:
    features_formatted = ", ".join(content.features)
    return (
        f"A high-converting landing page for a product called '{content.headline}'. "
        f"Tagline: '{content.subheadline}'. "
        f"Features include: {features_formatted}. "
        "Design should be visually modern, clean, and optimized for tech-savvy users. "
        "Use bright, appealing colors with clear typography."
    )


async def generate_landing_page_images(content: LandingPageContent) -> dict | None:
    prompt = create_image_prompt(content)
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": prompt}

    logger.info(f"Generating image with prompt: {prompt}")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(HUGGINGFACE_API_URL, headers=headers, json=payload) as response:
                if response.status != 200:
                    raise ValueError(f"Hugging Face API error: {response.status}")
                image_bytes = await response.read()
                # Convert image bytes to base64 data URL
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                image_url = f"data:image/png;base64,{image_base64}"

                return {
                    "image_bytes": image_bytes,
                    "image_url": image_url,   # <-- added this
                    "prompt_used": prompt
                }

        except Exception as e:
            logger.warning(f"Image generation failed: {e}. Retrying with fallback prompt.")
            payload["inputs"] = DEFAULT_IMAGE_PROMPT

            try:
                async with session.post(HUGGINGFACE_API_URL, headers=headers, json=payload) as fallback_response:
                    if fallback_response.status != 200:
                        raise ValueError(f"Fallback Hugging Face API error: {fallback_response.status}")
                    image_bytes = await fallback_response.read()
                    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                    image_url = f"data:image/png;base64,{image_base64}"

                    return {
                        "image_bytes": image_bytes,
                        "image_url": image_url,   # <-- added this
                        "prompt_used": DEFAULT_IMAGE_PROMPT
                    }
            except Exception as fallback_error:
                logger.error(f"Fallback image generation failed: {fallback_error}", exc_info=True)
                return None
