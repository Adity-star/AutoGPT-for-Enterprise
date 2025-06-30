from utils.idea_memory import load_ideas_from_db
from autogpt_core.modules.landing_page_builder.page_services.content_gen import generate_landing_page_content
from autogpt_core.modules.landing_page_builder.page_services.image_gen import generate_landing_page_images
from autogpt_core.modules.landing_page_builder.page_services.page_builder import building_landing_page
import aiofiles
import os
import base64

from utils.logger import logging
from langgraph.graph import StateGraph

logger = logging.getLogger(__name__)


def fetch_idea(state: dict) -> dict:
    logger.info("Fetching idea from database...")
    ideas = load_ideas_from_db(limit=1)
    if not ideas:
        logger.error("No ideas found in the database.")
        raise ValueError("No ideas found in the database.")
    idea_data = ideas[0]
    logger.info(f"Idea fetched: {idea_data}")
    return {**state, "idea_data": idea_data}


async def generate_content(state: dict) -> dict:
    idea_data  = state["idea_data"]
    logger.info("Generating content for idea...")
    content = await generate_landing_page_content(idea_data )
    logger.info("Content generation complete.")
    return {**state, "content": content}


async def generate_images(state: dict) -> dict:
    content = state["content"]
    logger.info("Generating images for content...")
    image_result = await generate_landing_page_images(content)
    
    if not image_result or "image_url" not in image_result or "image_bytes" not in image_result:
        raise ValueError("Image generation failed — missing image_url or image_bytes.")
    
    logger.info("Image successfully generated.")
    
    return {**state, "image_url": image_result["image_url"], "image_bytes": image_result["image_bytes"]}



async def build_html_page(state: dict) -> dict:
    content = state["content"]
    image_url = state["image_url"]  

    logger.info("Building HTML landing page...")
    html_page = await building_landing_page(content=content, image_url=image_url)

    output_path = "data/output/landing_page.html"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
        await f.write(html_page)
    
    logger.info(f"Landing page saved to {output_path}")
    return {**state, "html": html_page, "output_path": output_path}



def landing_page_graph():
    graph = StateGraph(dict)  

    graph.add_node("fetch_idea", fetch_idea)
    graph.add_node("generate_content", generate_content)
    graph.add_node("generate_images", generate_images)
    graph.add_node("build_html", build_html_page)

    graph.set_entry_point("fetch_idea")
    graph.add_edge("fetch_idea", "generate_content")
    graph.add_edge("generate_content", "generate_images")
    graph.add_edge("generate_images", "build_html")

    graph.set_finish_point("build_html")

    return graph.compile()