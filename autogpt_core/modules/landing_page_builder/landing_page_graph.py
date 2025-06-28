from utils.idea_memory import load_ideas_from_db
from .page_services.content_gen import generate_landing_page_content
from .page_services.image_gen import generate_landing_page_images
from .page_services.page_builder import building_landing_page
from utils.logger import logging
from langgraph.graph import StateGraph

logger = logging.getLogger(__name__)

def fetch_idea(state:dict) -> dict:
    ideas = load_ideas_from_db(limit=1)
    if not ideas:
        raise ValueError("No ideas found in the database.")
    idea_data = ideas[0]
    return {**state,"idea_data":idea_data}

async def generate_content(state: dict) -> dict:
    idea = state['idea_date']["idea"]
    content = await generate_landing_page_content(idea)
    return {**state, "content": content}

async def generate_images(state : dict) -> dict:
    idea = state["idea_date"]["idea"]
    image_url = await generate_landing_page_images(idea)
    return {**state, "image_url": image_url}

async def build_html_page(state: dict) -> dict:
    content = state["content"]
    image_url = state["image_url"]
    html_page = building_landing_page(content=content,image_url=image_url)
     # Optional: save to disk
    with open("output/landing_page.html", "w", encoding="utf-8") as f:
        f.write(html_page)

    return {**state, "html": html_page, "output_path": "output/landing_page.html"}


def landing_page_graph():

    graph = StateGraph()

    # Add each function node
    graph.add_node("fetch_idea", fetch_idea)
    graph.add_node("generate_content", generate_content)
    graph.add_node("generate_image", generate_images)
    graph.add_node("build_html", build_html_page)

     # Define transitions
    graph.set_entry_point("fetch_idea")
    graph.add_edge("fetch_idea", "generate_content")
    graph.add_edge("generate_content", "generate_image")
    graph.add_edge("generate_image", "build_html")

    graph.set_finish_point("build_html")

    return graph.compile()