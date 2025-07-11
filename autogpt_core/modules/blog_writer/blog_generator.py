 # Writes blog posts from keyword topics

from langgraph.graph import StateGraph, END
from autogpt_core.modules.blog_writer.blog_services.idea_loader import load_blog_idea
from autogpt_core.modules.blog_writer.blog_services.researcher import run_blog_research
from autogpt_core.modules.blog_writer.blog_services.seo_optimizer import extract_keywords
from autogpt_core.modules.blog_writer.blog_services.blog_gen import generate_blog_draft
from autogpt_core.modules.blog_writer.blog_services.blog_state import BlogWriterAgentState
from utils.logger import logging

logger = logger = logging.getLogger(__name__)


def load_idea_node(state: BlogWriterAgentState):
    logger.info(f"load_idea_node received state: {state}")
    new_state = load_blog_idea(state)
    logger.info(f"Idea loaded: {new_state.idea_data.get('idea')}")
    return {"state": new_state}

async def research_node(state: BlogWriterAgentState):
    logger.info(f"research_node received state: {state}")
    if not state.idea_data:
        logger.error("Missing idea_data in state!")
        raise ValueError("Missing idea_data in state!")
    new_state = await run_blog_research(state)
    return {"state": new_state}


async def keyword_node(state: BlogWriterAgentState):
    new_state = await extract_keywords(state)
    return {"state": new_state}

async def draft_node(state: BlogWriterAgentState):
    new_state = await generate_blog_draft(state)
    return {"state": new_state}


def blog_generator_graph():
    graph = StateGraph(BlogWriterAgentState)

    graph.add_node("load_idea", load_idea_node)
    graph.add_node("research", research_node)
    graph.add_node("keywords", keyword_node)
    graph.add_node("draft", draft_node)

    graph.set_entry_point("load_idea")

    graph.add_edge("load_idea", "research")
    graph.add_edge("research", "keywords")
    graph.add_edge("keywords", "draft")
    graph.add_edge("draft", END)

    return graph.compile()

