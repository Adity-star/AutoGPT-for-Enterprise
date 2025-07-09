 # Writes blog posts from keyword topics

from langgraph.graph import StateGraph, END
from autogpt_core.modules.blog_writer.blog_services.idea_loader import load_blog_idea
from autogpt_core.modules.blog_writer.blog_services.researcher import run_blog_research
from autogpt_core.modules.blog_writer.blog_services.seo_optimizer import extract_keywords
from autogpt_core.modules.blog_writer.blog_services.blog_gen import generate_blog_draft

from pydantic import BaseModel
from typing import Optional

class BlogState(BaseModel):
    idea_data: Optional[dict] = None
    research_summary: Optional[str] = None
    keywords: Optional[list[str]] = None
    blog_draft: Optional[str] = None


def load_idea_node(state: BlogState):
    new_state = load_blog_idea(state)
    return {"state": new_state}

async def research_node(state: BlogState):
    new_state = await run_blog_research(state)
    return {"state": new_state}

async def keyword_node(state: BlogState):
    new_state = await extract_keywords(state)
    return {"state": new_state}

async def draft_node(state: BlogState):
    new_state = await generate_blog_draft(state)
    return {"state": new_state}

def blog_generator_graph() -> BlogState:

    graph = StateGraph(BlogState)

    graph.add_node("load_idea",load_idea_node)
    graph.add_node("research", research_node)
    graph.add_node("keywords", keyword_node)  
    graph.add_node("draft", draft_node)

    graph.set_entry_point("load_idea")

    graph.add_edge("load_idea","research")
    graph.add_edge("research", "keywords")
    graph.add_edge("keywords", "draft")
    graph.add_edge("draft", END)

    return graph.compile()
