 # Blog draft generator via GPT and Grammar, tone, style polishing


from autogpt_core.modules.blog_writer.blog_services.blog_state import BlogWriterAgentState
from autogpt_core.core.llm_service import LLMService
from utils.logger import logger

async def generate_blog_draft(state: BlogWriterAgentState) -> BlogWriterAgentState:
    if not (state.idea_data and state.research_summary and state.keywords):
        raise ValueError("Missing data required to generate a blog draft.")

    idea = state.idea_data.get("idea")
    research = state.research_summary
    keywords = ", ".join(state.keywords)

    logger.info(f"Generating blog draft for idea: {idea}")

    prompt = f"""
    You're a professional tech blog writer.

    Write a well-structured blog post based on the following:
    - **Startup Idea**: "{idea}"
    - **Research Summary**: {research}
    - **Target Keywords**: {keywords}

    Requirements:
    - Use engaging, professional tone (similar to TechCrunch or The Verge)
    - Incorporate keywords naturally
    - Use headings, subheadings, bullet points
    - Start with a hook, end with a takeaway
    - Keep length ~600-800 words

    After drafting, revise grammar, tone, and flow to match an expert blog editor’s quality.
    """

    try:
        draft = await LLMService.sync_chat(prompt)
    except Exception as e:
        logger.error(f"Failed to generate blog draft: {e}")
        draft = "Draft generation failed. Manual writing may be required."

    return state.model_copy(update={"blog_draft": draft.strip()})
