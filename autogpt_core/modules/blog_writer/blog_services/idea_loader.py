 # Loads latest idea from DB
from utils.logger import logging
from autogpt_core.utils.idea_memory import load_ideas_from_db
from autogpt_core.modules.blog_writer.blog_services.blog_state import BlogWriterAgentState


logger = logging.getLogger(__name__)

def load_blog_idea(state: BlogWriterAgentState) -> BlogWriterAgentState:
    """
    Load the most recent business idea from the database into the state.
    """
    logger.info("Loading latest business idea for blog writer...")

    ideas = load_ideas_from_db(limit=1)

    if not ideas:
        logger.error("No ideas found in the database.")
        raise ValueError("No validated business idea found in memory.")

    idea_data = ideas[0]
    logger.info(f"Idea loaded: {idea_data['idea']}")

    # Update and return a new state instance
    return state.model_copy(update={"idea_data": idea_data})
