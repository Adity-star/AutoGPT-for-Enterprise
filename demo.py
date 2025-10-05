import asyncio
import logging
from autogpt_core.modules.market_researcher.graph import get_or_generate_market_research_idea
from autogpt_core.modules.landing_page_builder.landing_page_graph import landing_page_graph
from autogpt_core.modules.email_campaign.campaign_manager import email_campaign_graph
from autogpt_core.modules.blog_writer.blog_generator import blog_generator_graph
from autogpt_core.modules.blog_writer.blog_services.blog_state import BlogWriterAgentState
from autogpt_core.utils.idea_memory import load_ideas_from_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_market_research():
    """Tests the market research module."""
    try:
        results = await get_or_generate_market_research_idea()
        print("Market Research Results:")
        import json
        print(json.dumps(results, indent=2))
    except Exception as e:
        logger.error(f"Error running market research agent: {e}")

async def test_landing_page():
    """Tests the landing page module."""
    graph = landing_page_graph()
    final_state = await graph.ainvoke({})
    print("\n✅ Landing page generated successfully!")
    print("HTML file saved at:", final_state.get("output_path", "Unknown"))

async def test_email_campaign():
    """Tests the email campaign module."""
    initial_state = {}  # Or pre-fill with known state
    graph = email_campaign_graph()
    final_state = await graph.ainvoke(initial_state)
    print(final_state.get("send_status"))

async def test_blog_writer():
    """Tests the blog writer module."""
    graph = blog_generator_graph()
    initial_state = BlogWriterAgentState()
    final_state = await graph.ainvoke(initial_state)
    print("\n✅ Blog generated successfully!\n")
    print("📝 Final Draft:\n")
    print(final_state.get("state").blog_draft or "No draft found.")

def test_idea_memory():
    """Tests the idea memory module."""
    load_ideas_from_db()

if __name__ == "__main__":
    print("Running demo tests...")
    asyncio.run(test_market_research())
    # asyncio.run(test_landing_page())
    # asyncio.run(test_email_campaign())
    # asyncio.run(test_blog_writer())
    # test_idea_memory()
    print("Demo tests finished.")