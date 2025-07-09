
from autogpt_core.utils.logger import logger
# from serpapi import GoogleSearch

# Use the logger
logger.info("I am the Best")
logger.error("Error message here")



import asyncio
import logging
from autogpt_core.modules.market_researcher.graph import get_or_generate_market_research_idea

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Main entry point for the market research agent"""
    try:
        results = await get_or_generate_market_research_idea()
        print("Market Research Results:")
        import json
        print(json.dumps(results, indent=2))
    except Exception as e:
        logger.error(f"Error running market research agent: {e}")

if __name__ == "__main__":
    asyncio.run(main())

# import asyncio
# from autogpt_core.modules.landing_page_builder.landing_page_graph import landing_page_graph  
# from autogpt_core.modules.landing_page_builder.landing_page_graph import generate_landing_page_images
# from utils.logger import  logging
# from autogpt_core.modules.landing_page_builder.landing_page_graph import generate_images
# logging.basicConfig(level=logging.INFO)

# async def main():
#     # Compile the graph
#     graph = landing_page_graph()

#     # Run the graph starting with an empty state
#     final_state = await graph.ainvoke({})

#     print("\n✅ Landing page generated successfully!")
#     print("HTML file saved at:", final_state.get("output_path", "Unknown"))

# if __name__ == "__main__":
#     asyncio.run(main())

# from autogpt_core.modules.email_campaign.campaign_manager import email_campaign_graph


# import asyncio

# async def main():
#     initial_state = {}  # Or pre-fill with known state
#     graph = email_campaign_graph()
#     final_state = await graph.ainvoke(initial_state)
#     print(final_state.get("send_status"))

# asyncio.run(main())




# from autogpt_core.modules.blog_writer.blog_generator import blog_generator_graph
# import asyncio

# async def main():
#     graph = blog_generator_graph()

#     initial_state = {} 
#     final_state = await graph.ainvoke({})

#     print("\n✅ Blog generated successfully!\n")
#     print("📝 Final Draft:\n")
#     print(final_state.get("state").blog_draft or "No draft found.")

# if __name__ == "__main__":
#     asyncio.run(main())


from utils.idea_memory import load_ideas_from_db

load_ideas_from_db()

